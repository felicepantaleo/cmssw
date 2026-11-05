#ifndef RecoTICL_Inference_interface_alpaka_TracksterInference_h
#define RecoTICL_Inference_interface_alpaka_TracksterInference_h

#include <cstdint>

#include <xtd/math.h>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "RecoTICL/Inference/interface/TracksterInferenceSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using InferenceTensors = cms::torch::alpakatools::BatchedTensorCollection<Queue>;
  using TracksterPIDScoresDevice = PortableCollection<::ticl::TracksterPIDScoresLayout<>>;
  using TracksterEnergyDevice = PortableCollection<::ticl::TracksterEnergyLayout<>>;

  // The trackster models on the device. Each selects the tracksters it identifies, fills their inputs from the
  // tracksters (and their layer clusters), and gives their PID and, with a regression, their energy.
  enum class TracksterModel { CNN, DNN, PFN, ShowerShape };

  template <TracksterModel M>
  struct TracksterModelInputs;

  struct TracksterModelContext {
    ::ticl::TracksterSoA::ConstView tracksters;
    reco::CaloClusterDeviceCollection::ConstView layerClusters;
    uint32_t layersPerEndcap;
    float minEnergy;
  };

  namespace detail {

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float absoluteEta(float x, float y, float z) {
      return xtd::asinh(xtd::abs(z) / xtd::sqrt(x * x + y * y));
    }

    // Same seed-ID predicate as ticlgeom::Tools::isBarrel()
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool isBarrel(DetId id) {
      return (id.det() == DetId::Ecal && id.subdetId() == 1) ||
             (id.det() == DetId::Hcal && (id.subdetId() == 1 || id.subdetId() == 3));
    }

    // Which layer clusters of a trackster an image model uses
    struct AllClusters {
      ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool accept(const TracksterModelContext&, uint32_t) { return true; }
    };

    struct EndcapClusters {
      ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool accept(const TracksterModelContext& ctx, uint32_t cluster) {
        return not isBarrel(ctx.layerClusters.indexes()[cluster].seedID());
      }
    };

    // Whether the summed energy of the accepted layer clusters of trackster t reaches minEnergy
    template <typename Accept>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE bool reachesMinEnergy(const TracksterModelContext& ctx, uint32_t t) {
      const auto vertices = ctx.tracksters.vertices()[t];
      bool reached = false;
      float energy = 0.f;
      for (auto k = 0u; k < vertices.size() and not reached; ++k) {
        if (not Accept::accept(ctx, vertices[k]))
          continue;
        energy += ctx.layerClusters.energy()[vertices[k]].energy();
        reached = energy >= ctx.minEnergy;
      }
      return reached;
    }

    // The most energetic accepted layer clusters of each layer of trackster t (at most kMaxClusters, by decreasing
    // energy): write(layer, slot, cluster) stores each of them in the image.
    template <typename Accept, typename Write>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void fillImage(const TracksterModelContext& ctx, uint32_t t, Write write) {
      constexpr auto maxClusters = ::ticl::kMaxClusters;
      constexpr auto nLayers = ::ticl::kMaxLayers;
      const auto vertices = ctx.tracksters.vertices()[t];

      // the vertices of the most energetic layer clusters of each layer, by decreasing energy (equal energies in any
      // order)
      uint32_t kept[nLayers][maxClusters];
      uint8_t counts[nLayers] = {};
      for (auto k = 0u; k < vertices.size(); ++k) {
        const auto i = vertices[k];
        if (not Accept::accept(ctx, i))
          continue;
        const auto layer = static_cast<uint32_t>(ctx.layerClusters.position()[i].layer()) % ctx.layersPerEndcap;
        if (layer >= nLayers)
          continue;
        const auto energy = ctx.layerClusters.energy()[i].energy();
        auto& count = counts[layer];
        auto slot = 0;
        while (slot < count and ctx.layerClusters.energy()[vertices[kept[layer][slot]]].energy() >= energy)
          ++slot;
        if (slot >= maxClusters)
          continue;
        // insert, dropping the least energetic one when full
        for (auto j = count < maxClusters ? count : maxClusters - 1; j > slot; --j)
          kept[layer][j] = kept[layer][j - 1];
        kept[layer][slot] = k;
        if (count < maxClusters)
          ++count;
      }

      for (auto layer = 0u; layer < nLayers; ++layer)
        for (auto slot = 0u; slot < counts[layer]; ++slot)
          write(layer, slot, vertices[kept[layer][slot]]);
    }

    // The image of the CNN and the DNN. The trackster stores no vertex multiplicity: the layer cluster energy is used
    // as is.
    template <typename Accept, typename Input>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE void fillEnergyEtaPhi(const TracksterModelContext& ctx, uint32_t t, Input input) {
      fillImage<Accept>(ctx, t, [&](uint32_t layer, uint32_t slot, uint32_t cluster) {
        const auto position = ctx.layerClusters.position()[cluster];
        const float x = position.x(), y = position.y(), z = position.z();
        input.energy()(layer, slot) = ctx.layerClusters.energy()[cluster].energy();
        input.absEta()(layer, slot) = absoluteEta(x, y, z);
        input.phi()(layer, slot) = xtd::atan2(y, x);
      });
    }

  }  // namespace detail

  // CNN: tracksters of at least minEnergy summed layer-cluster energy. PID only.
  template <>
  struct TracksterModelInputs<TracksterModel::CNN> {
    using Layout = ::ticl::TracksterImageInputsLayout<>;
    static constexpr bool regression = false;
    static constexpr bool layerClusters = true;
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool select(const TracksterModelContext& ctx, uint32_t t) {
      return detail::reachesMinEnergy<detail::AllClusters>(ctx, t);
    }
    template <typename Input>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static void fill(const TracksterModelContext& ctx, uint32_t t, Input input) {
      detail::fillEnergyEtaPhi<detail::AllClusters>(ctx, t, input);
    }
    template <typename Records>
    static void add(InferenceTensors& tensors, Records const& r, uint32_t batch) {
      tensors.addBatched<Layout>("input", batch, r.energy(), r.absEta(), r.phi());
    }
  };

  // DNN: tracksters of at least minEnergy summed layer-cluster energy & in Endcap
  template <>
  struct TracksterModelInputs<TracksterModel::DNN> : TracksterModelInputs<TracksterModel::CNN> {
    static constexpr bool regression = true;
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool select(const TracksterModelContext& ctx, uint32_t t) {
      return detail::reachesMinEnergy<detail::EndcapClusters>(ctx, t);
    }
    template <typename Input>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static void fill(const TracksterModelContext& ctx, uint32_t t, Input input) {
      detail::fillEnergyEtaPhi<detail::EndcapClusters>(ctx, t, input);
    }
  };

  // PFN: tracksters without barrel layer clusters, with a regression, which keeps the raw energy up to minEnergy.
  template <>
  struct TracksterModelInputs<TracksterModel::PFN> {
    using Layout = ::ticl::TracksterPFNInputsLayout<>;
    static constexpr bool regression = true;
    static constexpr bool layerClusters = true;

    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool select(const TracksterModelContext& ctx, uint32_t t) {
      const auto vertices = ctx.tracksters.vertices()[t];
      bool noBarrel = true;
      for (auto k = 0u; k < vertices.size() and noBarrel; ++k)
        noBarrel = not detail::isBarrel(ctx.layerClusters.indexes()[vertices[k]].seedID());
      return noBarrel;
    }

    template <typename Input>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static void fill(const TracksterModelContext& ctx, uint32_t t, Input input) {
      const auto barycenter = ctx.tracksters.barycenter()[t];
      const float bx = barycenter.x(), by = barycenter.y(), bz = xtd::abs(barycenter.z());
      input.rawEnergy() = ctx.tracksters.energy()[t].rawEnergy();
      input.rawEmEnergy() = ctx.tracksters.energy()[t].rawEmEnergy();
      input.barycenterX() = bx;
      input.barycenterY() = by;
      input.barycenterAbsZ() = bz;
      input.barycenterAbsEta() = detail::absoluteEta(bx, by, bz);
      input.barycenterPhi() = xtd::atan2(by, bx);

      // the trackster stores no vertex multiplicity: the layer cluster energy is used as is
      detail::fillImage<detail::AllClusters>(ctx, t, [&](uint32_t layer, uint32_t slot, uint32_t cluster) {
        const auto position = ctx.layerClusters.position()[cluster];
        const float x = position.x(), y = position.y(), z = position.z();
        input.energy()(layer, slot) = ctx.layerClusters.energy()[cluster].energy();
        input.absEta()(layer, slot) = detail::absoluteEta(x, y, z);
        input.phi()(layer, slot) = xtd::atan2(y, x);
        input.x()(layer, slot) = x;
        input.y()(layer, slot) = y;
        input.absZ()(layer, slot) = xtd::abs(z);
        input.cells()(layer, slot) = static_cast<float>(position.cells());
      });
    }

    template <typename Records>
    static void add(InferenceTensors& tensors, Records const& r, uint32_t batch) {
      tensors.addBatched<Layout>("input", batch, r.energy(), r.absEta(), r.phi(), r.x(), r.y(), r.absZ(), r.cells());
      tensors.addBatched<Layout>("input_tr_features",
                                 batch,
                                 r.rawEnergy(),
                                 r.rawEmEnergy(),
                                 r.barycenterX(),
                                 r.barycenterY(),
                                 r.barycenterAbsZ(),
                                 r.barycenterAbsEta(),
                                 r.barycenterPhi());
    }
  };

  // Shower shape: tracksters of at least minEnergy raw energy, from their quantities and energy profile. PID only.
  template <>
  struct TracksterModelInputs<TracksterModel::ShowerShape> {
    using Layout = ::ticl::TracksterShowerShapeInputsLayout<>;
    static constexpr bool regression = false;
    static constexpr bool layerClusters = false;
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool select(const TracksterModelContext& ctx, uint32_t t) {
      return not(ctx.tracksters.energy()[t].rawEnergy() < ctx.minEnergy);
    }

    template <typename Input>
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static void fill(const TracksterModelContext& ctx, uint32_t t, Input input) {
      const auto energy = ctx.tracksters.energy()[t];
      const auto b = ctx.tracksters.barycenter()[t];
      const auto pca = ctx.tracksters.pca()[t];
      const auto profile = ctx.tracksters.profile()[t];
      // ROOT's eta, asinh(z / rho) for |eta| < 4.7
      const float eta = xtd::asinh(b.z() / xtd::sqrt(b.x() * b.x() + b.y() * b.y()));
      const auto axis = pca.eigenvectors().col(0);
      const float axisEta = xtd::asinh(axis(2) / xtd::fmax(xtd::hypot(axis(0), axis(1)), 1e-6f));
      float eigenvalues[3];
      for (int k = 0; k < 3; ++k)
        eigenvalues[k] = xtd::fmax(pca.eigenvalues()(k), 1e-6f);

      input.logRawEnergy() = xtd::log1p(energy.rawEnergy());
      input.emFraction() = energy.rawEmEnergy() / xtd::fmax(energy.rawEnergy(), 1e-6f);
      input.barycenterAbsEta() = xtd::abs(eta);
      input.barycenterAbsZ() = xtd::abs(b.z());
      input.logNLayerClusters() = xtd::log1p(static_cast<float>(ctx.tracksters.vertices()[t].size()));
      input.logEigenvalue0() = xtd::log(eigenvalues[0]);
      input.logEigenvalue1() = xtd::log(eigenvalues[1]);
      input.logEigenvalue2() = xtd::log(eigenvalues[2]);
      input.eigenvalue0Fraction() = eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);
      input.sigma0() = pca.sigmas()(0);
      input.sigma1() = pca.sigmas()(1);
      input.sigma2() = pca.sigmas()(2);
      input.sigmaPCA0() = pca.sigmasPCA()(0);
      input.sigmaPCA1() = pca.sigmasPCA()(1);
      input.sigmaPCA2() = pca.sigmasPCA()(2);
      input.axisDeltaAbsEta() = xtd::abs(axisEta - eta);
      input.maxEnergyFraction() = profile.maxEnergyFraction();
      input.core2EnergyFraction() = profile.core2EnergyFraction();
      input.core5EnergyFraction() = profile.core5EnergyFraction();
      input.rmsRadius() = profile.rmsRadius();
      input.minAbsZ() = profile.minAbsZ();
      input.lengthZ() = profile.lengthZ();
    }
    template <typename Records>
    static void add(InferenceTensors& tensors, Records const& r, uint32_t batch) {
      tensors.addBatched<Layout>("input",
                                 batch,
                                 r.logRawEnergy(),
                                 r.emFraction(),
                                 r.barycenterAbsEta(),
                                 r.barycenterAbsZ(),
                                 r.logNLayerClusters(),
                                 r.logEigenvalue0(),
                                 r.logEigenvalue1(),
                                 r.logEigenvalue2(),
                                 r.eigenvalue0Fraction(),
                                 r.sigma0(),
                                 r.sigma1(),
                                 r.sigma2(),
                                 r.sigmaPCA0(),
                                 r.sigmaPCA1(),
                                 r.sigmaPCA2(),
                                 r.axisDeltaAbsEta(),
                                 r.maxEnergyFraction(),
                                 r.core2EnergyFraction(),
                                 r.core5EnergyFraction(),
                                 r.rmsRadius(),
                                 r.minAbsZ(),
                                 r.lengthZ());
    }
  };

  template <TracksterModel M>
  using TracksterModelInputsDevice = PortableCollection<typename TracksterModelInputs<M>::Layout>;

  // The device work of a model. The layer clusters are only read by the models that use them.
  template <TracksterModel M>
  struct TracksterInferenceAlgo {
    // selected[t] whether model M identifies trackster t, and rows[t] the number of selected tracksters up to t
    static void select(Queue& queue, const TracksterModelContext& ctx, uint32_t* selected, uint32_t* rows);

    // the inputs of the selected tracksters, and trackster[row] the trackster of each row
    static void fill(Queue& queue,
                     const TracksterModelContext& ctx,
                     const uint32_t* selected,
                     const uint32_t* rows,
                     uint32_t* trackster,
                     typename TracksterModelInputs<M>::Layout::View inputs);

    // The predictions of the rows into their tracksters (pid and energy may be null). A model with a regression zeroes
    // the PID and the regressed energy of the selected tracksters first, and the PFN keeps the raw energy up to
    // minEnergy.
    static void write(Queue& queue,
                      const uint32_t* trackster,
                      uint32_t nRows,
                      const TracksterPIDScoresDevice* pid,
                      const TracksterEnergyDevice* energy,
                      float minEnergy,
                      ::ticl::TracksterSoA::View output);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif
