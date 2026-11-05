#ifndef RecoTICL_Inference_interface_alpaka_TracksterInference_h
#define RecoTICL_Inference_interface_alpaka_TracksterInference_h

#include <cstdint>

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
    template <typename Records>
    static void add(InferenceTensors& tensors, Records const& r, uint32_t batch) {
      tensors.addBatched<Layout>("input", batch, r.energy(), r.absEta(), r.phi());
    }
  };

  // DNN: tracksters of at least minEnergy summed layer-cluster energy & in Endcap. PID only 
  template <>
  struct TracksterModelInputs<TracksterModel::DNN> : TracksterModelInputs<TracksterModel::CNN> {
    static constexpr bool regression = true;
    // hides the select of the CNN, which would otherwise be inherited
    ALPAKA_FN_ACC ALPAKA_FN_INLINE static bool select(const TracksterModelContext& ctx, uint32_t t) {
      return detail::reachesMinEnergy<detail::EndcapClusters>(ctx, t);
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
