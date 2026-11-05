#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>
#include <xtd/math.h>

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInference.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using namespace cms::alpakatools;

  namespace {

    constexpr uint32_t kBlockSize = 256;

    ALPAKA_FN_HOST_ACC float absoluteEta(float x, float y, float z) {
      return xtd::asinh(xtd::abs(z) / xtd::sqrt(x * x + y * y));
    }

    template <TracksterModel M>
    struct KernelSelect {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc, TracksterModelContext ctx, uint32_t* selected) const {
        for (auto t : uniform_elements(acc, ctx.tracksters.energy().metadata().size()))
          selected[t] = TracksterModelInputs<M>::select(ctx, t) ? 1u : 0u;
      }
    };

    // One thread per selected trackster: the most energetic layer clusters of each layer (at most kMaxClusters, by
    // decreasing energy) and, for the PFN, the trackster quantities.
    template <TracksterModel M, typename Inputs>
    struct KernelFillImage {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    TracksterModelContext ctx,
                                    const uint32_t* selected,
                                    const uint32_t* rows,
                                    uint32_t* trackster,
                                    Inputs inputs) const {
        constexpr auto maxClusters = ::ticl::kMaxClusters;
        constexpr auto nLayers = ::ticl::kMaxLayers;

        for (auto t : uniform_elements(acc, ctx.tracksters.energy().metadata().size())) {
          if (selected[t] == 0u)
            continue;
          const auto row = rows[t] - 1u;
          trackster[row] = t;
          auto input = inputs[row];
          const auto vertices = ctx.tracksters.vertices()[t];

          if constexpr (M == TracksterModel::PFN) {
            const auto barycenter = ctx.tracksters.barycenter()[t];
            const float x = barycenter.x(), y = barycenter.y(), z = xtd::abs(barycenter.z());
            input.rawEnergy() = ctx.tracksters.energy()[t].rawEnergy();
            input.rawEmEnergy() = ctx.tracksters.energy()[t].rawEmEnergy();
            input.barycenterX() = x;
            input.barycenterY() = y;
            input.barycenterAbsZ() = z;
            input.barycenterAbsEta() = absoluteEta(x, y, z);
            input.barycenterPhi() = xtd::atan2(y, x);
          }

          // the vertices of the most energetic layer clusters of each layer, by decreasing energy (equal energies in
          // any order)
          uint32_t kept[nLayers][maxClusters];
          uint8_t counts[nLayers] = {};
          for (auto k = 0u; k < vertices.size(); ++k) {
            const auto i = vertices[k];
            if (M == TracksterModel::DNN and detail::isBarrel(ctx.layerClusters.indexes()[i].seedID()))
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

          // the trackster stores no vertex multiplicity: the layer cluster energy is used as is
          for (auto layer = 0u; layer < nLayers; ++layer) {
            for (auto slot = 0u; slot < counts[layer]; ++slot) {
              const auto i = vertices[kept[layer][slot]];
              const auto position = ctx.layerClusters.position()[i];
              const float x = position.x(), y = position.y(), z = position.z();
              input.energy()(layer, slot) = ctx.layerClusters.energy()[i].energy();
              input.absEta()(layer, slot) = absoluteEta(x, y, z);
              input.phi()(layer, slot) = xtd::atan2(y, x);
              if constexpr (M == TracksterModel::PFN) {
                input.x()(layer, slot) = x;
                input.y()(layer, slot) = y;
                input.absZ()(layer, slot) = xtd::abs(z);
                input.cells()(layer, slot) = static_cast<float>(position.cells());
              }
            }
          }
        }
      }
    };

    // One thread per selected trackster: the inputs of TracksterInferenceByShowerShape from the converted trackster,
    // whose principal axis points away from the interaction point, as the SoA one, and whose sigmas are those of the
    // SoA (sqrt(sigma^2) = sigma for sigma >= 0).
    template <typename Inputs>
    struct KernelFillShowerShape {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    TracksterModelContext ctx,
                                    const uint32_t* selected,
                                    const uint32_t* rows,
                                    uint32_t* trackster,
                                    Inputs inputs) const {
        for (auto t : uniform_elements(acc, ctx.tracksters.energy().metadata().size())) {
          if (selected[t] == 0u)
            continue;
          const auto row = rows[t] - 1u;
          trackster[row] = t;
          auto input = inputs[row];
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
      }
    };

    struct KernelWrite {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const uint32_t* trackster,
                                    uint32_t nRows,
                                    bool hasPID,
                                    ::ticl::TracksterPIDScoresLayout<>::ConstView pid,
                                    bool hasEnergy,
                                    ::ticl::TracksterEnergyLayout<>::ConstView energy,
                                    bool reset,
                                    bool keepLowRawEnergy,
                                    float minEnergy,
                                    ::ticl::TracksterSoA::View output) const {
        for (auto row : uniform_elements(acc, nRows)) {
          const auto t = trackster[row];
          auto probabilities = output.id()[t].idProbabilities();
          auto energies = output.energy()[t];
          if (reset) {
            probabilities.setZero();
            energies.regressedEnergy() = 0.f;
          }
          if (hasEnergy)
            energies.regressedEnergy() = keepLowRawEnergy and not(energies.rawEnergy() > minEnergy)
                                             ? energies.rawEnergy()
                                             : energy[row].energy();
          if (hasPID) {
            const auto p = pid[row];
            const float values[8] = {p.probability0(),
                                     p.probability1(),
                                     p.probability2(),
                                     p.probability3(),
                                     p.probability4(),
                                     p.probability5(),
                                     p.probability6(),
                                     p.probability7()};
            for (int k = 0; k < 8; ++k)
              probabilities(k) = values[k];
          }
        }
      }
    };

  }  // namespace

  template <TracksterModel M>
  void TracksterInferenceAlgo<M>::select(Queue& queue,
                                         const TracksterModelContext& ctx,
                                         uint32_t* selected,
                                         uint32_t* rows) {
    const uint32_t n = ctx.tracksters.energy().metadata().size();
    const auto workDiv = make_workdiv<Acc1D>(divide_up_by(n, kBlockSize), kBlockSize);
    alpaka::exec<Acc1D>(queue, workDiv, KernelSelect<M>{}, ctx, selected);
    iterativePrefixScan<Acc1D>(selected, rows, n, queue);
  }

  template <TracksterModel M>
  void TracksterInferenceAlgo<M>::fill(Queue& queue,
                                       const TracksterModelContext& ctx,
                                       const uint32_t* selected,
                                       const uint32_t* rows,
                                       uint32_t* trackster,
                                       typename TracksterModelInputs<M>::Layout::View inputs) {
    const uint32_t n = ctx.tracksters.energy().metadata().size();
    const auto workDiv = make_workdiv<Acc1D>(divide_up_by(n, kBlockSize), kBlockSize);
    using View = typename TracksterModelInputs<M>::Layout::View;
    if constexpr (M == TracksterModel::ShowerShape)
      alpaka::exec<Acc1D>(queue, workDiv, KernelFillShowerShape<View>{}, ctx, selected, rows, trackster, inputs);
    else
      alpaka::exec<Acc1D>(queue, workDiv, KernelFillImage<M, View>{}, ctx, selected, rows, trackster, inputs);
  }

  template <TracksterModel M>
  void TracksterInferenceAlgo<M>::write(Queue& queue,
                                        const uint32_t* trackster,
                                        uint32_t nRows,
                                        const TracksterPIDScoresDevice* pid,
                                        const TracksterEnergyDevice* energy,
                                        float minEnergy,
                                        ::ticl::TracksterSoA::View output) {
    const auto workDiv = make_workdiv<Acc1D>(divide_up_by(nRows, kBlockSize), kBlockSize);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        KernelWrite{},
                        trackster,
                        nRows,
                        pid != nullptr,
                        pid ? pid->const_view() : ::ticl::TracksterPIDScoresLayout<>::ConstView{},
                        energy != nullptr,
                        energy ? energy->const_view() : ::ticl::TracksterEnergyLayout<>::ConstView{},
                        TracksterModelInputs<M>::regression,
                        M == TracksterModel::PFN,
                        minEnergy,
                        output);
  }

  template struct TracksterInferenceAlgo<TracksterModel::CNN>;
  template struct TracksterInferenceAlgo<TracksterModel::DNN>;
  template struct TracksterInferenceAlgo<TracksterModel::PFN>;
  template struct TracksterInferenceAlgo<TracksterModel::ShowerShape>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl
