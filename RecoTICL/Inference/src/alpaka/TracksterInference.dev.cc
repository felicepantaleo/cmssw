#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInference.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using namespace cms::alpakatools;

  namespace {

    constexpr uint32_t kBlockSize = 256;

    template <TracksterModel M>
    struct KernelSelect {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc, TracksterModelContext ctx, uint32_t* selected) const {
        for (auto t : uniform_elements(acc, ctx.tracksters.energy().metadata().size()))
          selected[t] = M::select(ctx, t) ? 1u : 0u;
      }
    };

    // One thread per selected trackster: its row, whose inputs the model fills
    template <TracksterModel M>
    struct KernelFill {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    TracksterModelContext ctx,
                                    const uint32_t* selected,
                                    const uint32_t* rows,
                                    uint32_t* trackster,
                                    typename M::Layout::View inputs) const {
        for (auto t : uniform_elements(acc, ctx.tracksters.energy().metadata().size())) {
          if (selected[t] == 0u)
            continue;
          const auto row = rows[t] - 1u;
          trackster[row] = t;
          M::fill(ctx, t, inputs[row]);
        }
      }
    };

    template <TracksterModel M>
    struct KernelWrite {
      ALPAKA_FN_ACC void operator()(const Acc1D& acc,
                                    const uint32_t* trackster,
                                    uint32_t nRows,
                                    bool hasPID,
                                    ::ticl::TracksterPIDScoresLayout<>::ConstView pid,
                                    bool hasEnergy,
                                    ::ticl::TracksterEnergyLayout<>::ConstView energy,
                                    float minEnergy,
                                    ::ticl::TracksterSoA::View output) const {
        for (auto row : uniform_elements(acc, nRows)) {
          const auto t = trackster[row];
          auto probabilities = output.id()[t].idProbabilities();
          if constexpr (M::regression) {
            auto energies = output.energy()[t];
            probabilities.setZero();
            energies.regressedEnergy() = 0.f;
            if (hasEnergy)
              energies.regressedEnergy() = M::regressedEnergy(energy[row].energy(), energies.rawEnergy(), minEnergy);
          }
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
                                       typename M::Layout::View inputs) {
    const uint32_t n = ctx.tracksters.energy().metadata().size();
    const auto workDiv = make_workdiv<Acc1D>(divide_up_by(n, kBlockSize), kBlockSize);
    alpaka::exec<Acc1D>(queue, workDiv, KernelFill<M>{}, ctx, selected, rows, trackster, inputs);
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
                        KernelWrite<M>{},
                        trackster,
                        nRows,
                        pid != nullptr,
                        pid ? pid->const_view() : ::ticl::TracksterPIDScoresLayout<>::ConstView{},
                        energy != nullptr,
                        energy ? energy->const_view() : ::ticl::TracksterEnergyLayout<>::ConstView{},
                        minEnergy,
                        output);
  }

  template struct TracksterInferenceAlgo<TracksterCNN>;
  template struct TracksterInferenceAlgo<TracksterDNN>;
  template struct TracksterInferenceAlgo<TracksterPFN>;
  template struct TracksterInferenceAlgo<TracksterShowerShape>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl
