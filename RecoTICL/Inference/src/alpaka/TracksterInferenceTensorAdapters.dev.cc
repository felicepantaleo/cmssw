
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceTensorAdapters.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {

    struct ScatterPIDToTrackstersKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    uint32_t const* selectedTracksters,
                                    ::ticl::TracksterInferencePIDScoresSoA::ConstView scores,
                                    ::ticl::TracksterSoA::View outputTracksters,
                                    uint32_t total) const {
        for (auto row : cms::alpakatools::uniform_elements(acc, total)) {
          const auto source = scores[row];
          auto destination = outputTracksters.tracksters()[selectedTracksters[row]];

          destination.id_probabilities0() = source.id_probabilities0();
          destination.id_probabilities1() = source.id_probabilities1();
          destination.id_probabilities2() = source.id_probabilities2();
          destination.id_probabilities3() = source.id_probabilities3();
          destination.id_probabilities4() = source.id_probabilities4();
          destination.id_probabilities5() = source.id_probabilities5();
          destination.id_probabilities6() = source.id_probabilities6();
          destination.id_probabilities7() = source.id_probabilities7();
        }
      }
    };

    struct ResetPredictionsKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    uint32_t const* selected,
                                    ::ticl::TracksterSoA::View output,
                                    uint32_t total) const {
        for (auto row : cms::alpakatools::uniform_elements(acc, total)) {
          auto ts = output.tracksters()[selected[row]];
          ts.regressed_energy() = 0.f;
          ts.id_probabilities0() = ts.id_probabilities1() = ts.id_probabilities2() = ts.id_probabilities3() = 0.f;
          ts.id_probabilities4() = ts.id_probabilities5() = ts.id_probabilities6() = ts.id_probabilities7() = 0.f;
        }
      }
    };

    template <bool PFN>
    struct ScatterEnergyKernel {
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                    uint32_t const* selected,
                                    ::ticl::TracksterInferenceEnergySoA::ConstView predictions,
                                    ::ticl::TracksterSoA::View output,
                                    uint32_t total,
                                    float minEnergy) const {
        for (auto row : cms::alpakatools::uniform_elements(acc, total)) {
          auto ts = output.tracksters()[selected[row]];
          ts.regressed_energy() = PFN && !(ts.raw_energy() > minEnergy) ? ts.raw_energy() : predictions[row].energy();
        }
      }
    };

  }  // namespace

  void PIDOutputAdapter::applyResults(Queue& queue,
                                      uint32_t const* selectedTracksters,
                                      ModelOutputDevice const& predictions,
                                      PublishedOutputDevice& published,
                                      uint32_t total,
                                      float) {
    if (total == 0u) {
      return;
    }

    constexpr auto elementsPerBlock = 256u;
    const auto blocks = cms::alpakatools::divide_up_by(total, elementsPerBlock);
    const auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, elementsPerBlock);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        ScatterPIDToTrackstersKernel{},
                        selectedTracksters,
                        predictions.const_view(),
                        published.view(),
                        total);
  }

  void resetTracksterPredictions(Queue& queue, uint32_t const* selected, ticl::TracksterDevice& output, uint32_t total) {
    if (total == 0u)
      return;
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(total, 256u), 256u);
    alpaka::exec<Acc1D>(queue, workDiv, ResetPredictionsKernel{}, selected, output.view(), total);
  }

  template <bool PFN>
  void EnergyOutputAdapter<PFN>::applyResults(Queue& queue,
                                              uint32_t const* selected,
                                              ModelOutputDevice const& predictions,
                                              PublishedOutputDevice& output,
                                              uint32_t total,
                                              float minEnergy) {
    if (total == 0u)
      return;
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(total, 256u), 256u);
    alpaka::exec<Acc1D>(
        queue, workDiv, ScatterEnergyKernel<PFN>{}, selected, predictions.const_view(), output.view(), total, minEnergy);
  }
  template struct EnergyOutputAdapter<false>;
  template struct EnergyOutputAdapter<true>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
