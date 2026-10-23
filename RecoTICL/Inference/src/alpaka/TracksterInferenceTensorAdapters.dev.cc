
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
          auto destination = outputTracksters.id()[selectedTracksters[row]];

          destination.idProbabilities()(0) = source.id_probabilities0();
          destination.idProbabilities()(1) = source.id_probabilities1();
          destination.idProbabilities()(2) = source.id_probabilities2();
          destination.idProbabilities()(3) = source.id_probabilities3();
          destination.idProbabilities()(4) = source.id_probabilities4();
          destination.idProbabilities()(5) = source.id_probabilities5();
          destination.idProbabilities()(6) = source.id_probabilities6();
          destination.idProbabilities()(7) = source.id_probabilities7();
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
          output.energy()[selected[row]].regressedEnergy() = 0.f;
          output.id()[selected[row]].idProbabilities().setZero();
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
          auto energy = output.energy()[selected[row]];
          energy.regressedEnergy() = PFN && !(energy.rawEnergy() > minEnergy) ? energy.rawEnergy() : predictions[row].energy();
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
