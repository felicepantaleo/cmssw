#include <cstdint>
#include <limits>
#include <string>
#include <Eigen/Core>

#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceRunner.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename Adapter>
  TracksterInferenceRunner<Adapter>::TracksterInferenceRunner(std::string const& modelPath,
                                                              uint32_t batchSize,
                                                              bool convertToFP16,
                                                              uint32_t warmupIterations)
      : model_{modelPath},
        dtype_{convertToFP16 ? ::torch::kHalf : ::torch::kFloat},
        batchSize_{batchSize},
        warmupIterations_{warmupIterations} {
    if (batchSize_ == 0u || batchSize_ > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      throw cms::Exception("TracksterInferenceRunner") << "batchSize must be positive and fit the SoA int32_t size";
    }
    // Set precision on CPU as well; AlpakaModel moves to the queue device at forward().
    model_.to(dtype_);
  }

  template <typename Adapter>
  void TracksterInferenceRunner<Adapter>::warmup(Queue& queue) {
    if (warmupIterations_ == 0) {
      return;
    }

    InputDevice warmupInput(queue, batchSize_);
    ModelOutputDevice warmupOutput(queue, batchSize_);
    auto inputRecords = warmupInput.view().records();
    auto outputRecords = warmupOutput.view().records();

    for (uint32_t iteration = 0; iteration < warmupIterations_; ++iteration) {
      InferenceTensors inputs, outputs;
      Adapter::registerInputs(inputs, inputRecords, batchSize_);
      Adapter::registerOutputs(outputs, outputRecords, batchSize_);
      model_.forward(queue, inputs, outputs, dtype_);
    }
  }

  template <typename Adapter>
  void TracksterInferenceRunner<Adapter>::runInto(Queue& queue,
                                                  InputDevice const& features,
                                                  PublishedOutputDevice& output,
                                                  float minEnergy) {
    const auto total = static_cast<int32_t>(features.const_view().metadata().size());

    ModelOutputDevice predictions(queue, total);
    predictions.zeroInitialise(queue);
    auto featureRecords = features.const_view().records();
    auto predictionRecords = predictions.view().records();
    InferenceTensors inputs, outputs;
    Adapter::registerInputs(inputs, featureRecords, batchSize_);
    Adapter::registerOutputs(outputs, predictionRecords, batchSize_);
    model_.forward(queue, inputs, outputs, dtype_);
    Adapter::applyResults(queue, featureRecords, predictions, output, static_cast<uint32_t>(total), minEnergy);
  }

  template class TracksterInferenceRunner<CNNPIDAdapter>;
  template class TracksterInferenceRunner<DNNPIDAdapter>;
  template class TracksterInferenceRunner<PFNPIDAdapter>;
  template class TracksterInferenceRunner<DNNEnergyAdapter>;
  template class TracksterInferenceRunner<PFNEnergyAdapter>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
