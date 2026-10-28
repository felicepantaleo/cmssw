#include <cstdint>
#include <limits>
#include <string>

#include <torch/csrc/jit/runtime/graph_executor.h>

#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNInferenceRunner.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  SuperclusteringDNNInferenceRunner::SuperclusteringDNNInferenceRunner(std::string const& modelPath,
                                                                       unsigned int featureCount,
                                                                       uint32_t batchSize,
                                                                       bool convertToFP16,
                                                                       uint32_t warmupIterations)
      : model_{modelPath},
        dtype_{convertToFP16 ? ::torch::kHalf : ::torch::kFloat},
        featureCount_{featureCount},
        batchSize_{batchSize},
        warmupIterations_{warmupIterations} {
    // TensorCollection takes int sizes
    if (batchSize_ == 0 || batchSize_ > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      throw cms::Exception("Configuration") << "SuperclusteringDNNInferenceRunner: batchSize must be in [1, 2^31)";
    }
    if (featureCount_ != 9 && featureCount_ != 17 && featureCount_ != 18) {
      throw cms::Exception("Configuration")
          << "SuperclusteringDNNInferenceRunner: " << featureCount_ << " features, expected 9, 17 or 18 (v1, v2, v3)";
    }
    // Set the precision on the CPU; runInto() moves the model to the device of the queue.
    model_.to(dtype_);
  }

  void SuperclusteringDNNInferenceRunner::warmup(Queue& queue) {
    if (warmupIterations_ == 0) {
      return;
    }
    // A full batch of zeros: the first passes on a device load the model and initialise PyTorch's kernels, which
    // would otherwise land in the first events.
    InputDevice features(queue, batchSize_);
    features.zeroInitialise(queue);
    ModelOutputDevice scores(queue, batchSize_);
    for (uint32_t i = 0; i < warmupIterations_; ++i) {
      runInto(queue, features, scores);
    }
  }

  void SuperclusteringDNNInferenceRunner::runInto(Queue& queue,
                                                  InputDevice const& features,
                                                  ModelOutputDevice& scores) {
    const int total = features.const_view().metadata().size();
    if (scores.const_view().metadata().size() != total) {
      throw cms::Exception("LogicError") << "SuperclusteringDNNInferenceRunner: " << total << " pairs but "
                                         << scores.const_view().metadata().size() << " scores";
    }
    if (total == 0) {
      return;
    }

    const auto featureRecords = features.const_view().records();
    auto scoreRecords = scores.view().records();
    if (cms::torch::alpakatools::getDevice(queue) != model_.device()) {
      torch::QueueGuard guard(queue);
      model_.to(queue, dtype_);
    }

    SuperclusteringTensors inputs, outputs;
    Adapter::registerInputs(inputs, featureRecords, batchSize_, featureCount_);
    Adapter::registerOutputs(outputs, scoreRecords, batchSize_);
    model_.forward(queue, inputs, outputs, dtype_);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
