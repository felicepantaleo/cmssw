#include <cstdint>
#include <string>

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
    // TODO: validate batchSize_ and featureCount_
    // Set precision on CPU as well; AlpakaModel moves to the queue device at forward().
    model_.to(dtype_);
  }

  void SuperclusteringDNNInferenceRunner::warmup(Queue& queue) {
    if (warmupIterations_ == 0) {
      return;
    }
    // TODO: run warmupIterations_ forward passes on dummy InputDevice / ModelOutputDevice of batchSize_ rows
  }

  SuperclusteringDNNInferenceRunner::ModelOutputDevice SuperclusteringDNNInferenceRunner::run(
      Queue& queue, InputDevice const& features) {
    const auto total = features.const_view().metadata().size();
    ModelOutputDevice scores(queue, total);
    if (total == 0) {
      return scores;
    }

    // TODO: for each mini-batch i of batchSize_ pairs
    //   SuperclusteringTensors inputs(batchSize_, total), outputs(batchSize_, total);
    //   Adapter::registerInputs(inputs, features.const_view().records(), i, featureCount_);
    //   Adapter::registerOutputs(outputs, scores.view().records(), i);
    //   model_.forward(queue, inputs, outputs, dtype_);
    return scores;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
