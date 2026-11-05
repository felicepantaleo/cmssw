#ifndef RecoTICL_Superclustering_alpaka_SuperclusteringDNNInferenceRunner_h
#define RecoTICL_Superclustering_alpaka_SuperclusteringDNNInferenceRunner_h

#include <cstdint>
#include <string>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNTensorAdapter.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // PyTorchAlpaka inference of the superclustering DNN, independent of the EDM Event and of the module lifecycle.
  // Each SuperclusteringDNNProducer owns its own runner.
  class SuperclusteringDNNInferenceRunner {
  public:
    using Adapter = SuperclusteringDNNTensorAdapter;
    using InputDevice = Adapter::InputDevice;
    using ModelOutputDevice = Adapter::ModelOutputDevice;

    SuperclusteringDNNInferenceRunner(const std::string& modelPath,
                                      uint32_t batchSize,
                                      bool convertToFP16,
                                      uint32_t warmupIterations);

    void warmup(Queue& queue);

    // Enqueues the forward pass in mini-batches of batchSize pairs. The caller allocates scores with one row per row
    // of the features.
    void runInto(Queue& queue, const InputDevice& features, ModelOutputDevice& scores);

  private:
    torch::AlpakaModel model_;
    const ::torch::Dtype dtype_;
    const uint32_t batchSize_;
    const uint32_t warmupIterations_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Superclustering_alpaka_SuperclusteringDNNInferenceRunner_h
