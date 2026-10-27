#ifndef RecoTICL_Superclustering_alpaka_SuperclusteringDNNInferenceRunner_h
#define RecoTICL_Superclustering_alpaka_SuperclusteringDNNInferenceRunner_h

#include <cstdint>
#include <string>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNTensorAdapter.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // PyTorchAlpaka inference of the superclustering DNN, independent of the EDM Event and of the module lifecycle.
  // Each stream::FixedQueueEDProducer owns its own runner.
  class SuperclusteringDNNInferenceRunner {
  public:
    using Adapter = SuperclusteringDNNTensorAdapter;
    using InputDevice = Adapter::InputDevice;
    using ModelOutputDevice = Adapter::ModelOutputDevice;

    SuperclusteringDNNInferenceRunner(std::string const& modelPath,
                                      unsigned int featureCount,
                                      uint32_t batchSize,
                                      bool convertToFP16,
                                      uint32_t warmupIterations);

    void warmup(Queue& queue);

    // Enqueues the forward pass in mini-batches of batchSize pairs; one score per row of the features.
    ModelOutputDevice run(Queue& queue, InputDevice const& features);

  private:
    torch::AlpakaModel model_;
    ::torch::Dtype const dtype_;
    unsigned int const featureCount_;
    uint32_t const batchSize_;
    uint32_t const warmupIterations_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Superclustering_alpaka_SuperclusteringDNNInferenceRunner_h
