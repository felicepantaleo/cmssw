#ifndef RecoTICL_Inference_alpaka_TracksterFeatureExtractionRunner_h
#define RecoTICL_Inference_alpaka_TracksterFeatureExtractionRunner_h

#include <cstdint>
#include <optional>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterFeatureExtractors.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename Extractor>
  class TracksterFeatureExtractionRunner {
  public:
    using FeaturesDevice = typename Extractor::FeaturesDevice;

    explicit TracksterFeatureExtractionRunner(edm::ParameterSet const& config);

    // Enqueues selection, and the final device-to-host count transfer
    void prepare(Queue& queue,
                 ::ticl::TracksterSoA::ConstView tracksters,
                 LayerClusterEnergyMultiView clusterEnergies,
                 LayerClusterIndexMultiView clusterIndexes = {});

    // Only call after the framework has synchronized the prepare() work
    uint32_t selectedCount() const;

    // Enqueues feature construction
    FeaturesDevice run(Queue& queue,
                       ::ticl::TracksterSoA::ConstView tracksters,
                       LayerClusterEnergyMultiView clusterEnergies,
                       LayerClusterPositionMultiView clusterPositions,
                       uint32_t layersPerEndcap,
                       LayerClusterIndexMultiView clusterIndexes = {});

  private:
    Extractor extractor_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> selectionMaskDevice_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> selectionOffsetsDevice_;
    std::optional<cms::alpakatools::host_buffer<uint32_t>> selectedCountHost_;
  };

  // The implementation and its explicit instantiation live in src/alpaka.
  extern template class TracksterFeatureExtractionRunner<CNNFeatureExtractor>;
  extern template class TracksterFeatureExtractionRunner<DNNFeatureExtractor>;
  extern template class TracksterFeatureExtractionRunner<PFNFeatureExtractor>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Inference_alpaka_TracksterFeatureExtractionRunner_h
