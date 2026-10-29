#ifndef RecoTICL_Inference_alpaka_TracksterFeatureExtractors_h
#define RecoTICL_Inference_alpaka_TracksterFeatureExtractors_h

#include <cstdint>
#include <type_traits>

#include "DataFormats/CaloRecHit/interface/CaloClusterSoA.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/SoATemplate/interface/SoAConstMultiView.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Inference/interface/alpaka/TracksterInferenceDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  enum class TracksterModelFamily { CNN, DNN, PFN };

  using LayerClusterEnergyMultiView = SoAConstMultiView<::reco::CaloClusterSoAEnergy<>::ConstView, 3>;
  using LayerClusterPositionMultiView = SoAConstMultiView<::reco::CaloClusterSoAPosition<>::ConstView, 3>;
  using LayerClusterIndexMultiView = SoAConstMultiView<::reco::CaloClusterSoAIndexes<>::ConstView, 3>;

  // Extractor policy for the legacy Trackster CNN features.
  // The policy owns model-specific configuration and CNN GPU kernels.
  //
  // A different model can supply its own FeaturesDevice, selection policy,
  // and feature-filling implementation without changing the producer.
  template <TracksterModelFamily Family>
  struct TracksterFeatureExtractor {
    // clang-format off
    using FeaturesDevice =
      std::conditional_t<Family == TracksterModelFamily::CNN, ticl::TracksterCNNFeaturesDevice,
      std::conditional_t<Family == TracksterModelFamily::DNN, ticl::TracksterDNNFeaturesDevice,
      std::conditional_t<Family == TracksterModelFamily::PFN, ticl::TracksterPFNFeaturesDevice,
      void>>>;
    // clang-format on
    static_assert(!std::is_void_v<FeaturesDevice>, "Unsupported Trackster model family");

    explicit TracksterFeatureExtractor(edm::ParameterSet const& config);

    static void fillDescription(edm::ParameterSetDescription& description);

    // Must produce an inclusive selection prefix sum in selectionOffsets.
    // The last offset is the number of selected Tracksters.
    void select(Queue& queue,
                ::ticl::TracksterSoA::ConstView tracksters,
                LayerClusterEnergyMultiView clusterEnergies,
                LayerClusterIndexMultiView clusterIndexes,
                uint32_t* selectionMask,
                uint32_t* selectionOffsets,
                uint32_t nTracksters) const;

    // Fills the model-specific SoA and its selectedTracksterIndex mapping.
    void fill(Queue& queue,
              ::ticl::TracksterSoA::ConstView tracksters,
              LayerClusterEnergyMultiView clusterEnergies,
              LayerClusterIndexMultiView clusterIndexes,
              LayerClusterPositionMultiView clusterPositions,
              uint32_t layersPerEndcap,
              uint32_t const* selectionMask,
              uint32_t const* selectionOffsets,
              FeaturesDevice& features,
              uint32_t nTracksters) const;

  private:
    float minClusterEnergy_;
  };

  using CNNFeatureExtractor = TracksterFeatureExtractor<TracksterModelFamily::CNN>;
  using DNNFeatureExtractor = TracksterFeatureExtractor<TracksterModelFamily::DNN>;
  using PFNFeatureExtractor = TracksterFeatureExtractor<TracksterModelFamily::PFN>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Inference_alpaka_TracksterFeatureExtractors_h
