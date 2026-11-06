#ifndef RecoTICL_PatternRecognition_plugins_alpaka_HGCalRecoveryTrackstersSoAAlgoWrapper_h
#define RecoTICL_PatternRecognition_plugins_alpaka_HGCalRecoveryTrackstersSoAAlgoWrapper_h

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

#include <cstdint>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/LayerClusterMaskDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The tracksters of PatternRecognitionbyRecovery: one trackster for each layer cluster with a non-zero mask and an
  // energy of at least minEnergy, in the order of the layer clusters.
  class HGCalRecoveryTrackstersSoAAlgoWrapper {
  public:
    // positions[i] is the number of selected layer clusters up to and including i. The last value is the number of
    // tracksters.
    void selectLayerClusters(Queue& queue,
                             double minEnergy,
                             reco::CaloClusterDeviceCollection::ConstView layerClusters,
                             ticl::LayerClusterMaskDeviceCollection::ConstView mask,
                             uint32_t* selected,
                             uint32_t* positions) const;

    // Fills one trackster per selected layer cluster: its layer cluster, raw energy, barycenter and time from the
    // layer cluster. The raw EM energy is the energy when |z| <= zLimitEm, else 0. The regressed energy, the
    // principal components and the PID probabilities are 0.
    void fillTracksters(Queue& queue,
                        float zLimitEm,
                        reco::CaloClusterDeviceCollection::ConstView layerClusters,
                        uint32_t const* selected,
                        uint32_t const* positions,
                        ticl::TracksterDeviceCollection::View tracksters) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_PatternRecognition_plugins_alpaka_HGCalRecoveryTrackstersSoAAlgoWrapper_h
