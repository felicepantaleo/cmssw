#ifndef RecoTICL_PatternRecognition_plugins_alpaka_HGCalCLUEsteringTrackstersAlgoWrapper_h
#define RecoTICL_PatternRecognition_plugins_alpaka_HGCalCLUEsteringTrackstersAlgoWrapper_h

#include <array>
#include <cstdint>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoALayerClustersExtraDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalCLUEsteringTrackstersAlgoWrapper {
  public:
    // The HGCAL components: CE-E, CE-H silicon and CE-H scintillator, in the order of DetId::HGCalEE, HGCalHSi, HGCalHSc
    static constexpr uint32_t kComponents = 3;

    struct Parameters {
      // The clustering distances, in units of sigmaT transversely and of layerScale layers longitudinally
      float deltac;
      float rhoc;
      float outlierDistance;
      float seedingDistance;

      // The transverse angular scale of each HGCAL component [rad]
      std::array<float, kComponents> sigmaT;
      float layerScale;

      // The seed threshold of a layer cluster is rhoc * (rhocPivotRadius / r)^rhocEtaExponent, with r = sqrt(x^2 + y^2) / |z|
      float rhocEtaExponent;
      float rhocPivotRadius;

      // The silicon layer clusters with fewer cells are not clustered
      uint32_t minSiliconCells;

      // The 3D clusters with fewer layer clusters are not tracksters
      uint32_t minLayerClusters;
    };

    // Clusters the layer clusters into tracksters with CLUEstering writes the
    // trackster of each layer cluster, the number of tracksters and the number of layer clusters in them.
    void run(Queue& queue,
             const Parameters& parameters,
             reco::CaloClusterDeviceCollection::ConstView layerClusters,
             HGCalSoALayerClustersExtraDeviceCollection::View layerClustersClustering) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_PatternRecognition_plugins_alpaka_HGCalCLUEsteringTrackstersAlgoWrapper_h
