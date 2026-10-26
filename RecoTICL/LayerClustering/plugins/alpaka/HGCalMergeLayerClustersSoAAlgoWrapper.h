#ifndef RecoTICL_LayerClustering_plugins_alpaka_HGCalMergeLayerClustersSoAAlgoWrapper_h
#define RecoTICL_LayerClustering_plugins_alpaka_HGCalMergeLayerClustersSoAAlgoWrapper_h

#include <cstdint>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalMergeLayerClustersSoAAlgoWrapper {
  public:
    // Copies the layer clusters of one HGCAL component to the merged layer clusters, starting at row offset
    void run(Queue& queue,
             reco::CaloClusterDeviceCollection::ConstView layerClusters,
             uint32_t offset,
             reco::CaloClusterDeviceCollection::View mergedLayerClusters) const;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_LayerClustering_plugins_alpaka_HGCalMergeLayerClustersSoAAlgoWrapper_h
