#ifndef RecoTICL_ClusterFilters_plugins_alpaka_LayerClusterMaskAlgos_h
#define RecoTICL_ClusterFilters_plugins_alpaka_LayerClusterMaskAlgos_h

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

#include <array>
#include <cstdint>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/LayerClusterMaskDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  // Sets every value of the mask to 1.
  void fillMaskWithOnes(Queue& queue, LayerClusterMaskDeviceCollection::View mask);

  // Copies the values of input into output. The two masks have the same size.
  void copyMask(Queue& queue,
                LayerClusterMaskDeviceCollection::ConstView input,
                LayerClusterMaskDeviceCollection::View output);

  // Sets to 0 the mask of each layer cluster of the tracksters.
  void maskLayerClustersOfTracksters(Queue& queue,
                                     TracksterDeviceCollection::ConstView tracksters,
                                     LayerClusterMaskDeviceCollection::View mask);

  // The selection of ClusterFilterByAlgoAndSize.
  struct AlgoAndSizeFilter {
    static constexpr unsigned int kMaxAlgos = 8;
    std::array<int, kMaxAlgos> algos{};
    unsigned int nAlgos = 0;
    // A silicon layer cluster with fewer cells is rejected.
    int minSize = 0;
    // A layer cluster with more cells is rejected.
    int maxSize = 0;
  };

  // output = input, with 0 for each layer cluster that the filter rejects: its algorithm is not in the list, it has
  // more than maxSize cells, or it is a silicon layer cluster with fewer than minSize cells. The seed of a layer cluster
  // tells if it is silicon, as ticlgeom::Tools::isSilicon does.
  void filterByAlgoAndSize(Queue& queue,
                           AlgoAndSizeFilter const& filter,
                           reco::CaloClusterDeviceCollection::ConstView layerClusters,
                           LayerClusterMaskDeviceCollection::ConstView input,
                           LayerClusterMaskDeviceCollection::View output);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // RecoTICL_ClusterFilters_plugins_alpaka_LayerClusterMaskAlgos_h
