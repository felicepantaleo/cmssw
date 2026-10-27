#ifndef RecoTICL_Superclustering_alpaka_SuperclusteringDNNDeviceCollections_h
#define RecoTICL_Superclustering_alpaka_SuperclusteringDNNDeviceCollections_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNHostCollections.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using SuperclusteringDNNPairFeaturesDevice = PortableCollection<::ticl::SuperclusteringDNNPairFeaturesSoA>;
  using SuperclusteringDNNPairsDevice = PortableCollection<::ticl::SuperclusteringDNNPairsSoA>;
  using SuperclusteringDNNPairScoresDevice = PortableCollection<::ticl::SuperclusteringDNNPairScoresSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // RecoTICL_Superclustering_alpaka_SuperclusteringDNNDeviceCollections_h
