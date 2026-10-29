#ifndef RecoTICL_Superclustering_SuperclusteringDNNHostCollections_h
#define RecoTICL_Superclustering_SuperclusteringDNNHostCollections_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNSoA.h"

namespace ticl {

  using SuperclusteringDNNPairFeaturesHost = PortableHostCollection<SuperclusteringDNNPairFeaturesSoA>;
  using SuperclusteringDNNPairsHost = PortableHostCollection<SuperclusteringDNNPairsSoA>;
  using SuperclusteringDNNPairScoresHost = PortableHostCollection<SuperclusteringDNNPairScoresSoA>;

}  // namespace ticl

#endif  // RecoTICL_Superclustering_SuperclusteringDNNHostCollections_h
