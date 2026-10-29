#ifndef RecoTICL_Superclustering_SuperclusteringDNNSoA_h
#define RecoTICL_Superclustering_SuperclusteringDNNSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  // One row per (seed, candidate) pair that passes the preselection of TracksterLinkingbySuperClusteringDNN.
  // The columns follow the order of SuperclusteringDNNInputV3::featureNames(). V1 and V2 are prefixes of V3
  // (9 and 17 features), so the same layout serves every version: register only the first featureCount() columns.
  GENERATE_SOA_LAYOUT(SuperclusteringDNNPairFeaturesLayout,
                      SOA_COLUMN(float, deltaEtaBaryc),
                      SOA_COLUMN(float, deltaPhiBaryc),
                      SOA_COLUMN(float, multiEn),
                      SOA_COLUMN(float, multiEta),
                      SOA_COLUMN(float, multiPt),
                      SOA_COLUMN(float, seedEta),
                      SOA_COLUMN(float, seedPhi),
                      SOA_COLUMN(float, seedEn),
                      SOA_COLUMN(float, seedPt),
                      SOA_COLUMN(float, theta),
                      SOA_COLUMN(float, thetaXZSeedFrame),
                      SOA_COLUMN(float, thetaYZSeedFrame),
                      SOA_COLUMN(float, thetaXYCmsFrame),
                      SOA_COLUMN(float, thetaYZCmsFrame),
                      SOA_COLUMN(float, thetaXZCmsFrame),
                      SOA_COLUMN(float, explVar),
                      SOA_COLUMN(float, explVarRatio),
                      SOA_COLUMN(float, modDeltaTime))

  // The input tracksters of each pair, as indices in the trackster SoA. Same size and row order as the features.
  // Kept out of the features so that only the indices and the scores are copied to the host.
  GENERATE_SOA_LAYOUT(SuperclusteringDNNPairsLayout,
                      SOA_COLUMN(uint32_t, seedIndex),
                      SOA_COLUMN(uint32_t, candidateIndex))

  // DNN output: [pairs, 1].
  GENERATE_SOA_LAYOUT(SuperclusteringDNNPairScoresLayout, SOA_COLUMN(float, score))

  using SuperclusteringDNNPairFeaturesSoA = SuperclusteringDNNPairFeaturesLayout<>;
  using SuperclusteringDNNPairsSoA = SuperclusteringDNNPairsLayout<>;
  using SuperclusteringDNNPairScoresSoA = SuperclusteringDNNPairScoresLayout<>;

}  // namespace ticl

#endif  // RecoTICL_Superclustering_SuperclusteringDNNSoA_h
