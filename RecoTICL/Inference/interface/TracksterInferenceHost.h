#ifndef RecoHGCal_TICL_TracksterInferenceHost_h
#define RecoHGCal_TICL_TracksterInferenceHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "RecoTICL/Inference/interface/TracksterInferenceSoA.h"

namespace ticl {

  using TracksterCNNFeaturesHost = PortableHostCollection<::ticl::TracksterCNNFeaturesSoA>;
  using TracksterDNNFeaturesHost = PortableHostCollection<::ticl::TracksterDNNFeaturesSoA>;
  using TracksterPFNFeaturesHost = PortableHostCollection<::ticl::TracksterPFNFeaturesSoA>;

  using TracksterInferencePIDScoresHost = PortableHostCollection<::ticl::TracksterInferencePIDScoresSoA>;
  using TracksterInferenceEnergyHost = PortableHostCollection<::ticl::TracksterInferenceEnergySoA>;

}  // namespace ticl

#endif
