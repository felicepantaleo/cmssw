#ifndef RecoHGCal_TICL_alpaka_TracksterInferenceDevice_h
#define RecoHGCal_TICL_alpaka_TracksterInferenceDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTICL/Inference/interface/TracksterInferenceHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using TracksterCNNFeaturesDevice = PortableCollection<::ticl::TracksterCNNFeaturesSoA>;
  using TracksterDNNFeaturesDevice = PortableCollection<::ticl::TracksterDNNFeaturesSoA>;
  using TracksterPFNFeaturesDevice = PortableCollection<::ticl::TracksterPFNFeaturesSoA>;

  using TracksterInferencePIDScoresDevice = PortableCollection<::ticl::TracksterInferencePIDScoresSoA>;
  using TracksterInferenceEnergyDevice = PortableCollection<::ticl::TracksterInferenceEnergySoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif
