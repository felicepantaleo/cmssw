#include "RecoTICL/Linking/interface/TracksterLinkingPluginFactory.h"
#include "RecoTICL/Superclustering/interface/TracksterLinkingbySuperClusteringDNN.h"
#include "RecoTICL/Superclustering/interface/TracksterLinkingbySuperClusteringMustache.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory,
                            ticl::TracksterLinkingbySuperClusteringDNN,
                            "SuperClusteringDNN");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory,
                            ticl::TracksterLinkingbySuperClusteringMustache,
                            "SuperClusteringMustache");
