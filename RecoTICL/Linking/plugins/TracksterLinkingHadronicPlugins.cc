#include "RecoTICL/Linking/interface/TracksterLinkingPluginFactory.h"
#include "RecoTICL/Linking/interface/TracksterLinkingbySkeletons.h"
#include "RecoTICL/Linking/interface/TracksterLinkingbyFastJet.h"
#include "RecoTICL/Linking/interface/TracksterLinkingRecovery.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingbySkeletons, "Skeletons");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingbyFastJet, "FastJet");
DEFINE_EDM_VALIDATED_PLUGIN(TracksterLinkingPluginFactory, ticl::TracksterLinkingRecovery, "Recovery");
