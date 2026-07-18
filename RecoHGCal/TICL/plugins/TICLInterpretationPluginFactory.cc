// #include "TracksterLinkingbySkeletons.h"
// #include "TracksterLinkingbySuperClustering.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "RecoHGCal/TICL/plugins/TICLInterpretationPluginFactory.h"
#include "GeneralInterpretationAlgo.h"
#include "GNNInterpretationAlgo.h"
#include "MuonInterpretationAlgo.h"
#include "ChargedHadronInterpretationAlgo.h"
#include "EGammaInterpretationAlgo.h"
#include "JetInterpretationAlgo.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(TICLGeneralInterpretationPluginFactory, "TICLGeneralInterpretationPluginFactory");
EDM_REGISTER_VALIDATED_PLUGINFACTORY(TICLEGammaInterpretationPluginFactory, "TICLEGammaInterpretationPluginFactory");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::GeneralInterpretationAlgo, "General");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::GNNInterpretationAlgo, "GNNLink");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::MuonInterpretationAlgo, "Muon");
// TICLv6 (ticl_v6 arbitration chain): additive, distinct plugin names. "General" above
// stays mapped to GeneralInterpretationAlgo so TICLv5 is unaffected.
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::ChargedHadronInterpretationAlgo, "ChargedHadron");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::EGammaInterpretationAlgo, "EGamma");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::JetInterpretationAlgo, "Jet");
