// #include "TracksterLinkingbySkeletons.h"
// #include "TracksterLinkingbySuperClustering.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "RecoHGCal/TICL/plugins/TICLInterpretationPluginFactory.h"
#include "ChargedHadronInterpretationAlgo.h"
#include "GNNInterpretationAlgo.h"
#include "MuonInterpretationAlgo.h"
#include "EGammaInterpretationAlgo.h"
#include "JetInterpretationAlgo.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(TICLGeneralInterpretationPluginFactory, "TICLGeneralInterpretationPluginFactory");
EDM_REGISTER_VALIDATED_PLUGINFACTORY(TICLEGammaInterpretationPluginFactory, "TICLEGammaInterpretationPluginFactory");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory,
                            ticl::ChargedHadronInterpretationAlgo,
                            "ChargedHadron");
// Deprecated alias, kept for configuration compatibility.
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::ChargedHadronInterpretationAlgo, "General");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::GNNInterpretationAlgo, "GNNLink");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::MuonInterpretationAlgo, "Muon");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::EGammaInterpretationAlgo, "EGamma");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::JetInterpretationAlgo, "Jet");
