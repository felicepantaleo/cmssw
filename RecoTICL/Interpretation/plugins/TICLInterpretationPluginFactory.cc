// #include "RecoTICL/Linking/interface/TracksterLinkingbySkeletons.h"
// #include "TracksterLinkingbySuperClustering.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "RecoTICL/Interpretation/interface/TICLInterpretationPluginFactory.h"
#include "RecoTICL/Interpretation/interface/GeneralInterpretationAlgo.h"
#include "RecoTICL/Interpretation/interface/GNNInterpretationAlgo.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(TICLGeneralInterpretationPluginFactory, "TICLGeneralInterpretationPluginFactory");
EDM_REGISTER_VALIDATED_PLUGINFACTORY(TICLEGammaInterpretationPluginFactory, "TICLEGammaInterpretationPluginFactory");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::GeneralInterpretationAlgo, "General");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::GNNInterpretationAlgo, "GNNLink");
// DEFINE_EDM_VALIDATED_PLUGIN(TICLEGammaInterpretationPluginFactory, ticl::EGammaInterpretation, "EGamma");
