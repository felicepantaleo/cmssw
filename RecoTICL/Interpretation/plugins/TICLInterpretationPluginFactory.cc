#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "TICLInterpretationPluginFactory.h"
#include "EGammaInterpretationAlgo.h"
#include "GeneralInterpretationAlgo.h"
#include "GNNInterpretationAlgo.h"
#include "JetInterpretationAlgo.h"
#include "MuonInterpretationAlgo.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(TICLGeneralInterpretationPluginFactory, "TICLGeneralInterpretationPluginFactory");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::GeneralInterpretationAlgo, "General");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::GNNInterpretationAlgo, "GNNLink");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::MuonInterpretationAlgo, "Muon");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::EGammaInterpretationAlgo, "EGamma");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::JetInterpretationAlgo, "Jet");
