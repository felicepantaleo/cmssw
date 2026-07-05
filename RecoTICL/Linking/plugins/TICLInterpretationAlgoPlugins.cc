#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "RecoTICL/Interpretation/interface/TICLInterpretationPluginFactory.h"
#include "RecoTICL/Interpretation/interface/ChargedHadronInterpretationAlgo.h"
#include "RecoTICL/Interpretation/interface/GNNInterpretationAlgo.h"
#include "RecoTICL/Interpretation/interface/MuonInterpretationAlgo.h"
#include "RecoTICL/Interpretation/interface/EGammaInterpretationAlgo.h"
#include "RecoTICL/Interpretation/interface/JetInterpretationAlgo.h"

DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory,
                            ticl::ChargedHadronInterpretationAlgo,
                            "ChargedHadron");
// Deprecated alias, kept for configuration compatibility.
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::ChargedHadronInterpretationAlgo, "General");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::GNNInterpretationAlgo, "GNNLink");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::MuonInterpretationAlgo, "Muon");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::EGammaInterpretationAlgo, "EGamma");
DEFINE_EDM_VALIDATED_PLUGIN(TICLGeneralInterpretationPluginFactory, ticl::JetInterpretationAlgo, "Jet");
