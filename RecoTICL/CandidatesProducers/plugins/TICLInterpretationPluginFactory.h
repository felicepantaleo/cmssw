#ifndef RecoTICL_CandidatesProducers_TICLInterpretationPluginFactory_H
#define RecoTICL_CandidatesProducers_TICLInterpretationPluginFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoTICL/CandidatesProducers/interface/TICLInterpretationAlgoBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

using TICLGeneralInterpretationPluginFactory = edmplugin::PluginFactory<ticl::TICLInterpretationAlgoBase<reco::Track>*(
    const edm::ParameterSet&, edm::ConsumesCollector)>;
using TICLEGammaInterpretationPluginFactory =
    edmplugin::PluginFactory<ticl::TICLInterpretationAlgoBase<reco::GsfTrack>*(const edm::ParameterSet&,
                                                                               edm::ConsumesCollector)>;

#endif
