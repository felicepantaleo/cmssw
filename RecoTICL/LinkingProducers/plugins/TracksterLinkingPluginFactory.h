#ifndef RecoTICL_LinkingProducers_TracksterLinkingPluginFactory_H
#define RecoTICL_LinkingProducers_TracksterLinkingPluginFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoTICL/LinkingProducers/interface/TracksterLinkingAlgoBase.h"

using TracksterLinkingPluginFactory =
    edmplugin::PluginFactory<ticl::TracksterLinkingAlgoBase*(const edm::ParameterSet&, edm::ConsumesCollector)>;

#endif
