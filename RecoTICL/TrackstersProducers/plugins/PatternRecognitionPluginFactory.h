#ifndef RecoTICL_TrackstersProducers_PatternRecognitionPluginFactory_H
#define RecoTICL_TrackstersProducers_PatternRecognitionPluginFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoTICL/TrackstersProducers/interface/PatternRecognitionAlgoBase.h"
#include "RecoTICL/TrackstersProducers/interface/GlobalCache.h"

typedef edmplugin::PluginFactory<ticl::PatternRecognitionAlgoBaseT<TICLLayerTiles>*(const edm::ParameterSet&,
                                                                                    edm::ConsumesCollector)>
    PatternRecognitionFactory;
typedef edmplugin::PluginFactory<ticl::PatternRecognitionAlgoBaseT<TICLLayerTilesHFNose>*(const edm::ParameterSet&,
                                                                                          edm::ConsumesCollector)>
    PatternRecognitionHFNoseFactory;

#endif
