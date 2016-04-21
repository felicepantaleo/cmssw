#ifndef CANTupleGeneratorFactory_H
#define CANTupleGeneratorFactory_H

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {class ParameterSet; class ConsumesCollector;}

typedef edmplugin::PluginFactory<CANTupleGenerator *(const edm::ParameterSet &, edm::ConsumesCollector&)>
	CANTupleGeneratorFactory;
 
#endif
