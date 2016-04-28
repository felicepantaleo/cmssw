#include "RecoPixelVertexing/PixelTriplets/interface/CANtupleGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

CANtupleGenerator::CANtupleGenerator(unsigned int maxElement):
theKDTReeCache(nullptr),
theCACellsCache(nullptr),
theMaxElement(maxElement)
{}

CANtupleGenerator::CANtupleGenerator(const edm::ParameterSet& pset):
CANtupleGenerator(pset.getParameter<unsigned int>("maxElement"))
{}

CANtupleGenerator::~CANtupleGenerator() {}

void CANtupleGenerator::init(LayerFKDTreeCache *kdTReeCache,LayerDoubletsCache *doubletsCache) {
    theKDTReeCache = kdTReeCache;
    theDoubletsCache = doubletsCache;
}
