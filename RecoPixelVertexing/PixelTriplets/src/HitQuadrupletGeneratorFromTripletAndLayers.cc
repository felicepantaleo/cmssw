#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGeneratorFromTripletAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"

HitQuadrupletGeneratorFromTripletAndLayers::HitQuadrupletGeneratorFromTripletAndLayers():
  theLayerCache(nullptr)
{}

HitQuadrupletGeneratorFromTripletAndLayers::~HitQuadrupletGeneratorFromTripletAndLayers() {}

void HitQuadrupletGeneratorFromTripletAndLayers::init(std::unique_ptr<HitTripletGeneratorFromPairAndLayers>&& tripletGenerator, LayerCacheType *layerCache) {
  theTripletGenerator = std::move(tripletGenerator);
  theLayerCache = layerCache;
}

void HitQuadrupletGeneratorFromTripletAndLayers::init( std::unique_ptr<HitTripletGeneratorFromPairAndLayers>&& tripletGenerator, LayerCacheType* layerCache, LayerTreeCacheType* treeCache, LayerDoubletsCacheType* doubletsCache) {
    theTripletGenerator = std::move(tripletGenerator);
    theLayerCache = layerCache;
    theKDTreeCache = treeCache;
    theDoubletsCache = doubletsCache;
}

