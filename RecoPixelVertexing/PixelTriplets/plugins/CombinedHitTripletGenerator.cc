#include "CombinedHitTripletGenerator.h"

#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"
//#include "RecoPixelVertexing/PixelTriplets/interface/LayerQuadruplets.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#uinclude <iostream>

CombinedHitTripletGenerator::CombinedHitTripletGenerator(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) :
  theSeedingLayerToken(iC.consumes<SeedingLayerSetsHits>(cfg.getParameter<edm::InputTag>("SeedingLayers")))
{
  edm::ParameterSet generatorPSet = cfg.getParameter<edm::ParameterSet>("GeneratorPSet");
  std::string       generatorName = generatorPSet.getParameter<std::string>("ComponentName");
  theGenerator.reset(HitTripletGeneratorFromPairAndLayersFactory::get()->create(generatorName, generatorPSet, iC));
  theGenerator->init(std::make_unique<HitPairGeneratorFromLayerPair>(0, 1, &theLayerCache), &theLayerCache);
}

CombinedHitTripletGenerator::~CombinedHitTripletGenerator() {}

void CombinedHitTripletGenerator::hitTriplets(
   const TrackingRegion& region, OrderedHitTriplets & result,
   const edm::Event& ev, const edm::EventSetup& es)
{
  edm::Handle<SeedingLayerSetsHits> hlayers;
  ev.getByToken(theSeedingLayerToken, hlayers);
  const SeedingLayerSetsHits& layers = *hlayers;
 
  if(layers.numberOfLayersInSet() != 3)
    throw cms::Exception("Configuration") << "CombinedHitTripletGenerator expects SeedingLayerSetsHits::numberOfLayersInSet() to be 3, got " << layers.numberOfLayersInSet();
  //////////////////////////////////////////////////////////////////////////////////////////////////////
    //////LAYERS MAP & HITS VECTORS
    std::vector<std::string> layerNames;
    std::vector<std::string>::iterator layerNamesIt;
    unsigned int layerCounter = 0;
    std::map<unsigned int,SeedingLayerSetsHits::SeedingLayer*> layerMap;
    std::map<unsigned int,SeedingLayerSetsHits::SeedingLayer*>::iterator layerMapIterator;
    std::map<unsigned int,Hits> layerHitsMap;
    
    std::cout<< "SeedingLayerSetsHits with " << layers.numberOfLayersInSet() << " layers in each LayerSets, LayerSets has " << layers.size() << " items\n";
    for(SeedingLayerSetsHits::LayerSetIndex iLayers=0; iLayers<layers.size(); ++iLayers) {
        std::cout << " " << iLayers << ": ";
        SeedingLayerSetsHits::SeedingLayerSet Llayers = layers[iLayers];
        for(unsigned iLayer=0; iLayer<Llayers.size(); ++iLayer) {
            SeedingLayerSetsHits::SeedingLayer Slayer = Llayers[iLayer];
            std::cout << Slayer.name() << " (" << Slayer.index() << ", nhits " << Slayer.hits().size() << ") "<<std::endl;
            if(std::find(layerNames.begin(),layerNames.end(),Slayer.name())!=layerNames.end()){
                layerNames.push_back(Slayer.name());
                layerMap[layerCounter] = &Slayer;
                ++layerCounter;
                std::cout<<"Layer : "<<layerNames.back()<<" registered with id = "<<layerCounter<<std::endl;
            }
        }
    }
    
    for(layerMapIterator=layerMap.begin();layerMapIterator!=layerMap.end();layerMapIterator++){
        layerHitsMap[layerMapIterator->first]=(layerMapIterator->second->hits());
    }
    
  //////////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<LayerTriplets::LayerSetAndLayers> trilayers = LayerTriplets::layers(layers);
  for(const auto& setAndLayers: trilayers) {
    theGenerator->hitTriplets( region, result, ev, es, setAndLayers.first, setAndLayers.second);
  }
  theLayerCache.clear();
    
   
    
}

////////////////////////
//////Quadruplets
/*
void CombinedHitTripletGenerator::hitQuadruplets(
                                              const TrackingRegion& region, OrderedHitQuadruplets & result,
                                              const edm::Event& ev, const edm::EventSetup& es)
{
    edm::Handle<SeedingLayerSetsHits> hlayers;
    ev.getByToken(theSeedingLayerToken, hlayers);
    const SeedingLayerSetsHits& layers = *hlayers;
  
    std::vector<QuadrupletsLayersSets> quadlayers;
    

    std::vector<LayerTriplets::LayerSetAndLayers> trilayers = LayerTriplets::layers(layers);
    for(const auto& setAndLayers: trilayers) {
        theGenerator->hitTriplets( region, result, ev, es, setAndLayers.first, setAndLayers.second);
    }
    theLayerCache.clear();
 
    
    
}
*/
