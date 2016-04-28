#ifndef LayerDoubletsCache_H
#define LayerDoubletsCache_H

/** 
 
 1) PASSO I LAYERS PERCHE' HO BISOGNO DEGLI HITS E NON STANNO NEI KDTREE!
 
 *  Used to cache all the hits of a DetLayer.
 */

#include "RecoTracker/TkHitPairs/interface/HitDoubletsCA.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPairCA.h"
#include "RecoTracker/TkHitPairs/interface/LayerFKDTreeCache.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/FKDTree.h"
#include "RecoTracker/TkHitPairs/interface/CACell.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <cassert>

#define NUMLAYERS 20

class HitPairGeneratorFromLayerPairCA;
class LayerFKDTreeCache;

typedef SeedingLayerSetsHits::SeedingLayerSet Layers;

class LayerDoubletsCache {

private:

  class DoubletsCache {
  public:

    DoubletsCache(unsigned int initSize) : theContainer(initSize, nullptr){}
    ~DoubletsCache() { clear(); }
    void resize(int size) { theContainer.resize(size,nullptr); }
    const HitDoubletsCA*  get(int key) { return theContainer[key];}
    /// add object to cache. It is caller responsibility to check that object is not yet there.
    void add(int key, const HitDoubletsCA* value) {
      if (key>=int(theContainer.size())) resize(key+1);
      theContainer[key]=value;
    }
    /// emptify cache, delete values associated to Key
    void clear() {      
      for ( auto & v : theContainer)  { delete v; v=nullptr;}
    }
  private:
    std::vector< const HitDoubletsCA*> theContainer;
  private:
    DoubletsCache(const DoubletsCache &) { }
  };

private:
  typedef DoubletsCache Cache;
public:
  LayerDoubletsCache(unsigned int initSize=100) : theCache(initSize),theTreeCache(nullptr) { }
  
  void clear() { theCache.clear(); }
    
  void init(LayerFKDTreeCache* tree) { theTreeCache = std::move(tree); }
  
  const HitDoubletsCA &
    operator()(const SeedingLayer& innerLayer,const SeedingLayer& outerLayer, const FKDTree<float,2> & innerTree, const FKDTree<float,2> & outerTree, const TrackingRegion & region, const edm::Event & iE, const edm::EventSetup & iS) {
    const unsigned short int nLayers = layers.size();
    //assert (nLayers == 2, "Error : two layers needed!" );
        
    int key = (innerLayer.detLayer()->seqNum()-1)*NUMLAYERS + outerLayer.detLayer()->seqNum();
    assert (key>=0);
    const HitDoubletsCA* buffer = theCache.get(key);
    HitDoubletsCA result (innerLayer,outerLayer);
    if (buffer==nullptr) {
        HitPairGeneratorFromLayerPairCA* thePairGenerator(innerLayer.detLayer()->seqNum(),outerLayer.detLayer()->seqNum(),innerTree,outerTree,100);
      //buffer=make_CACellsFromRegionLayer(layer,region,iE,iS)
        result = thePairGenerator->doublets(const TrackingRegion & region,const edm::Event & iE, const edm::EventSetup & iS,
                                            const SeedingLayer& innerLayer,const SeedingLayer& outerLayer,
                                            const FKDTree<float,2> & innerTree, const FKDTree<float,2> & outerTree);
        buffer = &result;
    /*LogDebug("LayerHitMapCache")<<" I got"<< lhm->all().second-lhm->all().first<<" hits in the cache for: "<<layer.detLayer();*/
      theCache.add( key, buffer);
    }
    else{
      // std::cout << region.origin() << " " <<  lhm->theOrigin << std::endl;
      LogDebug("LayerDoubletsCache")<<" Doublets for layers"<< layers[0].detLayer()->seqNum() <<" & "<<layers[1].detLayer()->seqNum()<<" already in the cache with key: "<<key;
    }
    return *buffer;
  }

    void init(HitPairGeneratorFromLayerPairCA const & pairGenerator, ) {thePairGenerator = std::move(pairGenerator);}
    
    
    
private:
  Cache theCache;
};

#endif

