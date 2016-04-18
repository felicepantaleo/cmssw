#ifndef LayerFKDTreeCache_H
#define LayerFKDTreeCache_H

/** A cache adressable by DetLayer* and TrackingRegion* .
 *  Used to cache all the hits of a DetLayer.
 */

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
//#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/FKDTree.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "FWCore/Framework/interface/EventSetup.h"

using FKDTree = FKDTree<float,2>;

class LayerFKDTreeCache {

private:

  class FKDTreeCache {
  public:

    SimpleCache(unsigned int initSize) : theContainer(initSize, nullptr){}
    ~SimpleCache() { clear(); }
    void resize(int size) { theContainer.resize(size,nullptr); }
    const FKDTree*  get(int key) { return theContainer[key];}
    /// add object to cache. It is caller responsibility to check that object is not yet there.
    void add(int key, const FKDTree * value) {
      if (key>=int(theContainer.size())) resize(key+1);
      theContainer[key]=value;
    }
    /// emptify cache, delete values associated to Key
    void clear() {      
      for ( auto & v : theContainer)  { delete v; v=nullptr;}
    }
  private:
    std::vector< const FKDTree *> theContainer;
  private:
    FKDTreeCache(const FKDTreeCache &) { }
  };

private:
  typedef FKDTreeCache Cache;
public:
  LayerFKDTreeCache(unsigned int initSize=50) : theCache(initSize) { }

  void clear() { theCache.clear(); }
  
  const FKDTree &
  operator()(const SeedingLayerSetsHits::SeedingLayer& layer, const TrackingRegion & region,
	     const edm::Event & iE, const edm::EventSetup & iS) {
    int key = layer.index();
    assert (key>=0);
    const FKDTree * buffer = theCache.get(key);
    if (buffer==nullptr) {
      buffer=make_FKDTreeFromRegionLayer(layer,region,iE,iS)

    /*LogDebug("LayerHitMapCache")<<" I got"<< lhm->all().second-lhm->all().first<<" hits in the cache for: "<<layer.detLayer();*/
      theCache.add( key, buffer);
    }
    else{
      // std::cout << region.origin() << " " <<  lhm->theOrigin << std::endl;
      LogDebug("LayerFKDTreeCache")<<" FKDTree for layer"<< layer.detLayer()<<" already in the cache with key: "<<key;
    }
    return *buffer;
  }

private:
  Cache theCache; 
};

#endif

