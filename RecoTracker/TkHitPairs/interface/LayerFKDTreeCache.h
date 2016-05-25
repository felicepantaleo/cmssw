#ifndef LayerFKDTreeCache_H
#define LayerFKDTreeCache_H

/** A cache adressable by DetLayer* and TrackingRegion* .
 *  Used to cache all the hits of a DetLayer.
 */

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/FKDTree.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "FWCore/Framework/interface/EventSetup.h"

using LayerTree = FKDTree<float,3>;

class LayerFKDTreeCache {
//
private:

  class FKDTreeCache {
  public:

    FKDTreeCache(unsigned int initSize) : theContainer(initSize, nullptr){}
    ~FKDTreeCache() { clear(); }
    void resize(int size) { theContainer.resize(size,nullptr); }
    LayerTree*  get(int key) { return theContainer[key];}
    /// add object to cache. It is caller responsibility to check that object is not yet there.
    void add(int key,LayerTree * value) {
      if (key>=int(theContainer.size())) resize(key+1);
      theContainer[key]=value;
    }
    /// emptify cache, delete values associated to Key
    void clear() {      
      for ( auto & v : theContainer)  { delete v; v=nullptr;}
    }
  private:
    std::vector<LayerTree *> theContainer;
  private:
    FKDTreeCache(const FKDTreeCache &) { }
  };

private:
  typedef FKDTreeCache Cache;
public:
  LayerFKDTreeCache(unsigned int initSize=1000) : theCache(initSize) { }

  void clear() { theCache.clear(); }
  bool checkCache(int key) {return (!(theCache.get(key)==nullptr)); }
  /*void getCache(int key,LayerTree* tree){
        if(checkCache(key)) tree = theCache.get(key);
    }*/
  //void writeCache(int key,LayerTree* tree) {theCache.add(key,tree);}
  
   LayerTree & getTree(const SeedingLayerSetsHits::SeedingLayer& layer, const TrackingRegion & region, const edm::Event & iE, const edm::EventSetup & iS) {
      
      int key = layer.index();
      assert (key>=0);
      
      LayerTree * cache = theCache.get(key);
      LayerTree *buffer = new FKDTree<float,3>();
      if (cache==nullptr) {
          
          std::cout<<"Nullptr : let's do the tree!"<<std::endl;
          buffer->FKDTree<float,3>::make_FKDTreeFromRegionLayer(layer,region,iE,iS);
          
          cache = buffer;
          theCache.add(key,cache);
          
      }

      return *cache;}
  /*
  LayerTree &
  operator()(const SeedingLayerSetsHits::SeedingLayer& layer, const TrackingRegion & region,
	     const edm::Event & iE, const edm::EventSetup & iS) {
    int key = layer.index();
    assert (key>=0);
    LayerTree * cacheTree = theCache.get(key);
    if (cacheTree==nullptr) {
        cacheTree->FKDTree<float,3>::make_FKDTreeFromRegionLayer(layer,region,iE,iS);
    /*LogDebug("LayerHitMapCache")<<" I got"<< lhm->all().second-lhm->all().first<<" hits in the cache for: "<<layer.detLayer();
      theCache.add(key,cacheTree);
    }
    else{
      // std::cout << region.origin() << " " <<  lhm->theOrigin << std::endl;
      LogDebug("LayerFKDTreeCache")<<" FKDTree for layer"<< layer.detLayer()<<" already in the cache with key: "<<key;
    }
    //const Tree result(*buffer);
    return *cacheTree;
  }*/

private:
  Cache theCache; 
};

#endif

