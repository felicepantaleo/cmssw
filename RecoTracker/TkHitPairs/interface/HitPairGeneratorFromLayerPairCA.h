#ifndef HitPairGeneratorFromLayerPairCA_h
#define HitPairGeneratorFromLayerPairCA_h

#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/interface/LayerDoubletsCache.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "RecoTracker/TkHitPairs/interface/HitDoubletsCA.h"

class DetLayer;
class TrackingRegion;

class HitPairGeneratorFromLayerPairCA {

public:

  typedef LayerHitMapCache LayerCacheType;
  typedef SeedingLayerSetsHits::SeedingLayerSet Layers;
  typedef SeedingLayerSetsHits::SeedingLayer Layer;
  
  HitPairGeneratorFromLayerPairCA():
    theOuterLayer(0),theInnerLayer(0),
    theMaxElement(0),theKDTreeCache(nullptr){};
    
  HitPairGeneratorFromLayerPairCA(unsigned int inner, unsigned int outer,unsigned int max=0):
            theOuterLayer(outer),theInnerLayer(inner),
            theMaxElement(max)
            {};

  ~HitPairGeneratorFromLayerPairCA();

  HitDoublets doublets ;

  //void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs,const edm::Event & ev,  const edm::EventSetup& es, Layers layers);
  /*
  Layer innerLayer(const Layers& layers) const { return layers[theInnerLayer]; }
  Layer outerLayer(const Layers& layers) const { return layers[theOuterLayer]; }*/

private:
  const unsigned int theOuterLayer;
  const unsigned int theInnerLayer;
  const unsigned int theMaxElement;
  FKDTree<float,2>* theInnerTree;
  FKDTree<float,2>* theOuterTree;
};



#endif
