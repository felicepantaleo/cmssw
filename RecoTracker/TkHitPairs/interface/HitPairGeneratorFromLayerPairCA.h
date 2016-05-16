#ifndef HitPairGeneratorFromLayerPairCA_H
#define HitPairGeneratorFromLayerPairCA_H

#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "RecoTracker/TkHitPairs/interface/HitDoubletsCA.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/FKDTree.h"

class DetLayer;
class TrackingRegion;

using LayerTree = FKDTree<float,3>;
using LayerPoint = FKDPoint<float,3>;

class HitPairGeneratorFromLayerPairCA {

public:

  typedef SeedingLayerSetsHits::SeedingLayerSet Layers;
  typedef SeedingLayerSetsHits::SeedingLayer Layer;
  
  HitPairGeneratorFromLayerPairCA():
    theOuterLayer(0),theInnerLayer(0),
    theMaxElement(0){};
    
  HitPairGeneratorFromLayerPairCA(unsigned int inner, unsigned int outer,unsigned int max=0):
            theOuterLayer(outer),theInnerLayer(inner),
            theMaxElement(max)
            {};

  ~HitPairGeneratorFromLayerPairCA();

  HitDoubletsCA doublets (const TrackingRegion& reg, const edm::Event & ev,  const edm::EventSetup& es,
                          const SeedingLayerSetsHits::SeedingLayer& innerLayer,const SeedingLayerSetsHits::SeedingLayer& outerLayer,LayerTree & outerTree);

  //void hitPairs( const TrackingRegion& reg, OrderedHitPairs & prs,const edm::Event & ev,  const edm::EventSetup& es, Layers layers);
  /*
  Layer innerLayer(const Layers& layers) const { return layers[theInnerLayer]; }
  Layer outerLayer(const Layers& layers) const { return layers[theOuterLayer]; }*/

private:
  const unsigned int theOuterLayer;
  const unsigned int theInnerLayer;
  const unsigned int theMaxElement;
};



#endif
