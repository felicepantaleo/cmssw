#ifndef CANtupleGenerator_H
#define CANtupleGenerator_H

#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitQuadruplets.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkHitPairs/interface/LayersCACellsCache.h"
#include "RecoTracker/TkHitPairs/interface/LayerFKDTreeCache.h"
#include <vector>

namespace edm { class ParameterSet; class Event; class EventSetup; class ConsumesCollector; }
class TrackingRegion;

class CANtupleGenerator {

public:

  explicit CANtupleGenerator(unsigned int maxElement=0);
  explicit CANtupleGenerator(const edm::ParameterSet& pset);
  virtual ~CANtupleGenerator();

  void init(LayerFKDTreeCache *kdTReeCache,LayersCACellsCache *caCellsCache);

  virtual void getNTuplets( const TrackingRegion& region, OrderedSeedingHits & ntuplets,
                            const edm::Event & ev, const edm::EventSetup& es,
                            SeedingLayerSetsHits::SeedingLayerSet layers,
                            ) = 0;
protected:
    
  LayerFKDTreeCache *theKDTReeCache;
  LayersCACellsCache *theCACellsCache;
    
  const unsigned int theMaxElement;
};
#endif


