#ifndef HitQuadrupletGenerator_H
#define HitQuadrupletGenerator_H


/** A Hit Quadruplets Generator generating the set of possible layers combinations
    and the RecHitsKDTrees for all the layers.
 */

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitQuadruplets.h"

#include "FWCore/Utilities/interface/RunningAverage.h"

#include <vector>

#include <memory>
//#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "RecoTracker/TkHitPairs/interface/LayerFKDTreeCache.h"
#include "RecoTracker/TkHitPairs/interface/LayersCACellsCache.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <string>
#include <memory>

class TrackingRegion;
class CACellNtupleGenerator;
class SeedingLayerSetsHits;

namespace edm { class Event; class EventSetup; }

class HitQuadrupletGenerator : public OrderedHitsGenerator {
public:

  HitQuadrupletGenerator(unsigned int size=500);
  HitQuadrupletGenerator(HitQuadrupletGenerator const & other) : localRA(other.localRA.mean()){}

  HitQuadrupletGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);
  virtual ~HitQuadrupletGenerator() { }

  virtual const OrderedHitQuadruplets & run(
    const TrackingRegion& region, const edm::Event & ev, const edm::EventSetup& es) final;
    
  virtual void hitQuadruplets( const TrackingRegion& reg, OrderedHitQuadruplets & prs,
                                const edm::EventSetup& es);

  virtual void clear() final;

private:
  OrderedHitQuadruplets theQuadruplets;
    
  edm::EDGetTokenT<SeedingLayerSetsHits> theSeedingLayerToken;
  edm::RunningAverage localRA;
    
  std::unique_ptr<CACellNtupleGenerator> theGenerator;
    
  LayerFKDTreeCache            theKDTReeCache;
  //LayersCACellsCache           theCACellsCache;  //TODO
    
};

#endif
