#ifndef CANtupleHLTGenerator_H
#define CANtupleHLTGenerator_H

#include "RecoPixelVertexing/PixelTriplets/interface/HitQuadrupletGenerator.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CANtupleGenerator.h"

#include <utility>
#include <vector>

class SeedComparitor;

class CANtupleHLTGenerator : public CANtupleGenerator {

public:
  CANtupleHLTGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  virtual ~CANtupleHLTGenerator();

    virtual void getNTuplets(const TrackingRegion& region, OrderedSeedingHits & quads,
                                const edm::Event & ev, const edm::EventSetup& es,
                                SeedingLayerSetsHits::SeedingLayerSet fourLayers,
                                ) override;

private:
  const bool useFixedPreFiltering;
  const float extraHitRZtolerance;
  const float extraHitRPhitolerance;
  const bool useMScat;
  const bool useBend;
  const float dphi;
  std::unique_ptr<SeedComparitor> theComparitor;

};
#endif


