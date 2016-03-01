#ifndef OrderedHitQuadruplet_H
#define OrderedHitQuadruplet_H


/** \class OrderedHitTriplet 
 * Associate 4 RecHits into hit quadruplet of FirstHit,SecondHit,ThirdHit,FourthHit
 */

#include "RecoTracker/TkHitPairs/interface/OrderedHitPair.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"

class OrderedHitQuadruplet : public SeedingHitSet {

public:

  typedef SeedingHitSet::ConstRecHitPointer FirstRecHit;
  typedef SeedingHitSet::ConstRecHitPointer SecondRecHit;
  typedef SeedingHitSet::ConstRecHitPointer ThirdRecHit;
  typedef SeedingHitSet::ConstRecHitPointer FourthRecHit;


  OrderedHitQuadruplet(const FirstRecHit & first, const SecondRecHit & second, const ThirdRecHit & third, const FourthRecHit & fourth) : SeedingHitSet(first,second,third,fourth){}

  FirstRecHit    first() const { return get(0); }
  SecondRecHit  second() const { return get(1); }
  ThirdRecHit    third() const { return get(2); }
  FourthRecHit    fourth() const { return get(3); }

};

#endif
