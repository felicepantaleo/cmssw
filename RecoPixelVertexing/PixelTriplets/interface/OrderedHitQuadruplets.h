#ifndef OrderedHitQuadruplets_H
#define OrderedHitQuadruplets_H

#include <vector>
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitQuadruplet.h"
#include "RecoTracker/TkSeedingLayers/interface/OrderedSeedingHits.h"
#include <vector>

class OrderedHitQuadruplets : public std::vector<OrderedHitQuadruplet>, public OrderedSeedingHits {
public:

  virtual OrderedHitQuadruplets(){}

  virtual unsigned int size() const { return std::vector<OrderedHitQuadruplet>::size(); }

  virtual const OrderedHitQuadruplets & operator[](unsigned int i) const {
    return std::vector<OrderedHitQuadruplet>::operator[](i);
  }

};
#endif
