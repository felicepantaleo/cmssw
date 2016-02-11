#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"

namespace LayerTriplets {
std::vector<LayerSetAndLayers> layers(const SeedingLayerSetsHits& sets) {
  std::vector<LayerSetAndLayers> result;
  if(sets.numberOfLayersInSet() != 3)
    return result;

  for(LayerSet set: sets) {
    bool added = false;

    for(auto ir = result.begin(); ir < result.end(); ++ir) {
      const LayerSet & resSet = ir->first;
      if (resSet[0].index() == set[0].index() && resSet[1].index() == set[1].index()) {
        std::vector<Layer>& thirds = ir->second;
        thirds.push_back( set[2] );
        added = true;
        break;
      }
    }
    if (!added) {
      LayerSetAndLayers lpl = std::make_pair(set,  std::vector<Layer>(1, set[2]) );
      result.push_back(lpl);
    }
  }
  return result;
}
    
    //Create a vector of 4-layer vectors
    CALayersSet  CAQuadrupleLayers(const SeedingLayerSetsHits& sets) {
        CALayersSet  result;
        if(sets.numberOfLayersInSet() != 4)
            return result;

        for(LayerSet thisSet: sets){
            
            CALayers thisCASet;
            for (auto iS = 0; iS < thisSet.size(); ++iS) {
                thisCASet.push_back(thisSet[iS]);
            }
            result.push_back(thisCASet);
        }
        return result;
    }
    
    //Create a vector of which each memeber is a set of three couple each for a valid combination of layers to form doublets
    CALayersSetWithPairs  CAQuadruplePairsLayers(const SeedingLayerSetsHits& sets) {
        CALayersSetWithPairs result;
        if(sets.numberOfLayersInSet() != 4)
            return result;
        
        for(LayerSet thisSet: sets){
            CALayerPairs thisCAPairs;
            for (auto iS = 0; iS < thisSet.size() - 1; ++iS) {
                CALayerPair bufferPair(thisSet[iS],thisSet[iS+1]);
                thisCAPairs.push_back(bufferPair);
            }
            result.push_back(thisCAPairs);
        }
        return result;
    }

    
    
}
