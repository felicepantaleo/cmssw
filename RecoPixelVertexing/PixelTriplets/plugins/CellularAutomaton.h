#ifndef RECOPIXELVERTEXING_PIXELTRIPLETS_PLUGINS_CELLULARAUTOMATON_H_
#define RECOPIXELVERTEXING_PIXELTRIPLETS_PLUGINS_CELLULARAUTOMATON_H_
#include <array>
#include "CACell.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

template<unsigned int theNumberOfLayers>
class CellularAutomaton {
public:

    CellularAutomaton() {

    }

    void createAndConnectCells(std::array<const HitDoublets*, theNumberOfLayers-1>, const SeedingLayerSetsHits::SeedingLayerSet&, const TrackingRegion&, const float, const float);
    void createAndConnectCells(std::array<const HitDoublets*, theNumberOfLayers-1>, const SeedingLayerSetsHits::SeedingLayerSet&, std::vector<CACell::CAntuplet>&, const TrackingRegion&, const float, const float);

    void evolve();
    void findNtuplets(std::vector<CACell::CAntuplet>&, const unsigned int);
    void findTriplets(std::array<const HitDoublets*, theNumberOfLayers-1>, const SeedingLayerSetsHits::SeedingLayerSet&, std::vector<CACell::CAntuplet>&, const TrackingRegion&, const float, const float);

private:



    //for each hit in each layer, store the pointers of the Cells of which it is outerHit
    std::array<std::vector<std::vector<CACell*> >, theNumberOfLayers> isOuterHitOfCell;
    std::array<std::vector<CACell>, theNumberOfLayers> theFoundCellsPerLayer;

    std::vector<CACell*> theRootCells;
    std::vector<std::vector<CACell*> > theNtuplets;

};


#endif 
