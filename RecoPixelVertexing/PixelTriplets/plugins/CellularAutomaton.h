#ifndef RECOTRACKER_TKHITPAIRS_INTERFACE_CELLULARAUTOMATON_H_
#define RECOTRACKER_TKHITPAIRS_INTERFACE_CELLULARAUTOMATON_H_
#include <array>
#include "CACell.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

class CellularAutomaton {
public:

    CellularAutomaton() :
    theNumberOfLayers(4) {

    }

    CellularAutomaton(const unsigned int numberOfLayers) :
    theNumberOfLayers(numberOfLayers) {

        isOuterHitOfCell.resize(theNumberOfLayers);
        theFoundCellsPerLayer.resize(theNumberOfLayers);

    }



    void create_and_connect_cells(std::vector<const HitDoublets*>, const SeedingLayerSetsHits::SeedingLayerSet& , const float);
    void evolve();
    void find_root_cells(const unsigned int);
    void find_ntuplets(std::vector<CACell::CAntuplet>& , const unsigned int );



private:
    const unsigned int theNumberOfLayers;


    //for each hit in each layer, store the pointers of the Cells of which it is outerHit
    std::vector<std::vector<std::vector<CACell*> > > isOuterHitOfCell;
    std::vector<std::vector<CACell> > theFoundCellsPerLayer;

    std::vector<CACell*> theRootCells;
    std::vector<std::vector<CACell*> > theNtuplets;

};

#endif 
