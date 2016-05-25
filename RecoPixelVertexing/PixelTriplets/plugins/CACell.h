/*
 * CACell.h
 *
 *  Created on: Jan 29, 2016
 *      Author: fpantale
 */

#ifndef CACELL_H_
#define CACELL_H_

#include "RecoTracker/TkHitPairs/interface/HitDoubletsCA.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

#include <cmath>
#include <array>

class CACell {
public:

    using CAntuplet = std::vector<CACell>;


    CACell(const HitDoubletsCA* doublets, const SeedingLayerSetsHits::SeedingLayer& innerLayer, const SeedingLayerSetsHits::SeedingLayer& outerLayer, const unsigned int cellId, const int innerHitId, const int outerHitId) :
    theCAState(0), theInnerHitId(innerHitId), theOuterHitId(outerHitId), theCellId(cellId), hasSameStateNeighbors(0), theDoublets(doublets), theInnerLayer(innerLayer), theOuterLayer(outerLayer) {

    }
       CACell(const HitDoubletsCA* doublets, const SeedingLayerSetsHits::SeedingLayer& innerLayer, const SeedingLayerSetsHits::SeedingLayer& outerLayer, const int innerHitId, const int outerHitId) :
    theCAState(0), theInnerHitId(innerHitId), theOuterHitId(outerHitId), theCellId(0), hasSameStateNeighbors(0), theDoublets(doublets), theInnerLayer(innerLayer), theOuterLayer(outerLayer) {

    }
    
       
       unsigned int get_cell_id () const {
           return theCellId;
       }
       
    Hit const & get_inner_hit() const {
        return theInnerLayer.hits()[theInnerHitId];
    }

    Hit const & get_outer_hit() const {
        return theOuterLayer.hits()[theOuterHitId];
    }

    float get_inner_x() const {
        return theInnerLayer.hits()[theInnerHitId]->globalState().position.x();
    }

    float get_outer_x() const {
        return theOuterLayer.hits()[theOuterHitId]->globalState().position.x();
    }

    float get_inner_y() const {
        return theInnerLayer.hits()[theInnerHitId]->globalState().position.y();
    }

    float get_outer_y() const {
        return theOuterLayer.hits()[theOuterHitId]->globalState().position.y();
    }

    float get_inner_z() const {
        return theInnerLayer.hits()[theInnerHitId]->globalState().position.z();
    }

    float get_outer_z() const {
        return theOuterLayer.hits()[theOuterHitId]->globalState().position.z();
    }

    float get_inner_r() const {
        return theInnerLayer.hits()[theInnerHitId]->globalState().r;
    }

    float get_outer_r() const {
        return theOuterLayer.hits()[theOuterHitId]->globalState().r;
    }

    float get_inner_phi() const {
        return theInnerLayer.hits()[theInnerHitId]->globalState().position.phi();
    }

    float get_outer_phi() const {
        return theOuterLayer.hits()[theOuterHitId]->globalState().position.phi();
    }

    float get_inner_eta() const {
        return theInnerLayer.hits()[theInnerHitId]->globalState().position.eta();
    }

    float get_outer_eta() const {
        return theOuterLayer.hits()[theOuterHitId]->globalState().position.eta();
    }

    GlobalPoint inner_gp() const {
        return GlobalPoint(get_inner_x(), get_inner_y(), get_inner_z());
    }

    GlobalPoint outer_gp() const {
        return GlobalPoint(get_outer_x(), get_outer_y(), get_outer_z());
    }
    unsigned int get_inner_hit_id () const { return theInnerHitId; } 
    unsigned int get_outer_hit_id () const { return theOuterHitId; } 


    void check_alignment_and_tag(CACell*);
    void tag_as_inner_neighbor(CACell*);
    void tag_as_outer_neighbor(CACell*);

    bool are_aligned_RZ(const CACell*) const;


    bool is_root_cell(const unsigned int) const;
    
    void print_cell() const
    {
        std::cout << "\nprinting cell: " << theCellId << std::endl;
        std::cout << "CAState and hasSameStateNeighbors: " << theCAState <<" "<<  hasSameStateNeighbors << std::endl;

        std::cout << "inner hit Id: "  << theInnerHitId << " outer hit id: " << theOuterHitId << std::endl;
        
        std::cout << "it has inner and outer neighbors " << theInnerNeighbors.size() << " " << theOuterNeighbors.size() << std::endl; 
        std::cout << "its inner neighbors are: " << std::endl;
        for(unsigned int i = 0; i < theInnerNeighbors.size(); ++i)
            std::cout << theInnerNeighbors.at(i)->get_cell_id() << std::endl;
        
                std::cout << "its outer neighbors are: " << std::endl;
        for(unsigned int i = 0; i < theOuterNeighbors.size(); ++i)
            std::cout << theOuterNeighbors.at(i)->get_cell_id() << std::endl;
        
    }


    // if there is at least one left neighbor with the same state (friend), the state has to be increased by 1.

    unsigned int get_CA_state() const {
        return theCAState;
    }

    void evolve();
    void update_state();

    // trying to free the track building process from hardcoded layers, leaving the visit of the graph
    // based on the neighborhood connections between cells.
    void find_ntuplets(std::vector<CAntuplet>&, CAntuplet&, const unsigned int) const;

private:

    std::vector<CACell*> theInnerNeighbors;
    std::vector<CACell*> theOuterNeighbors;

    unsigned int theCAState;



    unsigned int theInnerHitId;
    unsigned int theOuterHitId;
    unsigned int theCellId;


    unsigned int hasSameStateNeighbors;
public:
    const HitDoubletsCA* theDoublets;
    const SeedingLayerSetsHits::SeedingLayer& theInnerLayer;
    const SeedingLayerSetsHits::SeedingLayer& theOuterLayer;


};




#endif /*CACELL_H_ */
