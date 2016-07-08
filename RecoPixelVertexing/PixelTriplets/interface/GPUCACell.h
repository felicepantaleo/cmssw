#ifndef GPU_CACELL_H_
#define GPU_CACELL_H_

#include "RecoPixelVertexing/PixelTriplets/interface/GPUHitsAndDoublets.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "GPUSimpleVector.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "GPUArena.h"
#include <cmath>
#include <array>


template<int numberOfLayers>
class GPUCACell {
public:

    using CAntuplet = GPUSimpleVector<numberOfLayers, GPUCACell<numberOfLayers>>;
    __device__
    GPUCACell()
    {

    }

    __device__
	void init(const GPULayerDoublets* doublets,  int layerId, int doubletId, int innerHitId, int outerHitId)
    {

        theInnerHitId = innerHitId;
        theOuterHitId =outerHitId;

        theDoublets=doublets;

        theDoubletId=doubletId;
        theLayerIdInFourLayers=layerId;
        theInnerX=doublets->layers[0].x[doublets->indices[2*doubletId]];
        theOuterX=doublets->layers[1].x[doublets->indices[2*doubletId+1]];

        theInnerY=doublets->layers[0].y[doublets->indices[2*doubletId]];
        theOuterY=doublets->layers[1].y[doublets->indices[2*doubletId+1]];

        theInnerZ=doublets->layers[0].z[doublets->indices[2*doubletId]];
        theOuterZ=doublets->layers[1].z[doublets->indices[2*doubletId+1]];
    	theInnerR=hypot (theInnerX, theInnerY);
    	theOuterR=hypot (theOuterX, theOuterY);

    	if(theLayerIdInFourLayers == 1 && theDoubletId == 42)
    	printf("theLayerIdInFourLayers %d, theInnerHitId %d, theOuterHitId %d, theDoubletId %d, theInnerR %f, theOuterR %f \n",
    			theLayerIdInFourLayers, theInnerHitId , theOuterHitId , theDoubletId , theInnerR , theOuterR);


    }


    __device__
    float get_inner_x() const {
        return theInnerX;
    }
    __device__
    float get_outer_x() const {
        return theOuterX;
    }
    __device__
    float get_inner_y() const {
        return theInnerY;
    }
    __device__
    float get_outer_y() const {
        return theOuterY;
    }
    __device__
    float get_inner_z() const {
        return theInnerZ;
    }
    __device__
    float get_outer_z() const {
        return theOuterZ;
    }
    __device__
    float get_inner_r() const {
        return theInnerR;
    }
    __device__
    float get_outer_r() const {
        return theOuterR;
    }
    __device__
    unsigned int get_inner_hit_id() const {
        return theInnerHitId;
    }
    __device__
    unsigned int get_outer_hit_id() const {
        return theOuterHitId;
    }

    __device__
    bool check_alignment_and_tag(const GPUCACell<numberOfLayers>* innerCell, const float ptmin, const float region_origin_x, const float region_origin_y, const float region_origin_radius, const float thetaCut, const float phiCut) {



        return (are_aligned_RZ(innerCell, ptmin, thetaCut) && have_similar_curvature(innerCell, region_origin_x, region_origin_y, region_origin_radius, phiCut));

    }

    __device__
    bool are_aligned_RZ(const GPUCACell<numberOfLayers>* otherCell, const float ptmin, const float thetaCut) const {

        float r1 = otherCell->get_inner_r();
        float z1 = otherCell->get_inner_z();
        float distance_13_squared = (r1 - theOuterR)*(r1 - theOuterR) + (z1 - theOuterZ)*(z1 - theOuterZ);
        float tan_12_13_half = fabs(z1 * (theInnerR - theOuterR) + theInnerZ * (theOuterR - r1) + theOuterZ * (r1 - theInnerR)) / distance_13_squared;
        return tan_12_13_half * ptmin <= thetaCut;
    }

    __device__
    bool have_similar_curvature(const GPUCACell<numberOfLayers>* otherCell,
            const float region_origin_x, const float region_origin_y, const float region_origin_radius, const float phiCut) const {
        auto x1 = otherCell->get_inner_x();
        auto y1 = otherCell->get_inner_y();

        auto x2 = get_inner_x();
        auto y2 = get_inner_y();

        auto x3 = get_outer_x();
        auto y3 = get_outer_y();

        auto precision = 0.5f;
        auto offset = x2 * x2 + y2*y2;

        auto bc = (x1 * x1 + y1 * y1 - offset) / 2.f;

        auto cd = (offset - x3 * x3 - y3 * y3) / 2.f;

        auto det = (x1 - x2) * (y2 - y3) - (x2 - x3)* (y1 - y2);

        //points are aligned
        if (fabs(det) < precision)
            return true;

        auto idet = 1.f / det;

        auto x_center = (bc * (y2 - y3) - cd * (y1 - y2)) * idet;
        auto y_center = (cd * (x1 - x2) - bc * (x2 - x3)) * idet;

        auto radius = std::sqrt((x2 - x_center)*(x2 - x_center) + (y2 - y_center)*(y2 - y_center));
        auto centers_distance_squared = (x_center - region_origin_x)*(x_center - region_origin_x) + (y_center - region_origin_y)*(y_center - region_origin_y);

        auto minimumOfIntesectionRange = (radius - region_origin_radius)*(radius - region_origin_radius) - phiCut;

        if (centers_distance_squared >= minimumOfIntesectionRange) {
            auto maximumOfIntesectionRange = (radius + region_origin_radius)*(radius + region_origin_radius) + phiCut;
            return centers_distance_squared <= maximumOfIntesectionRange;
        } else {

            return false;
        }

    }





    // trying to free the track building process from hardcoded layers, leaving the visit of the graph
    // based on the neighborhood connections between cells.

    template<int maxNumberOfQuadruplets>
    __device__
    void find_ntuplets(
        GPUSimpleVector<maxNumberOfQuadruplets,GPUSimpleVector<4, int>>* foundNtuplets, 
        GPUArena<numberOfLayers-2,16,GPUCACell<numberOfLayers>>& theInnerNeighbors,
        GPUSimpleVector<4, GPUCACell<4>*>& tmpNtuplet, 
        const unsigned int minHitsPerNtuplet
    ) const {

        // the building process for a track ends if:
        // it has no right neighbor
        // it has no compatible neighbor
        // the ntuplets is then saved if the number of hits it contains is greater than a threshold
        GPUArenaIterator<16, GPUCACell<numberOfLayers>> innerNeighborsIterator = theInnerNeighbors.iterator(theLayerIdInFourLayers,theDoubletId);
        GPUCACell<numberOfLayers>* otherCell;
        GPUSimpleVector<4, int> found;

        if (innerNeighborsIterator.has_next() == 0) {
            if (tmpNtuplet.size() >= minHitsPerNtuplet - 1) {
                found.reset();
                found.push_back(tmpNtuplet.m_data[0]->get_inner_hit_id());
                found.push_back(tmpNtuplet.m_data[1]->get_inner_hit_id());
                found.push_back(tmpNtuplet.m_data[2]->get_inner_hit_id());
                found.push_back(tmpNtuplet.m_data[2]->get_outer_hit_id());
                foundNtuplets->push_back(found);
            }
            else
                return;
        } else {


          while (innerNeighborsIterator.has_next())
          {
            otherCell = innerNeighborsIterator.get_next();
            tmpNtuplet.push_back(otherCell);
            otherCell->find_ntuplets(foundNtuplets, theInnerNeighbors, tmpNtuplet, minHitsPerNtuplet);
            tmpNtuplet.pop_back();

          }

        }
    }


private:

    unsigned int theInnerHitId;
    unsigned int theOuterHitId;
    const GPULayerDoublets* theDoublets;
    int theDoubletId;
    int theLayerIdInFourLayers;
    float theInnerX;
    float theOuterX;
    float theInnerY;
    float theOuterY;
    float theInnerZ;
    float theOuterZ;
    float theInnerR;
    float theOuterR;


};


#endif /*CACELL_H_ */
