#ifndef GPU_CACELL_H_
#define GPU_CACELL_H_

#include "RecoPixelVertexing/PixelTriplets/interface/GPUHitsAndDoublets.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "GPUSimpleVector.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include <cmath>
#include <array>

struct Quadruplet {
	int2 layerPairsAndCellId[3];
};

class GPUCACell
{
public:



	__device__ GPUCACell()
	{

	}

	__device__
	void init(const GPULayerDoublets* doublets, const GPULayerHits* hitsOnLayer, int layerPairId, int doubletId,
			int innerHitId, int outerHitId)
	{

		theInnerHitId = innerHitId;
		theOuterHitId = outerHitId;

		theDoublets = doublets;

		theDoubletId = doubletId;
		theLayerPairId = layerPairId;

		auto innerLayerId = doublets->innerLayerId;
		auto outerLayerId = doublets->outerLayerId;


		theInnerX = hitsOnLayer[innerLayerId].x[doublets->indices[2 * doubletId]];
		theOuterX = hitsOnLayer[outerLayerId].x[doublets->indices[2 * doubletId + 1]];

		theInnerY = hitsOnLayer[innerLayerId].y[doublets->indices[2 * doubletId]];
		theOuterY = hitsOnLayer[outerLayerId].y[doublets->indices[2 * doubletId + 1]];

		theInnerZ = hitsOnLayer[innerLayerId].z[doublets->indices[2 * doubletId]];
		theOuterZ = hitsOnLayer[outerLayerId].z[doublets->indices[2 * doubletId + 1]];
		theInnerR = hypot(theInnerX, theInnerY);
		theOuterR = hypot(theOuterX, theOuterY);
		theInnerNeighbors.reset();
	}

	__device__
	float get_inner_x() const
	{
		return theInnerX;
	}
	__device__
	float get_outer_x() const
	{
		return theOuterX;
	}
	__device__
	float get_inner_y() const
	{
		return theInnerY;
	}
	__device__
	float get_outer_y() const
	{
		return theOuterY;
	}
	__device__
	float get_inner_z() const
	{
		return theInnerZ;
	}
	__device__
	float get_outer_z() const
	{
		return theOuterZ;
	}
	__device__
	float get_inner_r() const
	{
		return theInnerR;
	}
	__device__
	float get_outer_r() const
	{
		return theOuterR;
	}
	__device__
	unsigned int get_inner_hit_id() const
	{
		return theInnerHitId;
	}
	__device__
	unsigned int get_outer_hit_id() const
	{
		return theOuterHitId;
	}

	__host__ __device__
	void print_cell() const
	{

		printf("printing cell: %d, on layer: %d, innerHitId: %d, outerHitId: %d, innerradius %f, outerRadius %f \n",
				theDoubletId, theLayerPairId, theInnerHitId,
				theOuterHitId, theInnerR, theOuterR);

	}

	__host__ __device__
	void print_neighbors() const
	{
		printf("\n\tIt has %d innerneighbors: \n", theInnerNeighbors.m_size);
		for(int i =0; i< theInnerNeighbors.m_size; ++i)
		{
			printf("\n\t\t%d innerneighbor: \n\t\t", i);
			 theInnerNeighbors.m_data[i]->print_cell();

		}
	}


	__device__
	bool check_alignment_and_tag(const GPUCACell * innerCell,
			const float ptmin, const float region_origin_x,
			const float region_origin_y, const float region_origin_radius,
			const float thetaCut, const float phiCut, const float hardPtCut)
	{
		return (are_aligned_RZ(innerCell, ptmin, thetaCut)
				&& have_similar_curvature(innerCell,ptmin, region_origin_x, region_origin_y,
        				region_origin_radius, phiCut, hardPtCut));

	}

	__device__
	bool are_aligned_RZ(const GPUCACell * otherCell,
			const float ptmin, const float thetaCut ) const
	{

		float r1 = otherCell->get_inner_r();
		float z1 = otherCell->get_inner_z();
        float radius_diff = fabs(r1 - theOuterR);

        float distance_13_squared = radius_diff*radius_diff + (z1 - theOuterZ)*(z1 - theOuterZ);

        float pMin = ptmin*sqrt(distance_13_squared); //this needs to be divided by radius_diff later

        float tan_12_13_half_mul_distance_13_squared = fabs(z1 * (theInnerR - theOuterR) + theInnerZ * (theOuterR - r1) + theOuterZ * (r1 - theInnerR)) ;
        return tan_12_13_half_mul_distance_13_squared * pMin <= thetaCut * distance_13_squared * radius_diff;
	}

	__device__
	bool have_similar_curvature(const GPUCACell * otherCell,
			const float ptmin,
			const float region_origin_x, const float region_origin_y, const float region_origin_radius, const float phiCut, const float hardPtCut) const
	{
		auto x1 = otherCell->get_inner_x();
		auto y1 = otherCell->get_inner_y();

		auto x2 = get_inner_x();
		auto y2 = get_inner_y();

		auto x3 = get_outer_x();
		auto y3 = get_outer_y();

        float distance_13_squared = (x1 - x3)*(x1 - x3) + (y1 - y3)*(y1 - y3);
        float tan_12_13_half_mul_distance_13_squared = fabs(y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)) ;
        if(tan_12_13_half_mul_distance_13_squared * ptmin <= 1.0e-4f*distance_13_squared)
        {
        	return true;

        }

        //87 cm/GeV = 1/(3.8T * 0.3)

        //take less than radius given by the hardPtCut and reject everything below
        float minRadius = hardPtCut*87.f;

        auto det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2);


        auto offset = x2 * x2 + y2*y2;

        auto bc = (x1 * x1 + y1 * y1 - offset)*0.5f;

        auto cd = (offset - x3 * x3 - y3 * y3)*0.5f;



        auto idet = 1.f / det;

        auto x_center = (bc * (y2 - y3) - cd * (y1 - y2)) * idet;
        auto y_center = (cd * (x1 - x2) - bc * (x2 - x3)) * idet;

        auto radius = std::sqrt((x2 - x_center)*(x2 - x_center) + (y2 - y_center)*(y2 - y_center));

        if(radius < minRadius)
        	return false;
        auto centers_distance_squared = (x_center - region_origin_x)*(x_center - region_origin_x) + (y_center - region_origin_y)*(y_center - region_origin_y);
        auto region_origin_radius_plus_tolerance = region_origin_radius + phiCut;
        auto minimumOfIntersectionRange = (radius - region_origin_radius_plus_tolerance)*(radius - region_origin_radius_plus_tolerance);

        if (centers_distance_squared >= minimumOfIntersectionRange) {
            auto minimumOfIntersectionRange = (radius + region_origin_radius_plus_tolerance)*(radius + region_origin_radius_plus_tolerance);
            return centers_distance_squared <= minimumOfIntersectionRange;
        } else {

            return false;
        }



	}

	// trying to free the track building process from hardcoded layers, leaving the visit of the graph
	// based on the neighborhood connections between cells.

	template<int maxNumberOfQuadruplets>
	__device__
	void find_ntuplets(
			GPUSimpleVector<maxNumberOfQuadruplets,Quadruplet>* foundNtuplets,
			GPUSimpleVector<3, GPUCACell *>& tmpNtuplet,
			const unsigned int minHitsPerNtuplet
	) const
	{

		// the building process for a track ends if:
		// it has no right neighbor
		// it has no compatible neighbor
		// the ntuplets is then saved if the number of hits it contains is greater than a threshold

		GPUCACell * otherCell;
		if (theInnerNeighbors.size() == 0)
		{
			if (tmpNtuplet.size() >= minHitsPerNtuplet - 1)
			{
				Quadruplet tmpQuadruplet;

				for(int i = 0; i<3; ++i)
				{
					tmpQuadruplet.layerPairsAndCellId[i].x = tmpNtuplet.m_data[2-i]->theLayerPairId;
					tmpQuadruplet.layerPairsAndCellId[i].y = tmpNtuplet.m_data[2-i]->theDoubletId;


				}
				foundNtuplets->push_back_ts(tmpQuadruplet);

			}
			else
			return;
		}
		else
		{

			for (int j = 0; j < theInnerNeighbors.size(); ++j)
			{


				otherCell = theInnerNeighbors.m_data[j];
				tmpNtuplet.push_back(otherCell);
				otherCell->find_ntuplets(foundNtuplets, tmpNtuplet, minHitsPerNtuplet);
				tmpNtuplet.pop_back();

			}

		}
	}
	GPUSimpleVector<40, GPUCACell *> theInnerNeighbors;

	int theDoubletId;
	int theLayerPairId;
private:

	unsigned int theInnerHitId;
	unsigned int theOuterHitId;
	const GPULayerDoublets* theDoublets;
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
