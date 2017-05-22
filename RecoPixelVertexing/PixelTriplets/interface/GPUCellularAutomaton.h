#ifndef GPUCELLULARAUTOMATON_H_
#define GPUCELLULARAUTOMATON_H_

#include <array>
#include <vector>
#include <cuda_runtime.h>
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/CACell.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUHitsAndDoublets.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "RecoPixelVertexing/PixelTriplets/interface/GPUMemoryManager.h"


template<unsigned int maxNumberOfQuadruplets>
class GPUCellularAutomaton {
public:

    GPUCellularAutomaton(TrackingRegion const & region, float thetaCut, float phiCut, float hardPtCut,
    		int numberOfLayers, int numberOfLayerPairs, const thrust::host_vector<int>& rootLayerPairs) :
      thePtMin{ region.ptMin() },
      theRegionOriginX{ region.origin().x() },
      theRegionOriginY{ region.origin().y() },
      theRegionOriginRadius{ region.originRBound() },
      theThetaCut{ thetaCut },
      thePhiCut{ phiCut },
	  theHardPtCut{ hardPtCut },
	  theNumberOfLayers{numberOfLayers},
	  theNumberOfLayerPairs{numberOfLayerPairs}
    {
	      theRootLayerPairs = rootLayerPairs;

    }

    void run(const std::vector<const HitDoublets *>& host_hitDoublets,
    		const std::vector<const RecHitsSortedInPhi  *>& hitsOnLayer,
    		const CAGraph& graph,
    		std::vector< std::array< std::array<int,2> , 3> > & quadruplets);

private:

    const float thePtMin;
    const float theRegionOriginX;
    const float  theRegionOriginY;
    const float  theRegionOriginRadius;
    const float  theThetaCut;
    const float  thePhiCut;
    const float  theHardPtCut;
    const int theNumberOfLayers;
    const int theNumberOfLayerPairs;
    GPUMemoryManager theGpuMem;

    thrust::device_vector<int> theRootLayerPairs;

};

#endif
