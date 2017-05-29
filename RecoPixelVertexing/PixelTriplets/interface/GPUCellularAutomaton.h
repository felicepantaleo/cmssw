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
#include "RecoPixelVertexing/PixelTriplets/plugins/CAGraph.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUMemoryManager.h"

template<unsigned int maxNumberOfQuadruplets>
class GPUCellularAutomaton
{
    public:

        GPUCellularAutomaton(TrackingRegion const & region, float thetaCut, float phiCut,
                float hardPtCut, const CAGraph& graph ) :
                thePtMin
                { region.ptMin() }, theRegionOriginX
                { region.origin().x() }, theRegionOriginY
                { region.origin().y() }, theRegionOriginRadius
                { region.originRBound() }, theThetaCut
                { thetaCut }, thePhiCut
                { phiCut }, theHardPtCut
                { hardPtCut }, theLayerGraph(graph)
        {

        }

        void run(const std::vector<const HitDoublets *>& host_hitDoublets,
                std::vector<std::array<std::array<int, 2>, 3> > & quadruplets);
        GPUMemoryManager theGpuMem;

    private:

        const float thePtMin;
        const float theRegionOriginX;
        const float theRegionOriginY;
        const float theRegionOriginRadius;
        const float theThetaCut;
        const float thePhiCut;
        const float theHardPtCut;

        CAGraph theLayerGraph;


};

#endif
