#include <vector>
#include <array>

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUCellularAutomaton.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUCACell.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUHitsAndDoublets.h"

__global__
void kernel_create(const GPULayerDoublets* gpuDoublets, const GPULayerHits* gpuHitsOnLayers,
        GPUCACell** cells, GPUSimpleVector<100, GPUCACell*> ** isOuterHitOfCell,
        int numberOfLayerPairs)
{

    unsigned int layerPairIndex = blockIdx.y;
    unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
    if (layerPairIndex < numberOfLayerPairs)
    {
        int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;

        for (int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size;
                i += gridDim.x * blockDim.x)
        {
            auto& thisCell = cells[layerPairIndex][i];

            thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers, layerPairIndex, i,
                    gpuDoublets[layerPairIndex].indices[2 * i],
                    gpuDoublets[layerPairIndex].indices[2 * i + 1]);

            isOuterHitOfCell[outerLayerId][thisCell.get_outer_hit_id()].push_back_ts(&(thisCell));

        }
    }

}

__global__
void kernel_connect(const GPULayerDoublets* gpuDoublets, GPUCACell** cells,
        GPUSimpleVector<100, GPUCACell*> ** isOuterHitOfCell, const float ptmin,
        const float region_origin_x, const float region_origin_y, const float region_origin_radius,
        const float thetaCut, const float phiCut, const float hardPtCut,
        const int numberOfLayerPairs)
{
    unsigned int layerPairIndex = blockIdx.y;
    unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
    if (layerPairIndex < numberOfLayerPairs)
    {
        int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;

        for (int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size;
                i += gridDim.x * blockDim.x)
        {
            auto& thisCell = cells[layerPairIndex][i];
            auto innerHitId = thisCell.get_inner_hit_id();
            int numberOfPossibleNeighbors = isOuterHitOfCell[innerLayerId][innerHitId].size();
            for (int j = 0; j < numberOfPossibleNeighbors; ++j)
            {
                GPUCACell* otherCell = isOuterHitOfCell[innerLayerId][innerHitId].m_data[j];

                if (thisCell.check_alignment_and_tag(otherCell, ptmin, region_origin_x,
                        region_origin_y, region_origin_radius, thetaCut, phiCut, hardPtCut))
                {

                    thisCell.theInnerNeighbors.push_back(otherCell);
                }
            }
        }
    }
}

template<int maxNumberOfQuadruplets>
__global__
void kernel_find_ntuplets(const GPULayerDoublets* gpuDoublets,
        GPUCACell** cells,
        GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets,
        int* rootLayerPairs, int numberOfExternalLayerPairs,
        unsigned int minHitsPerNtuplet)
{

    unsigned int cellIndexInLastLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int lastLayerPairIndex = rootLayerPairs[blockIdx.y];

    GPUSimpleVector<3, GPUCACell*> stack;
    for (int i = cellIndexInLastLayerPair; i < gpuDoublets[lastLayerPairIndex].size;
            i += gridDim.x * blockDim.x)
    {

        stack.reset();
        stack.push_back(&cells[lastLayerPairIndex][i]);
        cells[lastLayerPairIndex][i].find_ntuplets(foundNtuplets, stack, minHitsPerNtuplet);

    }
}

template<int maxNumberOfQuadruplets>
__global__
void kernel_find_ntuplets_unrolled_recursion(const GPULayerDoublets* gpuDoublets,
        GPUCACell** cells,
        GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets,
        int* externalLayerPairs, int numberOfExternalLayerPairs,
        unsigned int minHitsPerNtuplet)
{

    unsigned int cellIndexInLastLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int lastLayerPairIndex = externalLayerPairs[blockIdx.y];

    for (int i = cellIndexInLastLayerPair; i < gpuDoublets[lastLayerPairIndex].size;
            i += gridDim.x * blockDim.x)
    {

        GPUCACell * root = &cells[lastLayerPairIndex][i];
        Quadruplet tmpQuadruplet;
        // the building process for a track ends if:
        // it has no right neighbor
        // it has no compatible neighbor
        // the ntuplets is then saved if the number of hits it contains is greater than a threshold

        GPUCACell * firstCell;
        GPUCACell * secondCell;

        tmpQuadruplet.layerPairsAndCellId[2].x = root->theLayerPairId;
        tmpQuadruplet.layerPairsAndCellId[2].y = root->theDoubletId;
        for (int j = 0; j < root->theInnerNeighbors.size(); ++j)
        {
            firstCell = root->theInnerNeighbors.m_data[j];
            tmpQuadruplet.layerPairsAndCellId[1].x = firstCell->theLayerPairId;
            tmpQuadruplet.layerPairsAndCellId[1].y = firstCell->theDoubletId;
            for (int k = 0; j < firstCell->theInnerNeighbors.size(); ++j)
            {
                secondCell = firstCell->theInnerNeighbors.m_data[k];
                tmpQuadruplet.layerPairsAndCellId[1].x = secondCell->theLayerPairId;
                tmpQuadruplet.layerPairsAndCellId[1].y = secondCell->theDoubletId;
                foundNtuplets->push_back_ts(tmpQuadruplet);

            }

        }
    }
}

//template<int maxNumberOfQuadruplets>
//__global__
//void kernel_recursive_cell_find_ntuplets(GPUCACell* cell,
//		GPUSimpleVector<3, GPUCACell *>& tmpNtuplet,
//		GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets,
//		unsigned int minHitsPerNtuplet)
//{
//
// 		// the building process for a track ends if:
//			// it has no right neighbor
//			// it has no compatible neighbor
//			// the ntuplets is then saved if the number of hits it contains is greater than a threshold
//			Quadruplet tmpQuadruplet;
//			GPUCACell * otherCell;
//
//			if (cell->theInnerNeighbors.size() == 0)
//			{
//				if (tmpNtuplet.size() >= minHitsPerNtuplet - 1)
//				{
//
//
//					for(int i = 0; i<3; ++i)
//					{
//						tmpQuadruplet.layerPairsAndCellId[i].x = tmpNtuplet.m_data[2-i]->theLayerPairId;
//						tmpQuadruplet.layerPairsAndCellId[i].y = tmpNtuplet.m_data[2-i]->theDoubletId;
//
//
//					}
//					foundNtuplets->push_back_ts(tmpQuadruplet);
//
//				}
//				else
//				return;
//			}
//			else
//			{
//				if(threadIdx.x <cell->theInnerNeighbors.size() )
//				{
//
//					otherCell = cell->theInnerNeighbors.m_data[threadIdx.x];
//					tmpNtuplet.push_back(otherCell);
//					kernel_recursive_cell_find_ntuplets<<<1,16>>>(otherCell, tmpNtuplet, foundNtuplets, minHitsPerNtuplet);
//					cudaDeviceSynchronize();
//					tmpNtuplet.pop_back();
//
//				}
//
//			}
//
//}
//template<int maxNumberOfQuadruplets>
//__global__
//void kernel_find_ntuplets_dyn_parallelism(const GPULayerDoublets* gpuDoublets,
//		GPUCACell** cells,
//		GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets,
//		int* externalLayerPairs, int numberOfExternalLayerPairs,
//		unsigned int minHitsPerNtuplet)
//{
//
//	unsigned int cellIndexInLastLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int lastLayerPairIndex = externalLayerPairs[blockIdx.y];
//
//	for (int i = cellIndexInLastLayerPair; i < gpuDoublets[lastLayerPairIndex].size;
//			i += gridDim.x * blockDim.x)
//	{
//		cells[lastLayerPairIndex][i].stack.reset();
//		cells[lastLayerPairIndex][i].stack.push_back(&cells[lastLayerPairIndex][i]);
//		kernel_recursive_cell_find_ntuplets<<<1,16>>>(&cells[lastLayerPairIndex][i], cells[lastLayerPairIndex][i].stack,
//				foundNtuplets, minHitsPerNtuplet);
//
//	}
//}

template<unsigned int maxNumberOfQuadruplets>
void GPUCellularAutomaton<maxNumberOfQuadruplets>::run(
        const std::vector<const HitDoublets *>& host_hitDoublets,
        const std::vector<const RecHitsSortedInPhi *>& hitsOnLayer, const CAGraph& graph,
        std::vector<std::array<std::array<int, 2>, 3> > & quadruplets)
{
    theGpuMem.allocate(1 << 28, GPUMemoryManager::host);
    theGpuMem.allocate(1 << 28, GPUMemoryManager::device);

    std::vector<GPULayerDoublets> gpu_DoubletsVector;
    std::vector<GPULayerHits> gpu_HitsVector;
//we first move the content of doublets
    copy_hits_and_doublets_to_gpu(hitsOnLayer, host_hitDoublets, graph, gpu_HitsVector,
            gpu_DoubletsVector);
//then we move the containers of the doublets
    GPULayerDoublets* gpu_doublets = (GPULayerDoublets*) (theGpuMem.requestMemory(
            gpu_DoubletsVector.size() * sizeof(GPULayerDoublets), GPUMemoryManager::device));

    cudaMemcpy(gpu_doublets, gpu_DoubletsVector.data(),
            gpu_DoubletsVector.size() * sizeof(GPULayerDoublets), cudaMemcpyHostToDevice);
// and then we move the containers of the hits

    GPULayerHits* gpu_layerHits = (GPULayerHits*) (theGpuMem.requestMemory(
            gpu_HitsVector.size() * sizeof(GPULayerHits), GPUMemoryManager::device));

    cudaMemcpy(gpu_layerHits, gpu_HitsVector.data(), gpu_HitsVector.size() * sizeof(GPULayerHits),
            cudaMemcpyHostToDevice);
    int* gpu_rootLayerPairs =thrust::raw_pointer_cast(theRootLayerPairs.data());

//    cudaMemcpy(gpu_rootLayerPairs, theRootLayerPairs.data(),
//            theRootLayerPairs.size() * sizeof(int), cudaMemcpyHostToDevice);

    GPUSimpleVector<100, GPUCACell*> ** isOuterHitOfCell =
            (GPUSimpleVector<100, GPUCACell*> **) (theGpuMem.requestMemory(
                    (theNumberOfLayers) * sizeof(GPUSimpleVector<100, GPUCACell*> *),
                    GPUMemoryManager::device));
    GPUSimpleVector<100, GPUCACell*> ** host_isOuterHitOfCell =
            (GPUSimpleVector<100, GPUCACell*> **) (theGpuMem.requestMemory(
                    (theNumberOfLayers) * sizeof(GPUSimpleVector<100, GPUCACell*> *),
                    GPUMemoryManager::host));

    for (unsigned int i = 0; i < theNumberOfLayers; ++i)
    {
        host_isOuterHitOfCell[i] = (GPUSimpleVector<100, GPUCACell*> *) (theGpuMem.requestMemory(
                hitsOnLayer[i]->size() * sizeof(GPUSimpleVector<100, GPUCACell*> ),
                GPUMemoryManager::device));
        cudaMemset(host_isOuterHitOfCell[i], 0,
                hitsOnLayer[i]->size() * sizeof(GPUSimpleVector<100, GPUCACell*> ));

    }
    cudaMemcpyAsync(isOuterHitOfCell, host_isOuterHitOfCell,
            (theNumberOfLayers) * sizeof(GPUSimpleVector<100, GPUCACell*> *),
            cudaMemcpyHostToDevice, 0);

    GPUCACell **theCells;
    GPUCACell **host_Cells;

    cudaMallocHost(&host_Cells, (theNumberOfLayerPairs) * sizeof(GPUCACell *));
    cudaMalloc(&theCells, (theNumberOfLayerPairs) * sizeof(GPUCACell *));

    for (unsigned int i = 0; i < theNumberOfLayerPairs; ++i)
    {
        cudaMalloc(&host_Cells[i], host_hitDoublets[i]->size() * sizeof(GPUCACell));

    }

    GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> * foundNtuplets;
    GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> * host_foundNtuplets;

    cudaMallocHost(&host_foundNtuplets,
            sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ));

    cudaMalloc(&foundNtuplets, sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ));

    cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);
//	cudaFuncSetCacheConfig(kernel_create, cudaFuncCachePreferL1);
//	cudaFuncSetCacheConfig(kernel_connect, cudaFuncCachePreferL1);
//	cudaFuncSetCacheConfig(kernel_find_ntuplets_unrolled_recursion, cudaFuncCachePreferL1);
    cudaMemcpyAsync(theCells, host_Cells, (theNumberOfLayerPairs) * sizeof(GPUCACell *),
            cudaMemcpyHostToDevice, 0);

    dim3 numberOfBlocks_create(16, theNumberOfLayerPairs);
    dim3 numberOfBlocks_connect(32, theNumberOfLayerPairs);
    dim3 numberOfBlocks_find(8, theRootLayerPairs.size());
    cudaStreamSynchronize(0);

    cudaMemset(foundNtuplets, 0, sizeof(int));

    kernel_create<<<numberOfBlocks_create,256>>>(gpu_doublets, gpu_layerHits, theCells, isOuterHitOfCell, theNumberOfLayerPairs);

    kernel_connect<<<numberOfBlocks_connect,256>>>(gpu_doublets, theCells, isOuterHitOfCell, thePtMin, theRegionOriginX, theRegionOriginY, theRegionOriginRadius, theThetaCut, thePhiCut, theHardPtCut, theNumberOfLayerPairs);

    kernel_find_ntuplets<<<numberOfBlocks_find,128>>>(gpu_doublets, theCells, foundNtuplets,gpu_rootLayerPairs, theRootLayerPairs.size(), 4 );

//	kernel_find_ntuplets_dyn_parallelism<<<numberOfBlocks,128>>>(gpu_doublets, theCells, foundNtuplets,gpu_externalLayerPairs, theExternalLayerPairs.size(), 4 );
// 	kernel_find_ntuplets_unrolled_recursion<<<numberOfBlocks_find,256>>>(gpu_doublets, theCells, foundNtuplets,gpu_externalLayerPairs, theExternalLayerPairs.size(), 4 );

    cudaMemcpyAsync(host_foundNtuplets, foundNtuplets,
            sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ), cudaMemcpyDeviceToHost,
            0);
    cudaStreamSynchronize(0);
    quadruplets.resize(host_foundNtuplets->size());

    memcpy(quadruplets.data(), host_foundNtuplets->m_data,
            host_foundNtuplets->size() * sizeof(Quadruplet));

    for (unsigned int i = 0; i < theNumberOfLayerPairs; ++i)
    {
        cudaFree(host_Cells[i]);

    }
    for (unsigned int i = 0; i < theNumberOfLayers; ++i)
    {

        cudaFree(host_isOuterHitOfCell[i]);
    }

    cudaFree(foundNtuplets);
    cudaFree(theCells);
    cudaFree(isOuterHitOfCell);
    cudaFreeHost(host_Cells);
    cudaFreeHost(host_isOuterHitOfCell);
    cudaFreeHost(host_foundNtuplets);

    for (int i = 0; i < theNumberOfLayerPairs; ++i)
    {
        free_gpu_doublets(gpu_DoubletsVector[i]);

    }
    for (int i = 0; i < theNumberOfLayers; ++i)
    {
        free_gpu_hits(gpu_HitsVector[i]);
    }

    theGpuMem.freeMemory(GPUMemoryManager::host);
    theGpuMem.freeMemory(GPUMemoryManager::device);

}

template class GPUCellularAutomaton<1500> ;
