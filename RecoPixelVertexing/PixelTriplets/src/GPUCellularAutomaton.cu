#include <vector>
#include <array>
#include "RecoPixelVertexing/PixelTriplets/interface/GPUSimpleVector.h"
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
            auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * i + 1];
            thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers, layerPairIndex, i,
                    gpuDoublets[layerPairIndex].indices[2 * i], outerHitId);

            isOuterHitOfCell[outerLayerId][outerHitId].push_back_ts(&(thisCell));
        }
    }
}

__global__
void debug_input_data(const GPULayerDoublets* gpuDoublets, int numberOfLayerPairs)
{
    printf("GPU: numberOfLayerPairs: %d\n", numberOfLayerPairs);

    for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs; ++layerPairIndex)
    {
        printf("\t numberOfDoublets: %d \n", gpuDoublets[layerPairIndex].size);
        printf("\t innerLayer: %d outerLayer: %d \n", gpuDoublets[layerPairIndex].innerLayerId,
                gpuDoublets[layerPairIndex].outerLayerId);

        for (unsigned int cellIndexInLayerPair = 0;
                cellIndexInLayerPair < gpuDoublets[layerPairIndex].size; ++cellIndexInLayerPair)
        {
            printf("\t \t %d innerHit: %d outerHit: %d \n", cellIndexInLayerPair,
                    gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair],
                    gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair + 1]);

        }

    }
}

__global__
void kernel_create_debug(const GPULayerDoublets* gpuDoublets, const GPULayerHits* gpuHitsOnLayers,
        GPUCACell** cells, GPUSimpleVector<100, GPUCACell*> ** isOuterHitOfCell,
        unsigned int numberOfLayerPairs)
{

    for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs; ++layerPairIndex)
    {
        printf("\t numberOfDoublets: %d \n", gpuDoublets[layerPairIndex].size);

        for (unsigned int cellIndexInLayerPair = 0;
                cellIndexInLayerPair < gpuDoublets[layerPairIndex].size; ++cellIndexInLayerPair)
        {
            int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;

            auto& thisCell = cells[layerPairIndex][cellIndexInLayerPair];
            auto& layerMap = isOuterHitOfCell[outerLayerId];
            auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair + 1];

            thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers, layerPairIndex,
                    cellIndexInLayerPair,
                    gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair], outerHitId);
//            thisCell.print_cell();
            layerMap[outerHitId].push_back(&(thisCell));

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

                    otherCell->theOuterNeighbors.push_back_ts(&thisCell);
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
        int* rootLayerPairs, int numberOfRootLayerPairs,
        unsigned int minHitsPerNtuplet)
{
    if(blockIdx.y < numberOfRootLayerPairs)
    {
        unsigned int cellIndexInRootLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int rootLayerPairIndex = rootLayerPairs[blockIdx.y];

        GPUSimpleVector<3, GPUCACell*> stack;

        for (int i = cellIndexInRootLayerPair; i < gpuDoublets[rootLayerPairIndex].size;
                i += gridDim.x * blockDim.x)
        {

            stack.reset();
            stack.push_back(&cells[rootLayerPairIndex][i]);
            cells[rootLayerPairIndex][i].find_ntuplets(foundNtuplets, stack, minHitsPerNtuplet);

        }

    }
}
//
//template<int maxNumberOfQuadruplets>
//__global__
//void kernel_find_ntuplets_unrolled_recursion(const GPULayerDoublets* gpuDoublets,
//        GPUCACell** cells,
//        GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets,
//        int* externalLayerPairs, int numberOfExternalLayerPairs,
//        unsigned int minHitsPerNtuplet)
//{
//
//    unsigned int cellIndexInLastLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
//    unsigned int lastLayerPairIndex = externalLayerPairs[blockIdx.y];
//
//    for (int i = cellIndexInLastLayerPair; i < gpuDoublets[lastLayerPairIndex].size;
//            i += gridDim.x * blockDim.x)
//    {
//
//        GPUCACell * root = &cells[lastLayerPairIndex][i];
//        Quadruplet tmpQuadruplet;
//        // the building process for a track ends if:
//        // it has no right neighbor
//        // it has no compatible neighbor
//        // the ntuplets is then saved if the number of hits it contains is greater than a threshold
//
//        GPUCACell * firstCell;
//        GPUCACell * secondCell;
//
//        tmpQuadruplet.layerPairsAndCellId[2].x = root->theLayerPairId;
//        tmpQuadruplet.layerPairsAndCellId[2].y = root->theDoubletId;
//        for (int j = 0; j < root->theInnerNeighbors.size(); ++j)
//        {
//            firstCell = root->theInnerNeighbors.m_data[j];
//            tmpQuadruplet.layerPairsAndCellId[1].x = firstCell->theLayerPairId;
//            tmpQuadruplet.layerPairsAndCellId[1].y = firstCell->theDoubletId;
//            for (int k = 0; j < firstCell->theInnerNeighbors.size(); ++j)
//            {
//                secondCell = firstCell->theInnerNeighbors.m_data[k];
//                tmpQuadruplet.layerPairsAndCellId[1].x = secondCell->theLayerPairId;
//                tmpQuadruplet.layerPairsAndCellId[1].y = secondCell->theDoubletId;
//                foundNtuplets->push_back_ts(tmpQuadruplet);
//
//            }
//
//        }
//    }
//}

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
        std::vector<std::array<std::array<int, 2>, 3> > & quadruplets)
{
    constexpr int maxCellsPerHit = 100;
//    theGpuMem.allocate(1 << 19, GPUMemoryManager::host);
//    theGpuMem.allocate(1 << 19, GPUMemoryManager::device);

    std::vector<int> rootLayerPairs;
    int rootLayerPairsNum = 0;
    for (auto& layer : theLayerGraph.theRootLayers)
    {

        for (auto& layerPair : theLayerGraph.theLayers[layer].theOuterLayerPairs)
        {
            rootLayerPairsNum++;
            rootLayerPairs.push_back(layerPair);
        }
    }
    int* device_rootLayerPairs;
    int* host_rootLayerPairs;

    cudaMalloc(&device_rootLayerPairs, rootLayerPairsNum * sizeof(int));
    cudaMallocHost(&host_rootLayerPairs, rootLayerPairsNum * sizeof(int));

    for (int i = 0; i < rootLayerPairsNum; ++i)
    {

        host_rootLayerPairs[i] = rootLayerPairs[i];
    }

    cudaMemcpyAsync(device_rootLayerPairs, host_rootLayerPairs, rootLayerPairsNum * sizeof(int),
            cudaMemcpyHostToDevice);

//    int* device_rootLayerPairs = (int*) (theGpuMem.requestMemory(rootLayerPairsNum * sizeof(int),
//            GPUMemoryManager::host));
//    int* gpu_rootLayerPairs = (int*) (theGpuMem.requestMemory(rootLayerPairsNum * sizeof(int),
//            GPUMemoryManager::device));
//
//    cudaMemcpyAsync(gpu_rootLayerPairs, device_rootLayerPairs, rootLayerPairsNum * sizeof(int),
//            cudaMemcpyHostToDevice, 0);

//    GPULayerDoublets* gpu_doublets = (GPULayerDoublets*) (theGpuMem.requestMemory(
//            theLayerGraph.theLayerPairs.size() * sizeof(GPULayerDoublets), GPUMemoryManager::device));
//    GPULayerDoublets* host_doublets = (GPULayerDoublets*) (theGpuMem.requestMemory(
//            theLayerGraph.theLayerPairs.size() * sizeof(GPULayerDoublets), GPUMemoryManager::host));

    GPULayerDoublets* device_doublets;
    GPULayerDoublets* host_doublets;

    cudaMalloc(&device_doublets, theLayerGraph.theLayerPairs.size() * sizeof(GPULayerDoublets));
    cudaMallocHost(&host_doublets, theLayerGraph.theLayerPairs.size() * sizeof(GPULayerDoublets));

    for (int i = 0; i < theLayerGraph.theLayerPairs.size(); ++i)
    {
//        std::cout << " allocating on layer pair " << i << " number of doublets " << host_hitDoublets.at(i)->size() << std::endl;
        host_doublets[i].size = host_hitDoublets.at(i)->size();
        host_doublets[i].innerLayerId = theLayerGraph.theLayerPairs[i].theLayers[0];
        host_doublets[i].outerLayerId = theLayerGraph.theLayerPairs[i].theLayers[1];

        std::size_t memsize = host_hitDoublets[i]->size() * sizeof(int) * 2;
//
        cudaMalloc(&(host_doublets[i].indices), memsize);
        cudaMemcpy(host_doublets[i].indices, host_hitDoublets[i]->indeces.data(), memsize,
                cudaMemcpyHostToDevice);

    }

    cudaMemcpyAsync(device_doublets, host_doublets,
            theLayerGraph.theLayerPairs.size() * sizeof(GPULayerDoublets), cudaMemcpyHostToDevice);

    GPULayerHits* device_hits;
    cudaMalloc(&device_hits, theLayerGraph.theLayers.size() * sizeof(GPULayerHits));
    GPULayerHits* host_hits;
    cudaMallocHost(&host_hits, theLayerGraph.theLayers.size() * sizeof(GPULayerHits));

    for (int i = 0; i < theLayerGraph.theLayers.size(); ++i)
    {
        host_hits[i].layerId = i;
        host_hits[i].size = theLayerGraph.theLayers[i].isOuterHitOfCell.size();
//        std::cout << "allocating or layer " << i << " number of hits " << theLayerGraph.theLayers[i].isOuterHitOfCell.size() << std::endl;
        auto memsize = theLayerGraph.theLayers[i].isOuterHitOfCell.size() * sizeof(float);
        cudaMalloc(&(host_hits[i].x), memsize);
        cudaMalloc(&(host_hits[i].y), memsize);
        cudaMalloc(&(host_hits[i].z), memsize);

        cudaMemcpy(host_hits[i].x, theLayerGraph.theLayers[i].hits->x.data(), memsize,
                cudaMemcpyHostToDevice);
        cudaMemcpy(host_hits[i].y, theLayerGraph.theLayers[i].hits->y.data(), memsize,
                cudaMemcpyHostToDevice);
        cudaMemcpy(host_hits[i].z, theLayerGraph.theLayers[i].hits->z.data(), memsize,
                cudaMemcpyHostToDevice);

    }

    cudaMemcpyAsync(device_hits, host_hits, theLayerGraph.theLayers.size() * sizeof(GPULayerHits),
            cudaMemcpyHostToDevice);

    GPUSimpleVector<maxCellsPerHit, GPUCACell*> ** device_isOuterHitOfCell;
    GPUSimpleVector<maxCellsPerHit, GPUCACell*> ** tmp_isOuterHitOfCell;

    cudaMalloc(&device_isOuterHitOfCell,
            theLayerGraph.theLayers.size() * sizeof(GPUSimpleVector<maxCellsPerHit, GPUCACell*> *));
    cudaMallocHost(&tmp_isOuterHitOfCell,
            theLayerGraph.theLayers.size() * sizeof(GPUSimpleVector<maxCellsPerHit, GPUCACell*> *));

    for (unsigned int i = 0; i < theLayerGraph.theLayers.size(); ++i)
    {
        cudaMalloc(&tmp_isOuterHitOfCell[i],
                theLayerGraph.theLayers[i].isOuterHitOfCell.size()
                        * sizeof(GPUSimpleVector<maxCellsPerHit, GPUCACell*> ));
        cudaMemset(tmp_isOuterHitOfCell[i], 0,
                theLayerGraph.theLayers[i].isOuterHitOfCell.size()
                        * sizeof(GPUSimpleVector<maxCellsPerHit, GPUCACell*> ));

    }

    cudaMemcpyAsync(device_isOuterHitOfCell, tmp_isOuterHitOfCell,
            theLayerGraph.theLayers.size() * sizeof(GPUSimpleVector<maxCellsPerHit, GPUCACell*> *),
            cudaMemcpyHostToDevice);

    GPUCACell **device_theCells;
    GPUCACell **tmp_theCells;

    cudaMalloc(&device_theCells, theLayerGraph.theLayerPairs.size() * sizeof(GPUCACell *));
    cudaMallocHost(&tmp_theCells, theLayerGraph.theLayerPairs.size() * sizeof(GPUCACell *));

    for (unsigned int i = 0; i < theLayerGraph.theLayerPairs.size(); ++i)
    {
        cudaMalloc(&tmp_theCells[i], host_hitDoublets[i]->size() * sizeof(GPUCACell));
    }
    cudaMemcpyAsync(device_theCells, tmp_theCells,
            theLayerGraph.theLayerPairs.size() * sizeof(GPUCACell *), cudaMemcpyHostToDevice);

//
//    cudaMemcpy(gpu_doublets, host_doublets, theLayerGraph.theLayerPairs.size() * sizeof(GPULayerDoublets),
//            cudaMemcpyHostToDevice);

//    GPULayerHits* gpu_hits = (GPULayerHits*) (theGpuMem.requestMemory(
//            theLayerGraph.theLayers.size() * sizeof(GPULayerHits), GPUMemoryManager::device));
//    GPULayerHits* host_hits = (GPULayerHits*) (theGpuMem.requestMemory(
//            theLayerGraph.theLayers.size() * sizeof(GPULayerHits), GPUMemoryManager::host));
//
//    for (int i = 0; i < theLayerGraph.theLayers.size(); ++i)
//    {
//        host_hits[i].layerId = i;
//        host_hits[i].size = hitsOnLayer[i]->size();
//        auto memsize = host_hits[i].size * sizeof(float);
//
//        host_hits[i].x = (float*) (theGpuMem.requestMemory(memsize, GPUMemoryManager::device));
//        host_hits[i].y = (float*) (theGpuMem.requestMemory(memsize, GPUMemoryManager::device));
//        host_hits[i].z = (float*) (theGpuMem.requestMemory(memsize, GPUMemoryManager::device));
//
//        cudaMemcpy(host_hits[i].x, hitsOnLayer[i]->x.data(), memsize, cudaMemcpyHostToDevice);
//        cudaMemcpy(host_hits[i].y, hitsOnLayer[i]->y.data(), memsize, cudaMemcpyHostToDevice);
//        cudaMemcpy(host_hits[i].z, hitsOnLayer[i]->z.data(), memsize, cudaMemcpyHostToDevice);
//    }
//    cudaMemcpy(gpu_hits, host_hits, theLayerGraph.theLayers.size() * sizeof(GPULayerHits),
//            cudaMemcpyHostToDevice);

//    std::vector<GPULayerDoublets> gpu_DoubletsVector;
//    std::vector<GPULayerHits> gpu_HitsVector;
////we first move the content of doublets
//    copy_hits_and_doublets_to_gpu(hitsOnLayer, host_hitDoublets, theLayerGraph, gpu_HitsVector,
//            gpu_DoubletsVector, theGpuMem);
////then we move the containers of the doublets

//
//    for (int i = 0; i < gpu_DoubletsVector.size(); ++i)
//    {
//
//        int* returnCheckDoublets;
//        cudaMallocHost(&returnCheckDoublets, memsize);
//
//        cudaMemcpy(returnCheckDoublets, gpu_DoubletsVector[i].indices, memsize, cudaMemcpyDeviceToHost);
//        for (unsigned int cellIndexInLayerPair = 0; cellIndexInLayerPair < gpu_DoubletsVector[i].size;
//                ++cellIndexInLayerPair)
//        {
//            printf("verify \t innerHit: %d outerHit: %d \n",
//                    returnCheckDoublets[2 * cellIndexInLayerPair],
//                    returnCheckDoublets[2 * cellIndexInLayerPair + 1]);
//        }
//        cudaFreeHost(returnCheckDoublets);
//    }

// and then we move the containers of the hits

//    GPULayerHits* gpu_layerHits = (GPULayerHits*) (theGpuMem.requestMemory(        gpu_HitsVector.size() * sizeof(GPULayerHits), GPUMemoryManager::device));
//
//    cudaMemcpy(gpu_layerHits, gpu_HitsVector.data(), gpu_HitsVector.size() * sizeof(GPULayerHits),
//            cudaMemcpyHostToDevice);

//    cudaMemcpy(gpu_rootLayerPairs, theRootLayerPairs.data(),
//            theRootLayerPairs.size() * sizeof(int), cudaMemcpyHostToDevice);

//    GPUSimpleVector<maxCellsPerHit, GPUCACell*> ** isOuterHitOfCell = (GPUSimpleVector<
//            maxCellsPerHit, GPUCACell*> **) (theGpuMem.requestMemory(
//            (theNumberOfLayers) * sizeof(GPUSimpleVector<maxCellsPerHit, GPUCACell*> *),
//            GPUMemoryManager::device));
//    GPUSimpleVector<maxCellsPerHit, GPUCACell*> ** host_isOuterHitOfCell = (GPUSimpleVector<
//            maxCellsPerHit, GPUCACell*> **) (theGpuMem.requestMemory(
//            (theNumberOfLayers) * sizeof(GPUSimpleVector<maxCellsPerHit, GPUCACell*> *),
//            GPUMemoryManager::host));
//
//    for (unsigned int i = 0; i < theNumberOfLayers; ++i)
//    {
//        host_isOuterHitOfCell[i] =
//                (GPUSimpleVector<maxCellsPerHit, GPUCACell*> *) (theGpuMem.requestMemory(
//                        hitsOnLayer[i]->size()
//                                * sizeof(GPUSimpleVector<maxCellsPerHit, GPUCACell*> ),
//                        GPUMemoryManager::device));
//        cudaMemset(host_isOuterHitOfCell[i], 0,
//                hitsOnLayer[i]->size() * sizeof(GPUSimpleVector<maxCellsPerHit, GPUCACell*> ));
//
//    }
//    cudaMemcpyAsync(isOuterHitOfCell, host_isOuterHitOfCell,
//            (theNumberOfLayers) * sizeof(GPUSimpleVector<maxCellsPerHit, GPUCACell*> *),
//            cudaMemcpyHostToDevice, 0);
//
//    GPUCACell **theCells = (GPUCACell **) (theGpuMem.requestMemory(
//            theNumberOfLayerPairs * sizeof(GPUCACell *), GPUMemoryManager::device));
//    GPUCACell **host_Cells = (GPUCACell **) (theGpuMem.requestMemory(
//            theNumberOfLayerPairs * sizeof(GPUCACell *), GPUMemoryManager::host));
//
//    for (unsigned int i = 0; i < theNumberOfLayerPairs; ++i)
//    {
//        host_Cells[i] = (GPUCACell *) (theGpuMem.requestMemory(
//                host_hitDoublets[i]->size() * sizeof(GPUCACell), GPUMemoryManager::device));
//    }

    GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> * foundNtuplets;
    GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> * host_foundNtuplets;
    cudaMallocManaged(&foundNtuplets, sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ));

//    cudaMallocHost(&host_foundNtuplets,
//            sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ));
//    GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> * foundNtuplets = (GPUSimpleVector<
//            maxNumberOfQuadruplets, Quadruplet> *) (theGpuMem.requestMemory(
//            sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ), GPUMemoryManager::device));
//    GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> * host_foundNtuplets = (GPUSimpleVector<
//            maxNumberOfQuadruplets, Quadruplet> *) (theGpuMem.requestMemory(
//            sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ), GPUMemoryManager::host));

//    cudaDeviceSetCacheConfig (cudaFuncCachePreferL1);
//	cudaFuncSetCacheConfig(kernel_create, cudaFuncCachePreferL1);
//	cudaFuncSetCacheConfig(kernel_connect, cudaFuncCachePreferL1);
//	cudaFuncSetCacheConfig(kernel_find_ntuplets_unrolled_recursion, cudaFuncCachePreferL1);
//can become async
//    gpuErrchk(
//            cudaMemcpy(theCells, host_Cells, (theNumberOfLayerPairs) * sizeof(GPUCACell *),
//                    cudaMemcpyHostToDevice));
//
    dim3 numberOfBlocks_create(32, host_hitDoublets.size());
    dim3 numberOfBlocks_connect(32, host_hitDoublets.size());
    dim3 numberOfBlocks_find(8, rootLayerPairsNum);
//    cudaStreamSynchronize(0);
//
//    gpuErrchk(cudaMemset(foundNtuplets, 0, sizeof(int)));

    kernel_create<<<numberOfBlocks_create,256>>>(device_doublets, device_hits, device_theCells, device_isOuterHitOfCell, host_hitDoublets.size());
//     debug_input_data<<<1,1>>>(device_doublets, host_hitDoublets.size());
//     kernel_create_debug<<<1,1>>>(device_doublets, device_hits, device_theCells, device_isOuterHitOfCell, host_hitDoublets.size());
    kernel_connect<<<numberOfBlocks_connect,256>>>(device_doublets, device_theCells, device_isOuterHitOfCell, thePtMin, theRegionOriginX, theRegionOriginY, theRegionOriginRadius, theThetaCut, thePhiCut, theHardPtCut, host_hitDoublets.size());

    kernel_find_ntuplets<<<numberOfBlocks_find,128>>>(device_doublets, device_theCells, foundNtuplets,device_rootLayerPairs, rootLayerPairsNum, 4 );
    cudaStreamSynchronize(0);

//
//can become async
////        gpuErrchk(
////            cudaMemcpy(host_foundNtuplets, foundNtuplets,
////                    sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ),
////                    cudaMemcpyDeviceToHost));
//    cudaMemcpyAsync(host_foundNtuplets, foundNtuplets,
//            sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ), cudaMemcpyDeviceToHost,0);
    cudaStreamSynchronize(0);
    quadruplets.resize(foundNtuplets->size());

//    std::cout << "number of quadruplets found: " << foundNtuplets->size() << std::endl;
    memcpy(quadruplets.data(), foundNtuplets->m_data,
            foundNtuplets->size() * sizeof(Quadruplet));

    cudaFree(device_rootLayerPairs);
    cudaFreeHost(host_rootLayerPairs);


    cudaFree(device_hits);

    for (unsigned int i = 0; i < theLayerGraph.theLayers.size(); ++i)
    {
        cudaFree(tmp_isOuterHitOfCell[i]);
        cudaFree(host_hits[i].x);
        cudaFree(host_hits[i].y);
        cudaFree(host_hits[i].z);

    }
    cudaFreeHost(host_hits);
    cudaFreeHost(tmp_isOuterHitOfCell);
    cudaFree(device_isOuterHitOfCell);
    for (unsigned int i = 0; i < theLayerGraph.theLayerPairs.size(); ++i)
    {
        cudaFree(tmp_theCells[i]);
        cudaFree(host_doublets[i].indices);
    }
    cudaFree(device_doublets);

    cudaFreeHost(tmp_theCells);
    cudaFreeHost(host_doublets);

    cudaFree(device_theCells);

    cudaFree(foundNtuplets);

    cudaStreamSynchronize(0);
}

template class GPUCellularAutomaton<3000> ;
template class GPUCellularAutomaton<1500> ;
