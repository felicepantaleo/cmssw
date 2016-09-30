#include <vector>
#include <array>

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUCellularAutomaton.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUCACell.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUHitsAndDoublets.h"

__global__
void kernel_create(const GPULayerDoublets* gpuDoublets,
		const GPULayerHits* gpuHitsOnLayers, GPUCACell** cells,
		GPUSimpleVector<80, GPUCACell*> ** isOuterHitOfCell,
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

			thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers,
					layerPairIndex, i,
					gpuDoublets[layerPairIndex].indices[2 * i],
					gpuDoublets[layerPairIndex].indices[2 * i + 1]);

			isOuterHitOfCell[outerLayerId][thisCell.get_outer_hit_id()].push_back_ts(
					&(thisCell));


		}
	}

}

__global__
void kernel_connect(const GPULayerDoublets* gpuDoublets, GPUCACell** cells,
		GPUSimpleVector<80, GPUCACell*> ** isOuterHitOfCell, const float ptmin,
		const float region_origin_x, const float region_origin_y,
		const float region_origin_radius, const float thetaCut,
		const float phiCut, const float hardPtCut, const int numberOfLayerPairs)
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
			int numberOfPossibleNeighbors =
					isOuterHitOfCell[innerLayerId][innerHitId].size();

			for (int j = 0; j < numberOfPossibleNeighbors; ++j)
			{
				GPUCACell* otherCell =
						isOuterHitOfCell[innerLayerId][innerHitId].m_data[j];

				if (thisCell.check_alignment_and_tag(otherCell, ptmin,
						region_origin_x, region_origin_y, region_origin_radius,
						thetaCut, phiCut, hardPtCut))
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
		int* externalLayerPairs, int numberOfExternalLayerPairs,
		unsigned int minHitsPerNtuplet)
{

	unsigned int cellIndexInLastLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int lastLayerPairIndex = externalLayerPairs[blockIdx.y];

	GPUSimpleVector<3, GPUCACell*> stack;
	for (int i = cellIndexInLastLayerPair; i < gpuDoublets[lastLayerPairIndex].size;
			i += gridDim.x * blockDim.x)
	{

		stack.reset();
		stack.push_back(&cells[lastLayerPairIndex][i]);
		cells[lastLayerPairIndex][i].find_ntuplets(foundNtuplets, stack, minHitsPerNtuplet);

	}
}

template<unsigned int maxNumberOfQuadruplets>
void GPUCellularAutomaton<maxNumberOfQuadruplets>::run(
		const std::vector<HitDoublets>& host_hitDoublets,
		const std::vector<const RecHitsSortedInPhi *>& hitsOnLayer,
		const CAGraph& graph,
		std::vector<std::array<std::array<int, 2>, 3> > & quadruplets)
{

	std::vector<GPULayerDoublets> gpu_DoubletsVector;
	std::vector<GPULayerHits> gpu_HitsVector;
//we first move the content of doublets
	copy_hits_and_doublets_to_gpu(hitsOnLayer, host_hitDoublets, graph,
			gpu_HitsVector, gpu_DoubletsVector);
//then we move the containers of the doublets
	GPULayerDoublets* gpu_doublets;
	cudaMalloc(&gpu_doublets,
			gpu_DoubletsVector.size() * sizeof(GPULayerDoublets));

	cudaMemcpy(gpu_doublets, gpu_DoubletsVector.data(),
			gpu_DoubletsVector.size() * sizeof(GPULayerDoublets),
			cudaMemcpyHostToDevice);
// and then we move the containers of the hits

	GPULayerHits* gpu_layerHits;
	cudaMalloc(&gpu_layerHits, gpu_HitsVector.size() * sizeof(GPULayerHits));
	cudaMemcpy(gpu_layerHits, gpu_HitsVector.data(),
			gpu_HitsVector.size() * sizeof(GPULayerHits),
			cudaMemcpyHostToDevice);
	int* gpu_externalLayerPairs;
//

	cudaMalloc(&gpu_externalLayerPairs,
			theExternalLayerPairs.size() * sizeof(int));

	cudaMemcpy(gpu_externalLayerPairs, theExternalLayerPairs.data(),
			theExternalLayerPairs.size() * sizeof(int), cudaMemcpyHostToDevice);

	GPUSimpleVector<80, GPUCACell*> ** isOuterHitOfCell;
	GPUSimpleVector<80, GPUCACell*> ** host_isOuterHitOfCell;

	cudaMallocHost(&host_isOuterHitOfCell,
			(theNumberOfLayers) * sizeof(GPUSimpleVector<80, GPUCACell*> *));
	cudaMalloc(&isOuterHitOfCell,
			(theNumberOfLayers) * sizeof(GPUSimpleVector<80, GPUCACell*> *));

	for (unsigned int i = 0; i < theNumberOfLayers; ++i)
	{

		cudaMalloc(&host_isOuterHitOfCell[i],
				hitsOnLayer[i]->size()
						* sizeof(GPUSimpleVector<80, GPUCACell*> ));
		cudaMemset(host_isOuterHitOfCell[i], 0,
				hitsOnLayer[i]->size()
						* sizeof(GPUSimpleVector<80, GPUCACell*> ));

	}
	cudaMemcpyAsync(isOuterHitOfCell, host_isOuterHitOfCell,
			(theNumberOfLayers) * sizeof(GPUSimpleVector<80, GPUCACell*> *),
			cudaMemcpyHostToDevice, 0);

	GPUCACell **theCells;
	GPUCACell **host_Cells;

	cudaMallocHost(&host_Cells, (theNumberOfLayerPairs) * sizeof(GPUCACell *));
	cudaMalloc(&theCells, (theNumberOfLayerPairs) * sizeof(GPUCACell *));

	for (unsigned int i = 0; i < theNumberOfLayerPairs; ++i)
	{
		cudaMalloc(&host_Cells[i],
				host_hitDoublets[i].size() * sizeof(GPUCACell));

	}

	cudaMemcpyAsync(theCells, host_Cells,
			(theNumberOfLayerPairs) * sizeof(GPUCACell *),
			cudaMemcpyHostToDevice, 0);

	GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> * foundNtuplets;
	GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> * host_foundNtuplets;

	cudaMallocHost(&host_foundNtuplets,
			sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ));

	cudaMalloc(&foundNtuplets,
			sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ));
	cudaMemset(foundNtuplets, 0, sizeof(int));

	dim3 numberOfBlocks(64, theNumberOfLayerPairs);
	cudaThreadSynchronize();

	kernel_create<<<numberOfBlocks,256>>>(gpu_doublets, gpu_layerHits, theCells, isOuterHitOfCell, theNumberOfLayerPairs);

	kernel_connect<<<numberOfBlocks,256>>>(gpu_doublets, theCells, isOuterHitOfCell, thePtMin, theRegionOriginX, theRegionOriginY, theRegionOriginRadius, theThetaCut, thePhiCut, theHardPtCut, theNumberOfLayerPairs);

	numberOfBlocks.y = theExternalLayerPairs.size();
	kernel_find_ntuplets<<<numberOfBlocks,128>>>(gpu_doublets, theCells, foundNtuplets,gpu_externalLayerPairs, theExternalLayerPairs.size(), 4 );
	cudaStreamSynchronize(0);

//
	// check for errors
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
	}
	cudaMemcpyAsync(host_foundNtuplets, foundNtuplets,
			sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> ),
			cudaMemcpyDeviceToHost, 0);
	cudaStreamSynchronize(0);

	quadruplets.resize(host_foundNtuplets->size());

	memcpy(quadruplets.data(), host_foundNtuplets->m_data,
			host_foundNtuplets->size() * sizeof(Quadruplet));

	cudaFree(gpu_externalLayerPairs);
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

	cudaFree(gpu_doublets);

	cudaFree(gpu_layerHits);

}

template class GPUCellularAutomaton<2000> ;
