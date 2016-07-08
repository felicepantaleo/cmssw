#include <vector>
#include <array>

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUCACell.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUArena.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUCellularAutomaton.h"




template<int numberOfLayers>
__global__
void kernel_create(const GPULayerDoublets* gpuDoublets,
		GPUCACell<numberOfLayers>** cells,
		//GPUArena<numberOfLayers-1, 16, GPUCACell<numberOfLayers>> isOuterHitOfCell)
                GPUSimpleVector<16, GPUCACell<numberOfLayers>* > ** isOuterHitOfCell)
{

	unsigned int layerPairIndex = blockIdx.y;
	unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;

	if(layerPairIndex < numberOfLayers-1)
	{

		for(int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size; i+=gridDim.x * blockDim.x)
		{
			cells[layerPairIndex][i].init(&gpuDoublets[layerPairIndex],layerPairIndex,i,gpuDoublets[layerPairIndex].indices[2*i], gpuDoublets[layerPairIndex].indices[2*i+1]);

			if(layerPairIndex < 2)
				//isOuterHitOfCell.push_back(layerPairIndex,cells[layerPairIndex][i].get_outer_hit_id(), & cells[layerPairIndex][i]);
				isOuterHitOfCell[layerPairIndex][cells[layerPairIndex][i].get_outer_hit_id()].push_back_ts(& cells[layerPairIndex][i]);
		}
	}

}

template<int numberOfLayers>
__global__
void kernel_connect(const GPULayerDoublets* gpuDoublets,
		GPUCACell<numberOfLayers>** cells,
		//GPUArena<numberOfLayers-1, 16, GPUCACell<numberOfLayers>> isOuterHitOfCell,
		//GPUArena<numberOfLayers-2, 16, GPUCACell<numberOfLayers>> innerNeighbors,
                GPUSimpleVector<16, GPUCACell<numberOfLayers>* > ** isOuterHitOfCell,
                GPUSimpleVector<16, GPUCACell<numberOfLayers>* > ** innerNeighbors,
		float ptmin,
		float region_origin_x,
		float region_origin_y,
		float region_origin_radius,
		float thetaCut,
		float phiCut)
{
	unsigned int layerPairIndex = blockIdx.y+1;
	unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
	if(layerPairIndex < numberOfLayers-1)
	{
		for (int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size; i += gridDim.x * blockDim.x)
		{
			//GPUArenaIterator<16, GPUCACell<numberOfLayers>> innerNeighborsIterator = innerNeighbors.iterator(layerPairIndex,i);
			GPUCACell<numberOfLayers>* otherCell;
			//while (innerNeighborsIterator.has_next())
                        for (int j = 0; j < isOuterHitOfCell[layerPairIndex-1][cells[layerPairIndex][i].get_inner_hit_id()].size(); ++j)
			{
				//otherCell = innerNeighborsIterator.get_next();
                                otherCell = isOuterHitOfCell[layerPairIndex-1][cells[layerPairIndex][i].get_inner_hit_id()].m_data[j];
				if (cells[layerPairIndex][i].check_alignment_and_tag(otherCell,
								ptmin, region_origin_x, region_origin_y,
								region_origin_radius, thetaCut, phiCut))
				//innerNeighbors.push_back(layerPairIndex,i,otherCell);
                                innerNeighbors[layerPairIndex][i].push_back(otherCell);
			}
		}
	}
}

template<int numberOfLayers, int maxNumberOfQuadruplets>
__global__
void kernel_find_ntuplets(const GPULayerDoublets* gpuDoublets,
		GPUCACell<numberOfLayers>** cells,
		GPUSimpleVector<maxNumberOfQuadruplets, GPUSimpleVector<4, int>>* foundNtuplets,
		//GPUArena<numberOfLayers-2, 16, GPUCACell<numberOfLayers>> theInnerNeighbors,
                GPUSimpleVector<16, GPUCACell<numberOfLayers>* > ** theInnerNeighbors,
		unsigned int minHitsPerNtuplet)
{

	unsigned int cellIndexInLastLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
	constexpr unsigned int lastLayerPairIndex = numberOfLayers - 2;

	GPUSimpleVector<4, GPUCACell<4>*> stack;

	for (int i = cellIndexInLastLayerPair; i < gpuDoublets[lastLayerPairIndex].size;
			i += gridDim.x * blockDim.x)
	{
		stack.reset();
		printf("foundquadruplets: %d\n", foundNtuplets->size());

		cells[lastLayerPairIndex][i].find_ntuplets(foundNtuplets, theInnerNeighbors, stack, minHitsPerNtuplet);
	}


}

template<unsigned int theNumberOfLayers, unsigned int maxNumberOfQuadruplets>
void GPUCellularAutomaton<theNumberOfLayers, maxNumberOfQuadruplets>::run(
		std::array<const GPULayerDoublets *, theNumberOfLayers - 1> const & doublets,
		std::vector<std::array<int, 4>> & quadruplets)
{
        GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* > * hostIsOuterHitOfCell[theNumberOfLayers-2];
        GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* > * hostTheInnerNeighbors[theNumberOfLayers-2];
        GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* > ** isOuterHitOfCell;
        GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* > ** theInnerNeighbors;

	int numberOfChunksIn1stArena = 0;
	std::array<int, theNumberOfLayers - 1> numberOfKeysIn1stArena;
	std::cout << "numberOfKeysIn1stArena size " << numberOfKeysIn1stArena.size() << std::endl;
	for (size_t i = 0; i < theNumberOfLayers - 1; ++i)
	{
		numberOfKeysIn1stArena[i] = doublets[i]->layers[1].size;
		numberOfChunksIn1stArena += doublets[i]->size;
		std::cout << "numberOfKeysIn1stArena[" << i << "]: " << numberOfKeysIn1stArena[i] << std::endl;
	}
	GPULayerDoublets* gpu_doublets;

	cudaMalloc(&gpu_doublets, 3 * sizeof(GPULayerDoublets));

	for (int i = 0; i < 3; ++i)
	{
		cudaMemcpy(&gpu_doublets[i], doublets[i], sizeof(GPULayerDoublets),
				cudaMemcpyHostToDevice);

	}

        /*
	GPUArena<theNumberOfLayers - 1, 16, GPUCACell<theNumberOfLayers>> isOuterHitOfCell(
			numberOfChunksIn1stArena, numberOfKeysIn1stArena);
        */
        cudaMalloc(& hostIsOuterHitOfCell[0], numberOfKeysIn1stArena[0] * sizeof(GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* >) );
        cudaMemset(hostIsOuterHitOfCell[0], 0, numberOfKeysIn1stArena[0] * sizeof(GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* >) );
        cudaMalloc(& hostIsOuterHitOfCell[1], numberOfKeysIn1stArena[1] * sizeof(GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* >) );
        cudaMemset(hostIsOuterHitOfCell[1], 0, numberOfKeysIn1stArena[1] * sizeof(GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* >) );
        cudaMalloc(& isOuterHitOfCell, 2 * sizeof(void*));
        cudaMemcpy(isOuterHitOfCell, hostIsOuterHitOfCell, 2 * sizeof(void*), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	int numberOfChunksIn2ndArena = 0;
	std::array<int, theNumberOfLayers - 2> numberOfKeysIn2ndArena;
	for (size_t i = 1; i < theNumberOfLayers - 1; ++i)
	{
		numberOfKeysIn2ndArena[i - 1] = doublets[i]->size;
		numberOfChunksIn2ndArena += doublets[i - 1]->size;
	}
        /*
	GPUArena<theNumberOfLayers - 2, 16, GPUCACell<theNumberOfLayers>> theInnerNeighbors(
			numberOfChunksIn2ndArena, numberOfKeysIn2ndArena);
        */
        cudaMalloc(& hostTheInnerNeighbors[0], numberOfKeysIn2ndArena[0] * sizeof(GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* >) );
        cudaMemset(hostTheInnerNeighbors[0], 0, numberOfKeysIn2ndArena[0] * sizeof(GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* >) );
        cudaMalloc(& hostTheInnerNeighbors[1], numberOfKeysIn2ndArena[1] * sizeof(GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* >) );
        cudaMemset(hostTheInnerNeighbors[1], 0, numberOfKeysIn2ndArena[1] * sizeof(GPUSimpleVector<16, GPUCACell<theNumberOfLayers>* >) );
        cudaMalloc(& theInnerNeighbors, 2 * sizeof(void*));
        cudaMemcpy(theInnerNeighbors, hostTheInnerNeighbors, 2 * sizeof(void*), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	GPUCACell < theNumberOfLayers > **theCells;
	cudaMalloc(&theCells,
			(theNumberOfLayers - 1) * sizeof(GPUCACell<theNumberOfLayers> *));

	GPUCACell < theNumberOfLayers > *hostCells[theNumberOfLayers - 1];

	for (unsigned int i = 0; i < theNumberOfLayers - 1; ++i)
		checkCudaError(
				cudaMalloc(&hostCells[i],
						doublets[i]->size
								* sizeof(GPUCACell<theNumberOfLayers> )));

	cudaMemcpy(theCells, hostCells,
			(theNumberOfLayers - 1) * sizeof(GPUCACell<theNumberOfLayers> *),
			cudaMemcpyHostToDevice);

	GPUSimpleVector<maxNumberOfQuadruplets, GPUSimpleVector<4, int>>* foundNtuplets;
	checkCudaError(
			cudaMalloc(&foundNtuplets,
					sizeof(GPUSimpleVector<maxNumberOfQuadruplets,
							GPUSimpleVector<4, int>>)));
	checkCudaError(cudaMemset(foundNtuplets, 0, sizeof(GPUSimpleVector<maxNumberOfQuadruplets, GPUSimpleVector<4, int>>)));
	checkLastCudaError();

	kernel_create<<<dim3(1,3),256>>>(gpu_doublets, theCells, isOuterHitOfCell);
	checkLastCudaError();

	kernel_connect<<<dim3(1,2),256>>>(gpu_doublets, theCells, isOuterHitOfCell, theInnerNeighbors, thePtMin, theRegionOriginX, theRegionOriginY, theRegionOriginRadius, theThetaCut, thePhiCut);

	kernel_find_ntuplets<<<1,256>>>(gpu_doublets, theCells, foundNtuplets, theInnerNeighbors, 4);

	auto h_foundNtuplets = new GPUSimpleVector<maxNumberOfQuadruplets, GPUSimpleVector<4, GPUCACell<4>>>();
	cudaMemcpy(h_foundNtuplets, foundNtuplets, sizeof(GPUSimpleVector<maxNumberOfQuadruplets, GPUSimpleVector<4, GPUCACell<4>>>), cudaMemcpyDeviceToHost);

	quadruplets.resize(h_foundNtuplets->size());
	memcpy(quadruplets.data(), h_foundNtuplets->m_data, h_foundNtuplets->size() * sizeof(std::array<int, 4>));

  cudaFree(foundNtuplets);
  for (unsigned int i = 0; i< theNumberOfLayers-1; ++i)
    cudaFree(hostCells[i]);
  cudaFree(theCells);
}

template class GPUCellularAutomaton<4, 1000> ;
