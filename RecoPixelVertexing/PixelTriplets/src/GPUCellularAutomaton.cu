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
                GPUSimpleVector<80, GPUCACell<numberOfLayers>* > ** isOuterHitOfCell)
{

	unsigned int layerPairIndex = blockIdx.y;
	unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
	if(layerPairIndex < numberOfLayers-1)
	{

		for(int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size; i+=gridDim.x * blockDim.x)
		{

			cells[layerPairIndex][i].init(&gpuDoublets[layerPairIndex],layerPairIndex,i,gpuDoublets[layerPairIndex].indices[2*i], gpuDoublets[layerPairIndex].indices[2*i+1]);

			if(layerPairIndex < 2){
				isOuterHitOfCell[layerPairIndex][cells[layerPairIndex][i].get_outer_hit_id()].push_back_ts(& (cells[layerPairIndex][i]));
//				if(				isOuterHitOfCell[layerPairIndex][cells[layerPairIndex][i].get_outer_hit_id()].size() >100)
//					printf("size is outer hit > 80!");
			}
		}
	}

}

template<int numberOfLayers>
__global__
void kernel_connect(const GPULayerDoublets* gpuDoublets,
		GPUCACell<numberOfLayers>** cells,
		//GPUArena<numberOfLayers-1, 80, GPUCACell<numberOfLayers>> isOuterHitOfCell,
		//GPUArena<numberOfLayers-2, 80, GPUCACell<numberOfLayers>> innerNeighbors,
                GPUSimpleVector<80, GPUCACell<numberOfLayers>* > ** isOuterHitOfCell,
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

           for (int j = 0; j < isOuterHitOfCell[layerPairIndex-1][cells[layerPairIndex][i].get_inner_hit_id()].size(); ++j)
			{
        	   GPUCACell<numberOfLayers>* otherCell= isOuterHitOfCell[layerPairIndex-1][cells[layerPairIndex][i].get_inner_hit_id()].m_data[j];
//

        	   if (cells[layerPairIndex][i].check_alignment_and_tag(otherCell,
								ptmin, region_origin_x, region_origin_y,
								region_origin_radius, thetaCut, phiCut))
        	   {
				//innerNeighbors.push_back(layerPairIndex,i,otherCell);
        		   	   cells[layerPairIndex][i].theInnerNeighbors.push_back_ts(otherCell);
//        		   	if(cells[layerPairIndex][i].theInnerNeighbors.size()>60)
//    					printf("size of neighbors %d > 40!", cells[layerPairIndex][i].theInnerNeighbors.size());

        	   }
			}
		}
	}
}

template<int numberOfLayers, int maxNumberOfQuadruplets>
__global__
void kernel_find_ntuplets(const GPULayerDoublets* gpuDoublets,
		GPUCACell<numberOfLayers>** cells,
		GPUSimpleVector<maxNumberOfQuadruplets, int4>* foundNtuplets,
		//GPUArena<numberOfLayers-2, 80, GPUCACell<numberOfLayers>> theInnerNeighbors,
		unsigned int minHitsPerNtuplet)
{

	unsigned int cellIndexInLastLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
	constexpr unsigned int lastLayerPairIndex = numberOfLayers - 2;


	GPUSimpleVector<4, GPUCACell<4>*> stack;
	for (int i = cellIndexInLastLayerPair; i < gpuDoublets[lastLayerPairIndex].size;
			i += gridDim.x * blockDim.x)
	{


		stack.reset();
		stack.push_back(&cells[lastLayerPairIndex][i]);
		cells[lastLayerPairIndex][i].find_ntuplets(foundNtuplets, stack, minHitsPerNtuplet);


	}


}

template<unsigned int theNumberOfLayers, unsigned int maxNumberOfQuadruplets>
void GPUCellularAutomaton<theNumberOfLayers, maxNumberOfQuadruplets>::run(
		std::array<const GPULayerDoublets *, theNumberOfLayers - 1> const & doublets,
		std::vector<std::array<int, 4>> & quadruplets)
{
//        GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* > * hostIsOuterHitOfCell[theNumberOfLayers-2];
//       GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* > ** isOuterHitOfCell;

	int numberOfChunksIn1stArena = 0;
	std::array<int, theNumberOfLayers - 1> numberOfKeysIn1stArena;
//	std::cout << "numberOfKeysIn1stArena size " << numberOfKeysIn1stArena.size() << std::endl;
	for (size_t i = 0; i < theNumberOfLayers - 1; ++i)
	{
		numberOfKeysIn1stArena[i] = doublets[i]->layers[1].size;
		numberOfChunksIn1stArena += doublets[i]->size;
//		std::cout << "numberOfKeysIn1stArena[" << i << "]: " << numberOfKeysIn1stArena[i] << std::endl;
	}
	GPULayerDoublets* gpu_doublets;

	cudaMalloc(&gpu_doublets, 3 * sizeof(GPULayerDoublets));

	for (int i = 0; i < 3; ++i)
	{
		cudaMemcpy(&gpu_doublets[i], doublets[i], sizeof(GPULayerDoublets),
				cudaMemcpyHostToDevice);

	}


    GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* > ** isOuterHitOfCell;
	cudaMallocManaged(&isOuterHitOfCell,
			(theNumberOfLayers - 2) * sizeof(GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* > *));

	for (unsigned int i = 0; i < theNumberOfLayers - 2; ++i)
			checkCudaError(cudaMallocManaged(&isOuterHitOfCell[i],
							numberOfKeysIn1stArena[i] * sizeof(GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* > )));


        /*
	GPUArena<theNumberOfLayers - 1, 80, GPUCACell<theNumberOfLayers>> isOuterHitOfCell(
			numberOfChunksIn1stArena, numberOfKeysIn1stArena);
        */
//        cudaMalloc(& hostIsOuterHitOfCell[0], numberOfKeysIn1stArena[0] * sizeof(GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* >) );
//        cudaMemset(hostIsOuterHitOfCell[0], 0, numberOfKeysIn1stArena[0] * sizeof(GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* >) );
//        cudaMalloc(& hostIsOuterHitOfCell[1], numberOfKeysIn1stArena[1] * sizeof(GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* >) );
//        cudaMemset(hostIsOuterHitOfCell[1], 0, numberOfKeysIn1stArena[1] * sizeof(GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* >) );
//        cudaMalloc(& isOuterHitOfCell, 2 * sizeof(void*));
//        cudaMemcpy(isOuterHitOfCell, hostIsOuterHitOfCell, 2 * sizeof(void*), cudaMemcpyHostToDevice);

	int numberOfChunksIn2ndArena = 0;
	std::array<int, theNumberOfLayers - 2> numberOfKeysIn2ndArena;
	for (size_t i = 1; i < theNumberOfLayers - 1; ++i)
	{
		numberOfKeysIn2ndArena[i - 1] = doublets[i]->size;
		numberOfChunksIn2ndArena += doublets[i - 1]->size;
	}
        /*
	GPUArena<theNumberOfLayers - 2, 80, GPUCACell<theNumberOfLayers>> theInnerNeighbors(
			numberOfChunksIn2ndArena, numberOfKeysIn2ndArena);
        */
//        cudaMalloc(& hostTheInnerNeighbors[0], numberOfKeysIn2ndArena[0] * sizeof(GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* >) );
//        cudaMemset(hostTheInnerNeighbors[0], 0, numberOfKeysIn2ndArena[0] * sizeof(GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* >) );
//        cudaMalloc(& hostTheInnerNeighbors[1], numberOfKeysIn2ndArena[1] * sizeof(GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* >) );
//        cudaMemset(hostTheInnerNeighbors[1], 0, numberOfKeysIn2ndArena[1] * sizeof(GPUSimpleVector<80, GPUCACell<theNumberOfLayers>* >) );
//        cudaMalloc(& theInnerNeighbors, 2 * sizeof(void*));
//        cudaMemcpy(theInnerNeighbors, hostTheInnerNeighbors, 2 * sizeof(void*), cudaMemcpyHostToDevice);
//	cudaDeviceSynchronize();

	GPUCACell < theNumberOfLayers > **theCells;
	cudaMallocManaged(&theCells,
			(theNumberOfLayers - 1) * sizeof(GPUCACell<theNumberOfLayers> *));

	for (unsigned int i = 0; i < theNumberOfLayers - 1; ++i)
			checkCudaError(
					cudaMallocManaged(&theCells[i],
							doublets[i]->size
									* sizeof(GPUCACell<theNumberOfLayers> )));


//	GPUCACell < theNumberOfLayers > *hostCells[theNumberOfLayers - 1];
//
//	for (unsigned int i = 0; i < theNumberOfLayers - 1; ++i)
//		checkCudaError(
//				cudaMalloc(&hostCells[i],
//						doublets[i]->size
//								* sizeof(GPUCACell<theNumberOfLayers> )));

//	cudaMemcpy(theCells, hostCells,
//			(theNumberOfLayers - 1) * sizeof(GPUCACell<theNumberOfLayers> *),
//			cudaMemcpyHostToDevice);

	GPUSimpleVector<maxNumberOfQuadruplets, int4>* foundNtuplets;
	checkCudaError(
			cudaMallocManaged(&foundNtuplets,
					sizeof(GPUSimpleVector<maxNumberOfQuadruplets,
							int4>)));
	checkCudaError(cudaMemset(foundNtuplets, 0, sizeof(int)));
	checkLastCudaError();

	kernel_create<<<dim3(100,3),512>>>(gpu_doublets, theCells, isOuterHitOfCell);

	kernel_connect<<<dim3(100,2),512>>>(gpu_doublets, theCells, isOuterHitOfCell, thePtMin, theRegionOriginX, theRegionOriginY, theRegionOriginRadius, theThetaCut, thePhiCut);


	kernel_find_ntuplets<<<64,128>>>(gpu_doublets, theCells, foundNtuplets, 4);
	cudaStreamSynchronize(0);

	quadruplets.resize(foundNtuplets->size());
	memcpy(quadruplets.data(), foundNtuplets->m_data, foundNtuplets->size() * sizeof(std::array<int, 4>));
//	std::cout << "found quadruplets: " << foundNtuplets->size() << std::endl;

  for (unsigned int i = 0; i< theNumberOfLayers-1; ++i)
    cudaFree(theCells[i]);

	for (unsigned int i = 0; i < theNumberOfLayers - 2; ++i)
				cudaFree(isOuterHitOfCell[i]);

	cudaFree(foundNtuplets);
	cudaFree(theCells);
	cudaFree(isOuterHitOfCell);
}

template class GPUCellularAutomaton<4, 2000> ;
