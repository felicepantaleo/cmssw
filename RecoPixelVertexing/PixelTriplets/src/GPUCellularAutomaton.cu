#include <vector>
#include <array>

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUCACell.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUArena.h"
#include "RecoPixelVertexing/PixelTriplets/interface/GPUCellularAutomaton.h"

template<int numberOfLayers>
__global__
void kernel_create(const GPULayerDoublets* const* gpuDoublets,
		   GPUCACell<numberOfLayers>** cells,
                   GPUArena<numberOfLayers-1, 4, GPUCACell<numberOfLayers>> isOuterHitOfCell)
{
	unsigned int layerPairIndex = blockIdx.y;
	unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
	if(layerPairIndex < numberOfLayers-1)
	{
		for(int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex]->size; i+=gridDim.x * blockDim.x)
		{
			cells[layerPairIndex][i].init(gpuDoublets[layerPairIndex],layerPairIndex,i,gpuDoublets[layerPairIndex]->indices[2*i], gpuDoublets[layerPairIndex]->indices[2*i+1]);
			isOuterHitOfCell.push_back(layerPairIndex,cells[layerPairIndex][i].get_outer_hit_id(), & cells[layerPairIndex][i]);
		}
	}

}


template<int numberOfLayers>
__global__
void kernel_connect(const GPULayerDoublets* const* gpuDoublets,
                    GPUCACell<numberOfLayers>** cells,
		    GPUArena<numberOfLayers-1, 4, GPUCACell<numberOfLayers>> isOuterHitOfCell,
		    GPUArena<numberOfLayers-2, 4, GPUCACell<numberOfLayers>> innerNeighbors,
                    float ptmin,
                    float region_origin_x,
		    float region_origin_y,
                    float region_origin_radius,
                    float thetaCut,
		    float phiCut)
{
  unsigned int layerPairIndex = blockIdx.y;
  unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
  if(layerPairIndex < numberOfLayers-1)
  {
    for (int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex]->size; i += gridDim.x * blockDim.x)
    {
      GPUArenaIterator<4, GPUCACell<numberOfLayers>> innerNeighborsIterator = innerNeighbors.iterator(layerPairIndex,i);
      GPUCACell<numberOfLayers>* otherCell;
      while (innerNeighborsIterator.has_next())
      {
        otherCell = innerNeighborsIterator.get_next();
        if (cells[layerPairIndex][i].check_alignment_and_tag(otherCell,
              ptmin, region_origin_x, region_origin_y,
              region_origin_radius, thetaCut, phiCut))
          innerNeighbors.push_back(layerPairIndex,i,otherCell);
      }
    }
  }
}

template<int numberOfLayers, int maxNumberOfQuadruplets>
__global__
void kernel_find_ntuplets(const GPULayerDoublets* const* gpuDoublets,
                          GPUCACell<numberOfLayers>** cells,
                          GPUSimpleVector<maxNumberOfQuadruplets, GPUSimpleVector<4, int>>* foundNtuplets,
                          GPUArena<numberOfLayers-2, 4, GPUCACell<numberOfLayers>> theInnerNeighbors,
                          unsigned int minHitsPerNtuplet)
{
  unsigned int cellIndexInLastLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
  constexpr unsigned int lastLayerPairIndex = numberOfLayers - 2;

  GPUSimpleVector<4, GPUCACell<4>*> stack;

  for (int i = cellIndexInLastLayerPair; i < gpuDoublets[lastLayerPairIndex]->size;
      i += gridDim.x * blockDim.x)
  {
    stack.reset();
    cells[lastLayerPairIndex][i].find_ntuplets(foundNtuplets, theInnerNeighbors, stack, minHitsPerNtuplet);
  }

}


template<unsigned int theNumberOfLayers, unsigned int maxNumberOfQuadruplets>
void
GPUCellularAutomaton<theNumberOfLayers, maxNumberOfQuadruplets>::run(
  std::array<const GPULayerDoublets *, theNumberOfLayers-1> const & doublets, 
  std::vector<std::array<int, 4>> & quadruplets
) {
  int numberOfChunksIn1stArena = 0;
  std::array<int, theNumberOfLayers-1> numberOfKeysIn1stArena;
  std::cout << "numberOfKeysIn1stArena size " <<  numberOfKeysIn1stArena.size() << std::endl;
  for (size_t i = 0; i < theNumberOfLayers-1; ++i) {
    numberOfKeysIn1stArena[i] = doublets[i]->layers[1].size;
    numberOfChunksIn1stArena += doublets[i]->size;
    std::cout << "numberOfKeysIn1stArena[" << i << "]: " << numberOfKeysIn1stArena[i] << std::endl;
  }
  GPUArena<theNumberOfLayers-1, 4, GPUCACell<theNumberOfLayers>> isOuterHitOfCell(numberOfChunksIn1stArena, numberOfKeysIn1stArena);

  int numberOfChunksIn2ndArena = 0;
  std::array<int, theNumberOfLayers-2> numberOfKeysIn2ndArena;
  for (size_t i = 1; i < theNumberOfLayers-1; ++i) {
    numberOfKeysIn2ndArena[i-1] = doublets[i]->size;
    numberOfChunksIn2ndArena += doublets[i-1]->size;
  }
  GPUArena<theNumberOfLayers-2, 4, GPUCACell<theNumberOfLayers>>
                          theInnerNeighbors(numberOfChunksIn2ndArena, numberOfKeysIn2ndArena);

  GPUCACell<theNumberOfLayers>* theCells[theNumberOfLayers-1];
  for (unsigned int i = 0; i< theNumberOfLayers-1; ++i)
    cudaMalloc(& theCells[i], doublets[i]->size * sizeof(GPUCACell<theNumberOfLayers>));

  GPUSimpleVector<maxNumberOfQuadruplets, GPUSimpleVector<4, int>>* foundNtuplets;
  cudaMalloc(& foundNtuplets, sizeof(GPUSimpleVector<maxNumberOfQuadruplets, GPUSimpleVector<4, int>>));
  cudaMemset(foundNtuplets, 0x00, sizeof(GPUSimpleVector<maxNumberOfQuadruplets, GPUSimpleVector<4, int>>));

  kernel_create<<<1000,256>>>(doublets.data(), theCells, isOuterHitOfCell);

  kernel_connect<<<1000,256>>>(doublets.data(), theCells, isOuterHitOfCell, theInnerNeighbors, thePtMin, theRegionOriginX, theRegionOriginY, theRegionOriginRadius, theThetaCut, thePhiCut);

  kernel_find_ntuplets<<<1000,256>>>(doublets.data(), theCells, foundNtuplets, theInnerNeighbors, 4);

  auto h_foundNtuplets = new GPUSimpleVector<maxNumberOfQuadruplets, GPUSimpleVector<4, GPUCACell<4>>>();
  cudaMemcpy(h_foundNtuplets, foundNtuplets, sizeof(GPUSimpleVector<maxNumberOfQuadruplets, GPUSimpleVector<4, GPUCACell<4>>>), cudaMemcpyDeviceToHost);

  quadruplets.resize(h_foundNtuplets->size());
  memcpy(quadruplets.data(), h_foundNtuplets->m_data, h_foundNtuplets->size() * sizeof(std::array<int, 4>));

}

template class GPUCellularAutomaton<4, 1000>;
