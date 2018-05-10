//
// Author: Felice Pantaleo, CERN
//

#include "CAHitQuadrupletGeneratorGPU.h"
#include "GPUCACell.h"


void CAHitQuadrupletGeneratorGPU::deallocateOnGPU()
{
  cudaStreamDestroy(cudaStream_);

  cudaFreeHost(h_indices);
  cudaFreeHost(h_doublets);
  cudaFreeHost(h_x);
  cudaFreeHost(h_y);
  cudaFreeHost(h_z);
  cudaFreeHost(h_rootLayerPairs);
  cudaFreeHost(h_foundNtuplets);
  cudaFreeHost(tmp_layers);
  cudaFreeHost(tmp_layerDoublets);
  cudaFreeHost(h_layers);

  cudaFree(d_indices);
  cudaFree(d_doublets);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_rootLayerPairs);
  cudaFree(device_theCells);
  cudaFree(device_isOuterHitOfCell);
  cudaFree(d_foundNtuplets);
}

void CAHitQuadrupletGeneratorGPU::allocateOnGPU()
{
  cudaStreamCreateWithFlags(&cudaStream_, cudaStreamNonBlocking);
  unsigned int maxNumberOfLayerPairs = 13;
  unsigned int maxNumberOfLayers = 10;
  unsigned int maxNumberOfHits = 2000;
  unsigned int maxNumberOfRootLayerPairs = 13;

  cudaMallocHost(&h_doublets, maxNumberOfLayerPairs * sizeof(GPULayerDoublets));

  unsigned int maxNumberOfDoublets = 1;

  cudaMallocHost(&h_indices, maxNumberOfLayerPairs * maxNumberOfDoublets * 2 * sizeof(int));
  cudaMallocHost(&h_x, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMallocHost(&h_y, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMallocHost(&h_z, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMallocHost(&h_rootLayerPairs, maxNumberOfRootLayerPairs * sizeof(int));
  cudaMallocHost(&h_foundNtuplets,
                 sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>));

  cudaMalloc(&d_indices,
             maxNumberOfLayerPairs * maxNumberOfDoublets * 2 * sizeof(int));
  cudaMalloc(&d_doublets, maxNumberOfLayerPairs * sizeof(GPULayerDoublets));
  cudaMalloc(&d_layers, maxNumberOfLayers * sizeof(GPULayerHits));
  cudaMalloc(&d_x, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMalloc(&d_y, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMalloc(&d_z, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMalloc(&d_rootLayerPairs,
             maxNumberOfRootLayerPairs * sizeof(unsigned int));
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  cudaMalloc(&device_theCells,
             maxNumberOfLayerPairs * maxNumberOfDoublets * sizeof(GPUCACell));

  cudaMalloc(&device_isOuterHitOfCell,
             maxNumberOfLayers * maxNumberOfHits *
                 sizeof(GPUSimpleVector<maxCellsPerHit, unsigned int>));
  cudaMemset(device_isOuterHitOfCell, 0,
             maxNumberOfLayers * maxNumberOfHits *
                 sizeof(GPUSimpleVector<maxCellsPerHit, unsigned int>));

  cudaMalloc(&d_foundNtuplets,
             sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>));

  cudaMallocHost(&tmp_layers,
                   maxNumberOfLayers * sizeof(GPULayerHits));
  cudaMallocHost(&tmp_layerDoublets,
                   maxNumberOfLayerPairs * sizeof(GPULayerDoublets));
  cudaMallocHost(&h_layers, maxNumberOfLayers * sizeof(GPULayerHits));

}

void CAHitQuadrupletGeneratorGPU::launchKernels()
{



}
