//
// Author: Felice Pantaleo, CERN
//

#include "CAHitQuadrupletGeneratorGPU.h"
#include "GPUCACell.h"

template <int maxNumberOfQuadruplets>
__global__ void
kernel_debug(unsigned int numberOfLayerPairs, unsigned int numberOfLayers,
             const GPULayerDoublets *gpuDoublets,
             const GPULayerHits *gpuHitsOnLayers, GPUCACell *cells,
             GPUSimpleVector<200, unsigned int> *isOuterHitOfCell,
             GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> *foundNtuplets,
             float ptmin, float region_origin_x, float region_origin_y,
             float region_origin_radius, const float thetaCut,
             const float phiCut, const float hardPtCut,
             unsigned int maxNumberOfDoublets, unsigned int maxNumberOfHits) {
  if (threadIdx.x == 0 and blockIdx.x == 0)
    foundNtuplets->reset();

  printf("kernel_debug_create: theEvent contains numberOfLayerPairs: %d\n",
         numberOfLayerPairs);
  for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs;
       ++layerPairIndex) {

    int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
    int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
    int numberOfDoublets = gpuDoublets[layerPairIndex].size;
    printf(
        "kernel_debug_create: layerPairIndex: %d inner %d outer %d size %u\n",
        layerPairIndex, innerLayerId, outerLayerId, numberOfDoublets);

    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets;
    auto globalFirstHitIdx = outerLayerId * maxNumberOfHits;
    printf("kernel_debug_create: theIdOfThefirstCellInLayerPair: %d "
           "globalFirstHitIdx %d\n",
           globalFirstDoubletIdx, globalFirstHitIdx);

    for (unsigned int i = 0; i < gpuDoublets[layerPairIndex].size; i++) {

      auto globalCellIdx = i + globalFirstDoubletIdx;
      auto &thisCell = cells[globalCellIdx];
      auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * i + 1];
      thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers,
                    layerPairIndex, globalCellIdx,
                    gpuDoublets[layerPairIndex].indices[2 * i], outerHitId,
                    region_origin_x, region_origin_y);

      isOuterHitOfCell[globalFirstHitIdx + outerHitId].push_back_ts(
          globalCellIdx);
    }
  }

  // for(unsigned int layerIndex = 0; layerIndex < numberOfLayers;++layerIndex )
  // {
  //     auto numberOfHitsOnLayer = gpuHitsOnLayers[layerIndex].size;
  //     for(unsigned hitId = 0; hitId < numberOfHitsOnLayer; hitId++)
  //     {
  //
  //         if(isOuterHitOfCell[layerIndex*maxNumberOfHits+hitId].size()>0)
  //         {
  //             printf("\nlayer %d hit %d is outer hit of %d
  //             cells\n",layerIndex, hitId,
  //             isOuterHitOfCell[layerIndex*maxNumberOfHits+hitId].size());
  //             printf("\n\t%f %f %f
  //             \n",gpuHitsOnLayers[layerIndex].x[hitId],gpuHitsOnLayers[layerIndex].y[hitId],gpuHitsOnLayers[layerIndex].z[hitId]);
  //
  //             for(unsigned cell = 0; cell<
  //             isOuterHitOfCell[layerIndex*maxNumberOfHits+hitId].size();
  //             cell++)
  //             {
  //                 printf("cell %d\n",
  //                 isOuterHitOfCell[layerIndex*maxNumberOfHits+hitId].m_data[cell]);
  //                 auto& thisCell =
  //                 cells[isOuterHitOfCell[layerIndex*maxNumberOfHits+hitId].m_data[cell]];
  //                             float x1, y1, z1, x2, y2, z2;
  //
  //                             x1 = thisCell.get_inner_x();
  //                             y1 = thisCell.get_inner_y();
  //                             z1 = thisCell.get_inner_z();
  //                             x2 = thisCell.get_outer_x();
  //                             y2 = thisCell.get_outer_y();
  //                             z2 = thisCell.get_outer_z();
  //                 printf("\n\tDEBUG cellid %d innerhit outerhit (xyz) (%f %f
  //                 %f), (%f %f
  //                 %f)\n",isOuterHitOfCell[layerIndex*maxNumberOfHits+hitId].m_data[cell],
  //                 x1,y1,z1,x2,y2,z2);
  //             }
  //         }
  //     }
  // }

  // starting connect

  for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs;
       ++layerPairIndex) {

    int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
    int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
    int numberOfDoublets = gpuDoublets[layerPairIndex].size;
    printf("kernel_debug_connect: connecting layerPairIndex: %d inner %d outer "
           "%d size %u\n",
           layerPairIndex, innerLayerId, outerLayerId, numberOfDoublets);

    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets;
    auto globalFirstHitIdx = innerLayerId * maxNumberOfHits;
    //        printf("kernel_debug_connect: theIdOfThefirstCellInLayerPair: %d
    //        globalFirstHitIdx %d\n", globalFirstDoubletIdx,
    //        globalFirstHitIdx);

    for (unsigned int i = 0; i < numberOfDoublets; i++) {

      auto globalCellIdx = i + globalFirstDoubletIdx;

      auto &thisCell = cells[globalCellIdx];
      auto innerHitId = thisCell.get_inner_hit_id();
      auto numberOfPossibleNeighbors =
          isOuterHitOfCell[globalFirstHitIdx + innerHitId].size();
      //            if(numberOfPossibleNeighbors>0)
      //            printf("kernel_debug_connect: cell: %d has %d possible
      //            neighbors\n", globalCellIdx, numberOfPossibleNeighbors);
      float x1, y1, z1, x2, y2, z2;

      x1 = thisCell.get_inner_x();
      y1 = thisCell.get_inner_y();
      z1 = thisCell.get_inner_z();
      x2 = thisCell.get_outer_x();
      y2 = thisCell.get_outer_y();
      z2 = thisCell.get_outer_z();
      printf("\n\n\nDEBUG cellid %d innerhit outerhit (xyz) (%f %f %f), (%f %f "
             "%f)\n",
             globalCellIdx, x1, y1, z1, x2, y2, z2);

      for (auto j = 0; j < numberOfPossibleNeighbors; ++j) {
        unsigned int otherCell =
            isOuterHitOfCell[globalFirstHitIdx + innerHitId].m_data[j];

        float x3, y3, z3, x4, y4, z4;
        x3 = cells[otherCell].get_inner_x();
        y3 = cells[otherCell].get_inner_y();
        z3 = cells[otherCell].get_inner_z();
        x4 = cells[otherCell].get_outer_x();
        y4 = cells[otherCell].get_outer_y();
        z4 = cells[otherCell].get_outer_z();

        printf("kernel_debug_connect: checking compatibility with %d \n",
               otherCell);
        printf("DEBUG \tinnerhit outerhit (xyz) (%f %f %f), (%f %f %f)\n", x3,
               y3, z3, x4, y4, z4);

        if (thisCell.check_alignment_and_tag(
                cells, otherCell, ptmin, region_origin_x, region_origin_y,
                region_origin_radius, thetaCut, phiCut, hardPtCut)) {

          printf("kernel_debug_connect: \t\tcell %d is outer neighbor of %d \n",
                 globalCellIdx, otherCell);

          cells[otherCell].theOuterNeighbors.push_back_ts(globalCellIdx);
        }
      }
    }
  }
}

__global__ void debug_input_data(unsigned int numberOfLayerPairs,
                                 const GPULayerDoublets *gpuDoublets,
                                 const GPULayerHits *gpuHitsOnLayers,
                                 float ptmin, float region_origin_x,
                                 float region_origin_y,
                                 float region_origin_radius,
                                 unsigned int maxNumberOfHits) {
  printf("GPU: Region ptmin %f , region_origin_x %f , region_origin_y %f , "
         "region_origin_radius  %f \n",
         ptmin, region_origin_x, region_origin_y, region_origin_radius);
  printf("GPU: numberOfLayerPairs: %d\n", numberOfLayerPairs);

  for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs;
       ++layerPairIndex) {
    printf("\t numberOfDoublets: %d \n", gpuDoublets[layerPairIndex].size);
    printf("\t innerLayer: %d outerLayer: %d \n",
           gpuDoublets[layerPairIndex].innerLayerId,
           gpuDoublets[layerPairIndex].outerLayerId);

    for (unsigned int cellIndexInLayerPair = 0;
         cellIndexInLayerPair < gpuDoublets[layerPairIndex].size;
         ++cellIndexInLayerPair) {

      if (cellIndexInLayerPair < 5) {
        auto innerhit =
            gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair];
        auto innerX = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId]
                          .x[innerhit];
        auto innerY = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId]
                          .y[innerhit];
        auto innerZ = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId]
                          .z[innerhit];

        auto outerhit =
            gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair + 1];
        auto outerX = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId]
                          .x[outerhit];
        auto outerY = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId]
                          .y[outerhit];
        auto outerZ = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId]
                          .z[outerhit];
        printf("\t \t %d innerHit: %d %f %f %f outerHit: %d %f %f %f\n",
               cellIndexInLayerPair, innerhit, innerX, innerY, innerZ, outerhit,
               outerX, outerY, outerZ);
      }
    }
  }
}

template <int maxNumberOfQuadruplets>
__global__ void kernel_debug_find_ntuplets(
    unsigned int numberOfRootLayerPairs, const GPULayerDoublets *gpuDoublets,
    GPUCACell *cells,
    GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> *foundNtuplets,
    unsigned int *rootLayerPairs, unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets) {
  printf("numberOfRootLayerPairs = %d", numberOfRootLayerPairs);
  for (int rootLayerPair = 0; rootLayerPair < numberOfRootLayerPairs;
       ++rootLayerPair) {
    unsigned int rootLayerPairIndex = rootLayerPairs[rootLayerPair];
    auto globalFirstDoubletIdx = rootLayerPairIndex * maxNumberOfDoublets;

    GPUSimpleVector<3, unsigned int> stack;
    for (int i = 0; i < gpuDoublets[rootLayerPairIndex].size; i++) {
      auto globalCellIdx = i + globalFirstDoubletIdx;
      stack.reset();
      stack.push_back(globalCellIdx);
      cells[globalCellIdx].find_ntuplets(cells, foundNtuplets, stack,
                                         minHitsPerNtuplet);
    }
    printf("found quadruplets: %d", foundNtuplets->size());
  }
}

template <int maxNumberOfQuadruplets>
__global__ void kernel_create(
    const unsigned int numberOfLayerPairs, const GPULayerDoublets *gpuDoublets,
    const GPULayerHits *gpuHitsOnLayers, GPUCACell *cells,
    GPUSimpleVector<200, unsigned int> *isOuterHitOfCell,
    GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> *foundNtuplets,
    const float region_origin_x, const float region_origin_y,
    unsigned int maxNumberOfDoublets, unsigned int maxNumberOfHits) {

  unsigned int layerPairIndex = blockIdx.y;
  unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndexInLayerPair == 0 && layerPairIndex == 0) {
    foundNtuplets->reset();
  }

  if (layerPairIndex < numberOfLayerPairs) {
    int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets;
    auto globalFirstHitIdx = outerLayerId * maxNumberOfHits;

    for (unsigned int i = cellIndexInLayerPair;
         i < gpuDoublets[layerPairIndex].size; i += gridDim.x * blockDim.x) {
      auto globalCellIdx = i + globalFirstDoubletIdx;
      auto &thisCell = cells[globalCellIdx];
      auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * i + 1];
      thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers,
                    layerPairIndex, globalCellIdx,
                    gpuDoublets[layerPairIndex].indices[2 * i], outerHitId,
                    region_origin_x, region_origin_y);

      isOuterHitOfCell[globalFirstHitIdx + outerHitId].push_back_ts(
          globalCellIdx);
    }
  }
}

__global__ void
kernel_connect(unsigned int numberOfLayerPairs,
               const GPULayerDoublets *gpuDoublets, GPUCACell *cells,
               GPUSimpleVector<200, unsigned int> *isOuterHitOfCell,
               float ptmin, float region_origin_x, float region_origin_y,
               float region_origin_radius, const float thetaCut,
               const float phiCut, const float hardPtCut,
               unsigned int maxNumberOfDoublets, unsigned int maxNumberOfHits) {
  unsigned int layerPairIndex = blockIdx.y;
  unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
  if (layerPairIndex < numberOfLayerPairs) {
    int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets;
    auto globalFirstHitIdx = innerLayerId * maxNumberOfHits;

    for (int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size;
         i += gridDim.x * blockDim.x) {
      auto globalCellIdx = i + globalFirstDoubletIdx;

      auto &thisCell = cells[globalCellIdx];
      auto innerHitId = thisCell.get_inner_hit_id();
      auto numberOfPossibleNeighbors =
          isOuterHitOfCell[globalFirstHitIdx + innerHitId].size();
      for (auto j = 0; j < numberOfPossibleNeighbors; ++j) {
        unsigned int otherCell =
            isOuterHitOfCell[globalFirstHitIdx + innerHitId].m_data[j];

        if (thisCell.check_alignment_and_tag(
                cells, otherCell, ptmin, region_origin_x, region_origin_y,
                region_origin_radius, thetaCut, phiCut, hardPtCut)) {
          //                    printf("kernel_debug_connect: \t\tcell %d is
          //                    outer neighbor of %d \n", globalCellIdx,
          //                    otherCell);

          cells[otherCell].theOuterNeighbors.push_back_ts(globalCellIdx);
        }
      }
    }
  }
}

template <int maxNumberOfQuadruplets>
__global__ void kernel_find_ntuplets(
    unsigned int numberOfRootLayerPairs, const GPULayerDoublets *gpuDoublets,
    GPUCACell *cells,
    GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet> *foundNtuplets,
    unsigned int *rootLayerPairs, unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets) {

  if (blockIdx.y < numberOfRootLayerPairs) {
    unsigned int cellIndexInRootLayerPair =
        threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int rootLayerPairIndex = rootLayerPairs[blockIdx.y];
    auto globalFirstDoubletIdx = rootLayerPairIndex * maxNumberOfDoublets;
    GPUSimpleVector<3, unsigned int> stack;
    for (int i = cellIndexInRootLayerPair;
         i < gpuDoublets[rootLayerPairIndex].size;
         i += gridDim.x * blockDim.x) {
      auto globalCellIdx = i + globalFirstDoubletIdx;
      stack.reset();
      stack.push_back(globalCellIdx);
      cells[globalCellIdx].find_ntuplets(cells, foundNtuplets, stack,
                                         minHitsPerNtuplet);
    }
  }
}

void CAHitQuadrupletGeneratorGPU::deallocateOnGPU() {
  cudaStreamDestroy(cudaStream_);

  cudaFreeHost(h_indices);
  cudaFreeHost(h_doublets);
  cudaFreeHost(h_x);
  cudaFreeHost(h_y);
  cudaFreeHost(h_z);
  cudaFreeHost(h_rootLayerPairs);
  for (int i = 0; i < maxNumberOfRegions; ++i)
    cudaFreeHost(h_foundNtuplets[i]);
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
  for (int i = 0; i < maxNumberOfRegions; ++i)
    cudaFree(d_foundNtuplets[i]);
}

void CAHitQuadrupletGeneratorGPU::allocateOnGPU() {
  cudaStreamCreateWithFlags(&cudaStream_, cudaStreamNonBlocking);

  cudaMallocHost(&h_doublets, maxNumberOfLayerPairs * sizeof(GPULayerDoublets));

  cudaMallocHost(&h_indices,
                 maxNumberOfLayerPairs * maxNumberOfDoublets * 2 * sizeof(int));
  cudaMallocHost(&h_x, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMallocHost(&h_y, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMallocHost(&h_z, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMallocHost(&h_rootLayerPairs, maxNumberOfRootLayerPairs * sizeof(int));

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
  h_foundNtuplets.resize(maxNumberOfRegions);
  d_foundNtuplets.resize(maxNumberOfRegions);
  for (int i = 0; i < maxNumberOfRegions; ++i) {
    cudaMalloc(&d_foundNtuplets[i],
               sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>));
    cudaMallocHost(&h_foundNtuplets[i],
                   sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>));
  }

  cudaMallocHost(&tmp_layers, maxNumberOfLayers * sizeof(GPULayerHits));
  cudaMallocHost(&tmp_layerDoublets,
                 maxNumberOfLayerPairs * sizeof(GPULayerDoublets));
  cudaMallocHost(&h_layers, maxNumberOfLayers * sizeof(GPULayerHits));
}

void CAHitQuadrupletGeneratorGPU::launchKernels(const TrackingRegion &region,
                                                int regionIndex) {

  assert(regionIndex < maxNumberOfRegions);
  dim3 numberOfBlocks_create(32, numberOfLayerPairs);
  dim3 numberOfBlocks_connect(16, numberOfLayerPairs);
  dim3 numberOfBlocks_find(8, numberOfRootLayerPairs);
  h_foundNtuplets[regionIndex]->reset();
  kernel_create<<<numberOfBlocks_create, 32, 0, cudaStream_>>>(
      numberOfLayerPairs, d_doublets, d_layers, device_theCells,
      device_isOuterHitOfCell, d_foundNtuplets[regionIndex],
      region.origin().x(), region.origin().y(), maxNumberOfDoublets,
      maxNumberOfHits);

  kernel_connect<<<numberOfBlocks_connect, 512, 0, cudaStream_>>>(
      numberOfLayerPairs, d_doublets, device_theCells,
      device_isOuterHitOfCell,
      region.ptMin(), region.origin().x(), region.origin().y(),
      region.originRBound(), caThetaCut, caPhiCut, caHardPtCut,
      maxNumberOfDoublets, maxNumberOfHits);

  kernel_find_ntuplets<<<numberOfBlocks_find, 1024, 0, cudaStream_>>>(
      numberOfRootLayerPairs, d_doublets, device_theCells,
      d_foundNtuplets[regionIndex],
      d_rootLayerPairs, 4, maxNumberOfDoublets);

  cudaMemcpyAsync(h_foundNtuplets[regionIndex], d_foundNtuplets[regionIndex],
                  sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>),
                  cudaMemcpyDeviceToHost, cudaStream_);
}

std::vector<std::array<std::pair<int, int>, 3>>
CAHitQuadrupletGeneratorGPU::fetchKernelResult(int regionIndex) {
  //this lazily resets temporary memory for the next event, and is not needed for reading the output
  cudaMemsetAsync(device_isOuterHitOfCell, 0,
                  maxNumberOfLayers * maxNumberOfHits *
                      sizeof(GPUSimpleVector<maxCellsPerHit, unsigned int>),
                  cudaStream_);

  std::vector<std::array<std::pair<int, int>, 3>> quadsInterface;

  for (int i = 0; i < h_foundNtuplets[regionIndex]->size(); ++i) {
    std::array<std::pair<int, int>, 3> tmpQuad = {
        {std::make_pair(h_foundNtuplets[regionIndex]->m_data[i].layerPairsAndCellId[0].x,
                        h_foundNtuplets[regionIndex]->m_data[i].layerPairsAndCellId[0].y -
                            maxNumberOfDoublets *
                                h_foundNtuplets[regionIndex]->m_data[i].layerPairsAndCellId[0].x),
         std::make_pair(h_foundNtuplets[regionIndex]->m_data[i].layerPairsAndCellId[1].x,
                        h_foundNtuplets[regionIndex]->m_data[i].layerPairsAndCellId[1].y -
                            maxNumberOfDoublets *
                                h_foundNtuplets[regionIndex]->m_data[i].layerPairsAndCellId[1].x),
         std::make_pair(h_foundNtuplets[regionIndex]->m_data[i].layerPairsAndCellId[2].x,
                        h_foundNtuplets[regionIndex]->m_data[i].layerPairsAndCellId[2].y -
                            maxNumberOfDoublets *
                                h_foundNtuplets[regionIndex]->m_data[i].layerPairsAndCellId[2].x)}};
    quadsInterface.push_back(tmpQuad);
  }

  return quadsInterface;
}
