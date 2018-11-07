//
// Author: Felice Pantaleo, CERN
//

#include <cstdint>
#include <cuda_runtime.h>
#include <memory>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "CAHitQuadrupletGeneratorGPU.h"
#include "GPUCACell.h"
#include "gpuPixelDoublets.h"

using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;
using namespace Eigen;

__global__ void
KernelFastFitAllHits(GPU::SimpleVector<Quadruplet> * foundNtuplets,
    siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hhp,
    int hits_in_fit,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (helix_start >= foundNtuplets->size()) {
    return;
  }

#ifdef GPU_DEBUG
  printf("BlockDim.x: %d, BlockIdx.x: %d, threadIdx.x: %d, helix_start: %d, cumulative_size: %d\n",
      blockDim.x, blockIdx.x, threadIdx.x, helix_start, foundNtuplets->size());
#endif

  hits[helix_start].resize(3, hits_in_fit);
  hits_cov[helix_start].resize(3 * hits_in_fit, 3 * hits_in_fit);

  // Prepare data structure
  for (unsigned int i = 0; i < hits_in_fit; ++i) {
    auto hit = (*foundNtuplets)[helix_start].hitId[i];
    //  printf("Hit global_x: %f\n", hhp->xg_d[hit]);
    float ge[6];
    hhp->cpeParams->detParams(hhp->detInd_d[hit]).frame.toGlobal(hhp->xerr_d[hit], 0, hhp->yerr_d[hit], ge);
    //  printf("Error: %d: %f,%f,%f,%f,%f,%f\n",hhp->detInd_d[hit],ge[0],ge[1],ge[2],ge[3],ge[4],ge[5]);

    hits[helix_start].col(i) << hhp->xg_d[hit], hhp->yg_d[hit], hhp->zg_d[hit];

    for (auto j = 0; j < 3; ++j) {
      for (auto l = 0; l < 3; ++l) {
        // Index numerology:
        // i: index of the hits/point (0,..,3)
        // j: index of space component (x,y,z)
        // l: index of space components (x,y,z)
        // ge is always in sync with the index i and is formatted as:
        // ge[] ==> [xx, xy, xz, yy, yz, zz]
        // in (j,l) notation, we have:
        // ge[] ==> [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
        // so the index ge_idx corresponds to the matrix elements:
        // | 0  1  2 |
        // | 1  3  4 |
        // | 2  4  5 |
        auto ge_idx = j + l + (j > 0 and l > 0);
        hits_cov[helix_start](i + j * hits_in_fit, i + l * hits_in_fit) = ge[ge_idx];
      }
    }
  }
  fast_fit[helix_start] = Rfit::Fast_fit(hits[helix_start]);
}

__global__ void
KernelCircleFitAllHits(GPU::SimpleVector<Quadruplet> * foundNtuplets,
    int hits_in_fit,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (helix_start >= foundNtuplets->size()) {
    return;
  }

#ifdef GPU_DEBUG
    printf("BlockDim.x: %d, BlockIdx.x: %d, threadIdx.x: %d, helix_start: %d"
           "cumulative_size: %d\n",
           blockDim.x, blockIdx.x, threadIdx.x, helix_start, foundNtuplets->size());
#endif
  u_int n = hits[helix_start].cols();

  Rfit::VectorNd rad = (hits[helix_start].block(0, 0, 2, n).colwise().norm());

  circle_fit[helix_start] =
      Rfit::Circle_fit(hits[helix_start].block(0, 0, 2, n),
                       hits_cov[helix_start].block(0, 0, 2 * n, 2 * n),
                       fast_fit[helix_start], rad, B, true);

#ifdef GPU_DEBUG
    printf("KernelCircleFitAllHits circle.par(0): %d %f\n", helix_start,
           circle_fit[helix_start].par(0));
    printf("KernelCircleFitAllHits circle.par(1): %d %f\n", helix_start,
           circle_fit[helix_start].par(1));
    printf("KernelCircleFitAllHits circle.par(2): %d %f\n", helix_start,
           circle_fit[helix_start].par(2));
#endif
}

__global__ void
KernelLineFitAllHits(GPU::SimpleVector<Quadruplet> * foundNtuplets,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (helix_start >= foundNtuplets->size()) {
    return;
  }

#ifdef GPU_DEBUG

    printf("BlockDim.x: %d, BlockIdx.x: %d, threadIdx.x: %d, helix_start: %d, "
           "cumulative_size: %d\n",
           blockDim.x, blockIdx.x, threadIdx.x, helix_start, foundNtuplets->size());
#endif

  line_fit[helix_start] =
      Rfit::Line_fit(hits[helix_start], hits_cov[helix_start],
                     circle_fit[helix_start], fast_fit[helix_start], B, true);

  par_uvrtopak(circle_fit[helix_start], B, true);

  // Grab helix_fit from the proper location in the output vector
  Rfit::helix_fit &helix = results[helix_start];
  helix.par << circle_fit[helix_start].par, line_fit[helix_start].par;

  // TODO: pass properly error booleans

  helix.cov = MatrixXd::Zero(5, 5);
  helix.cov.block(0, 0, 3, 3) = circle_fit[helix_start].cov;
  helix.cov.block(3, 3, 2, 2) = line_fit[helix_start].cov;

  helix.q = circle_fit[helix_start].q;
  helix.chi2_circle = circle_fit[helix_start].chi2;
  helix.chi2_line = line_fit[helix_start].chi2;

#ifdef GPU_DEBUG

    printf("KernelLineFitAllHits line.par(0): %d %f\n", helix_start,
           circle_fit[helix_start].par(0));
    printf("KernelLineFitAllHits line.par(1): %d %f\n", helix_start,
           line_fit[helix_start].par(1));
#endif
}

// __global__
// void kernel_filterGPUPixelTracks(Rfit::helix_fit * const __restrict__  helix_fit_results,
//   GPU::SimpleVector<Quadruplet> * const __restrict__ foundNtuplets,
//   CAHitNtupletHeterogeneousProduct::GPUProduct* filteredPixelTracks, float fitMaxChi2, float fitMaxTip, float nSigmaTipMaxTolerance )
// {
//   const auto nSeeds = foundNtuplets->size();
//   auto idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if(idx < nSeeds)
//   {
//     const auto& helix = helix_fit_results[idx];
//     float valTip = helix.par(1);
//     float errTip = std::sqrt(helix.cov(1, 1));
//     if( (std::abs(valTip) - fitMaxTip < nSigmaTipMaxTolerance * errTip ) and
//         (helix.chi2_circle + helix.chi2_line <= fitMaxChi2))
//     {
//       filteredPixelTracks->emplace_back(CAHitNtupletHeterogeneousProduct::GPUPixelTrack{helix, (*foundNtuplets)[idx]});
//     }
//   }
// }


__global__
void kernel_makeGPUPixelTracks(Rfit::helix_fit * const __restrict__  helix_fit_results,
  Rfit::circle_fit * const __restrict__ circle_fit_results,
  GPU::SimpleVector<Quadruplet> * const __restrict__ foundNtuplets,
  FakeRecoTrack* recoTracks)
{
  const auto nSeeds = foundNtuplets->size();
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < nSeeds)
  {
    const auto& helix = helix_fit_results[idx];
    auto& recoTrack = recoTracks[idx];
    recoTrack.charge = circle_fit_results[idx].q;
    recoTrack.chi2 = helix.chi2_circle + helix.chi2_line;
    recoTrack.ndof = 3;
    recoTrack.covariance[0] =  helix.cov(0,0);
    recoTrack.covariance[1] =  helix.cov(0,1);
    recoTrack.covariance[2] =  helix.cov(1,1);
    recoTrack.covariance[3] =  helix.cov(0,2);
    recoTrack.covariance[4] =  helix.cov(1,2);
    recoTrack.covariance[5] =  helix.cov(2,2);
    recoTrack.covariance[6] =  helix.cov(0,3);
    recoTrack.covariance[7] =  helix.cov(1,3);
    recoTrack.covariance[8] =  helix.cov(2,3);
    recoTrack.covariance[9] =  helix.cov(3,3);
    recoTrack.covariance[10] = helix.cov(0,4);
    recoTrack.covariance[11] = helix.cov(1,4);
    recoTrack.covariance[12] = helix.cov(2,4);
    recoTrack.covariance[13] = helix.cov(3,4);
    recoTrack.covariance[14] = helix.cov(4,4);
    float valTip = helix.par(1);
    float errTip = std::sqrt(helix.cov(1, 1));

    //TODO: Fill the parameters of the track and point of closest approach position
    // recoTrack.vertexPos[0] =
    // Vector5d par;  //!<(phi,Tip,pt,cotan(theta)),Zip)
    // Matrix5d cov;
    // /*!< ()->cov() \n
    //   |(phi,phi)|(Tip,phi)|(p_t,phi)|(c_t,phi)|(Zip,phi)| \n
    //   |(phi,Tip)|(Tip,Tip)|(p_t,Tip)|(c_t,Tip)|(Zip,Tip)| \n
    //   |(phi,p_t)|(Tip,p_t)|(p_t,p_t)|(c_t,p_t)|(Zip,p_t)| \n
    //   |(phi,c_t)|(Tip,c_t)|(p_t,c_t)|(c_t,c_t)|(Zip,c_t)| \n
    //   |(phi,Zip)|(Tip,Zip)|(p_t,Zip)|(c_t,Zip)|(Zip,Zip)|
    // */
  }
}


__global__ void
kernel_checkOverflows(GPU::SimpleVector<Quadruplet> *foundNtuplets,
               GPUCACell *cells, uint32_t const * nCells,
               GPU::VecArray< unsigned int, 256> *isOuterHitOfCell,
               uint32_t nHits, uint32_t maxNumberOfDoublets) {

 auto idx = threadIdx.x + blockIdx.x * blockDim.x;
 #ifdef GPU_DEBUG
 if (0==idx)
   printf("number of found cells %d\n",*nCells);
 #endif
 if (idx < (*nCells) ) {
   auto &thisCell = cells[idx];
   if (thisCell.theOuterNeighbors.full()) //++tooManyNeighbors[thisCell.theLayerPairId];
     printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.theLayerPairId);
 }
 if (idx < nHits) {
   if (isOuterHitOfCell[idx].full()) // ++tooManyOuterHitOfCell;
     printf("OuterHitOfCell overflow %d\n", idx);
 }
}


__global__ void
kernel_connect(GPU::SimpleVector<Quadruplet> *foundNtuplets,
               GPUCACell *cells, uint32_t const * nCells,
               GPU::VecArray< unsigned int, 256> *isOuterHitOfCell,
               float ptmin,
               float region_origin_radius, const float thetaCut,
               const float phiCut, const float hardPtCut,
               unsigned int maxNumberOfDoublets, unsigned int maxNumberOfHits_) {

  float region_origin_x = 0.;
  float region_origin_y = 0.;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (0==cellIndex) foundNtuplets->reset(); // ready for next kernel

  if (cellIndex >= (*nCells) ) return;
  auto &thisCell = cells[cellIndex];
  auto innerHitId = thisCell.get_inner_hit_id();
  auto numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
  for (auto j = 0; j < numberOfPossibleNeighbors; ++j) {
     auto otherCell = isOuterHitOfCell[innerHitId][j];

     if (thisCell.check_alignment_and_tag(
                 cells, otherCell, ptmin, region_origin_x, region_origin_y,
                  region_origin_radius, thetaCut, phiCut, hardPtCut)
        ) {
          cells[otherCell].theOuterNeighbors.push_back(cellIndex);
     }
  }
}

__global__ void kernel_find_ntuplets(
    GPUCACell *cells, uint32_t const * nCells,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets)
{

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndex >= (*nCells)) return;
  auto &thisCell = cells[cellIndex];
  if (thisCell.theLayerPairId!=0 && thisCell.theLayerPairId!=3 && thisCell.theLayerPairId!=8) return; // inner layer is 0 FIXME
  GPU::VecArray<unsigned int, 3> stack;
  stack.reset();
  thisCell.find_ntuplets(cells, foundNtuplets, stack, minHitsPerNtuplet);
  assert(stack.size()==0);
  // printf("in %d found quadruplets: %d\n", cellIndex, foundNtuplets->size());
}

__global__ void
kernel_print_found_ntuplets(GPU::SimpleVector<Quadruplet> *foundNtuplets, int maxPrint) {
  for (int i = 0; i < std::min(maxPrint, foundNtuplets->size()); ++i) {
    printf("\nquadruplet %d: %d %d %d %d\n", i,
           (*foundNtuplets)[i].hitId[0],
           (*foundNtuplets)[i].hitId[1],
           (*foundNtuplets)[i].hitId[2],
           (*foundNtuplets)[i].hitId[3]
          );
  }
}

void CAHitQuadrupletGeneratorGPU::deallocateOnGPU()
{
  for (size_t i = 0; i < h_foundNtupletsVec_.size(); ++i)
  {
    cudaFreeHost(h_foundNtupletsVec_[i]);
    cudaFreeHost(h_foundNtupletsData_[i]);
    cudaFree(d_foundNtupletsVec_[i]);
    cudaFree(d_foundNtupletsData_[i]);
    cudaFreeHost(h_foundGPUPixelTracks_[i]);
    cudaFree(d_foundGPUPixelTracks_[i]);
  }

  cudaFree(device_theCells_);
  cudaFree(device_isOuterHitOfCell_);
  cudaFree(device_nCells_);

  // Free Riemann Fit stuff
  cudaFree(hitsGPU_);
  cudaFree(hits_covGPU_);
  cudaFree(fast_fit_resultsGPU_);
  cudaFree(circle_fit_resultsGPU_);
  cudaFree(line_fit_resultsGPU_);
  cudaFree(helix_fit_resultsGPU_);
  cudaFree(d_recoTracks_);
}



void CAHitQuadrupletGeneratorGPU::allocateOnGPU()
{
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  cudaCheck(cudaMalloc(&device_theCells_,
             PixelGPUConstants::MaxNumOfDoublets * sizeof(GPUCACell)));
  cudaCheck(cudaMalloc(&device_nCells_, sizeof(uint32_t)));
  cudaCheck(cudaMemset(device_nCells_, 0, sizeof(uint32_t)));

  cudaCheck(cudaMalloc(&device_isOuterHitOfCell_,
             PixelGPUConstants::maxNumberOfHits * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>)));
  cudaCheck(cudaMemset(device_isOuterHitOfCell_, 0,
             PixelGPUConstants::maxNumberOfHits * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>)));

  h_foundNtupletsVec_.resize(maxNumberOfRegions_);
  h_foundNtupletsData_.resize(maxNumberOfRegions_);
  d_foundNtupletsVec_.resize(maxNumberOfRegions_);
  d_foundNtupletsData_.resize(maxNumberOfRegions_);
  h_foundGPUPixelTracks_.resize(maxNumberOfRegions_);
  d_foundGPUPixelTracks_.resize(maxNumberOfRegions_);

  // FIXME this could be rewritten with a single pair of cudaMallocHost / cudaMalloc
  for (int i = 0; i < maxNumberOfRegions_; ++i) {
    cudaCheck(cudaMallocHost(&h_foundNtupletsData_[i],  sizeof(Quadruplet) * PixelGPUConstants::maxNumberOfQuadruplets));
    cudaCheck(cudaMallocHost(&h_foundNtupletsVec_[i],   sizeof(GPU::SimpleVector<Quadruplet>)));
    new(h_foundNtupletsVec_[i]) GPU::SimpleVector<Quadruplet>(PixelGPUConstants::maxNumberOfQuadruplets, h_foundNtupletsData_[i]);
    cudaCheck(cudaMalloc(&d_foundNtupletsData_[i],      sizeof(Quadruplet) * PixelGPUConstants::maxNumberOfQuadruplets));
    cudaCheck(cudaMemset(d_foundNtupletsData_[i], 0x00, sizeof(Quadruplet) * PixelGPUConstants::maxNumberOfQuadruplets));
    cudaCheck(cudaMalloc(&d_foundNtupletsVec_[i],       sizeof(GPU::SimpleVector<Quadruplet>)));
    GPU::SimpleVector<Quadruplet> tmp_foundNtuplets(PixelGPUConstants::maxNumberOfQuadruplets, d_foundNtupletsData_[i]);
    cudaCheck(cudaMemcpy(d_foundNtupletsVec_[i], & tmp_foundNtuplets, sizeof(GPU::SimpleVector<Quadruplet>), cudaMemcpyDefault));

    cudaCheck(cudaMallocHost(&h_foundGPUPixelTracks_[i],sizeof(CAHitNtupletHeterogeneousProduct::GPUProduct)));
    cudaCheck(cudaMalloc(&d_foundGPUPixelTracks_[i],sizeof(CAHitNtupletHeterogeneousProduct::GPUProduct)));
    cudaCheck(cudaMemset(d_foundGPUPixelTracks_[i], 0x00, sizeof(CAHitNtupletHeterogeneousProduct::GPUProduct)));

  }

  // Riemann-Fit related allocations
  cudaCheck(cudaMalloc(&hitsGPU_, 48 * PixelGPUConstants::maxNumberOfQuadruplets * sizeof(Rfit::Matrix3xNd(3, 4))));
  cudaCheck(cudaMemset(hitsGPU_, 0x00, 48 * PixelGPUConstants::maxNumberOfQuadruplets * sizeof(Rfit::Matrix3xNd(3, 4))));

  cudaCheck(cudaMalloc(&hits_covGPU_, 48 * PixelGPUConstants::maxNumberOfQuadruplets * sizeof(Rfit::Matrix3Nd(12, 12))));
  cudaCheck(cudaMemset(hits_covGPU_, 0x00, 48 * PixelGPUConstants::maxNumberOfQuadruplets * sizeof(Rfit::Matrix3Nd(12, 12))));

  cudaCheck(cudaMalloc(&fast_fit_resultsGPU_, 48 * PixelGPUConstants::maxNumberOfQuadruplets * sizeof(Vector4d)));
  cudaCheck(cudaMemset(fast_fit_resultsGPU_, 0x00, 48 * PixelGPUConstants::maxNumberOfQuadruplets * sizeof(Vector4d)));

  cudaCheck(cudaMalloc(&circle_fit_resultsGPU_, 48 * PixelGPUConstants::maxNumberOfQuadruplets * sizeof(Rfit::circle_fit)));
  cudaCheck(cudaMemset(circle_fit_resultsGPU_, 0x00, 48 * PixelGPUConstants::maxNumberOfQuadruplets * sizeof(Rfit::circle_fit)));

  cudaCheck(cudaMalloc(&line_fit_resultsGPU_, PixelGPUConstants::maxNumberOfQuadruplets * sizeof(Rfit::line_fit)));
  cudaCheck(cudaMemset(line_fit_resultsGPU_, 0x00, PixelGPUConstants::maxNumberOfQuadruplets * sizeof(Rfit::line_fit)));

  cudaCheck(cudaMalloc(&helix_fit_resultsGPU_, sizeof(Rfit::helix_fit)*PixelGPUConstants::maxNumberOfQuadruplets));
  cudaCheck(cudaMemset(helix_fit_resultsGPU_, 0x00, sizeof(Rfit::helix_fit)*PixelGPUConstants::maxNumberOfQuadruplets));

  cudaCheck(cudaMalloc(&d_recoTracks_, sizeof(FakeRecoTrack)*PixelGPUConstants::maxNumberOfQuadruplets)));
  h_product_.fitResults = helix_fit_resultsGPU_;
  h_product_.d_foundNtuplets = d_foundNtupletsVec_;
  h_product.d_foundNtupletsData = d_foundNtupletsData_;
  h_product.d_recoTracks = d_recoTracks_;

}

void CAHitQuadrupletGeneratorGPU::launchKernels(const TrackingRegion &region,
                                                int regionIndex, HitsOnCPU const & hh,
                                                bool transferToCPU,
                                                cudaStream_t cudaStream)
{
  assert(regionIndex < maxNumberOfRegions_);
  assert(0==regionIndex);

  h_foundNtupletsVec_[regionIndex]->reset();

  auto nhits = hh.nHits;
  assert(nhits <= PixelGPUConstants::maxNumberOfHits);
  auto numberOfBlocks = (PixelGPUConstants::MaxNumOfDoublets + 512 - 1)/512;
  kernel_connect<<<numberOfBlocks, 512, 0, cudaStream>>>(
      d_foundNtupletsVec_[regionIndex], // needed only to be reset, ready for next kernel
      device_theCells_, device_nCells_,
      device_isOuterHitOfCell_,
      region.ptMin(),
      region.originRBound(), caThetaCut, caPhiCut, caHardPtCut,
      PixelGPUConstants::MaxNumOfDoublets, PixelGPUConstants::maxNumberOfHits
  );
  cudaCheck(cudaGetLastError());

  kernel_find_ntuplets<<<numberOfBlocks, 512, 0, cudaStream>>>(
      device_theCells_, device_nCells_,
      d_foundNtupletsVec_[regionIndex],
      4, PixelGPUConstants::MaxNumOfDoublets);
  cudaCheck(cudaGetLastError());


  numberOfBlocks = (std::max(int(nhits), PixelGPUConstants::MaxNumOfDoublets) + 512 - 1)/512;
  kernel_checkOverflows<<<numberOfBlocks, 512, 0, cudaStream>>>(
                        d_foundNtupletsVec_[regionIndex],
                        device_theCells_, device_nCells_,
                        device_isOuterHitOfCell_, nhits,
                        PixelGPUConstants::MaxNumOfDoublets
                       );


  // kernel_print_found_ntuplets<<<1, 1, 0, cudaStream>>>(d_foundNtupletsVec_[regionIndex], 10);


  numberOfBlocks = (PixelGPUConstants::maxNumberOfQuadruplets + 512 - 1)/512;

  KernelFastFitAllHits<<<numberOfBlocks, 512, 0, cudaStream>>>(
      d_foundNtupletsVec_[regionIndex], hh.gpu_d, 4, bField_, helix_fit_resultsGPU_,
      hitsGPU_, hits_covGPU_, circle_fit_resultsGPU_, fast_fit_resultsGPU_,
      line_fit_resultsGPU_);
  cudaCheck(cudaGetLastError());

  KernelCircleFitAllHits<<<numberOfBlocks, 256, 0, cudaStream>>>(
      d_foundNtupletsVec_[regionIndex], 4, bField_, helix_fit_resultsGPU_,
      hitsGPU_, hits_covGPU_, circle_fit_resultsGPU_, fast_fit_resultsGPU_,
      line_fit_resultsGPU_);
  cudaCheck(cudaGetLastError());

  KernelLineFitAllHits<<<numberOfBlocks, 256, 0, cudaStream>>>(
      d_foundNtupletsVec_[regionIndex], bField_, helix_fit_resultsGPU_,
      hitsGPU_, hits_covGPU_, circle_fit_resultsGPU_, fast_fit_resultsGPU_,
      line_fit_resultsGPU_);
  cudaCheck(cudaGetLastError());


  kernel_filterGPUPixelTracks<<<numberOfBlocks, 256, 0, cudaStream>>>(
    helix_fit_resultsGPU_, d_foundNtupletsVec_[regionIndex], d_foundGPUPixelTracks_[regionIndex],
    fitMaxChi2_,  fitMaxTip_, nSigmaTipMaxTolerance_ );

  if(transferToCPU) {
    cudaCheck(cudaMemcpyAsync(h_foundNtupletsVec_[regionIndex], d_foundNtupletsVec_[regionIndex],
                              sizeof(GPU::SimpleVector<Quadruplet>),
                              cudaMemcpyDeviceToHost, cudaStream));

    cudaCheck(cudaMemcpyAsync(h_foundNtupletsData_[regionIndex], d_foundNtupletsData_[regionIndex],
                              PixelGPUConstants::maxNumberOfQuadruplets*sizeof(Quadruplet),
                              cudaMemcpyDeviceToHost, cudaStream));
  }
}

void CAHitQuadrupletGeneratorGPU::cleanup(cudaStream_t cudaStream) {
  // this lazily resets temporary memory for the next event, and is not needed for reading the output
  cudaCheck(cudaMemsetAsync(device_isOuterHitOfCell_, 0,
                            PixelGPUConstants::maxNumberOfHits * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>),
                            cudaStream));
  cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), cudaStream));
}

std::vector<std::array<int, 4>>
CAHitQuadrupletGeneratorGPU::fetchKernelResult(int regionIndex)
{
  assert(0==regionIndex);
  h_foundNtupletsVec_[regionIndex]->set_data(h_foundNtupletsData_[regionIndex]);

  std::vector<std::array<int, 4>> quadsInterface(h_foundNtupletsVec_[regionIndex]->size());
  for (int i = 0; i < h_foundNtupletsVec_[regionIndex]->size(); ++i) {
    for (int j = 0; j<4; ++j) quadsInterface[i][j] = (*h_foundNtupletsVec_[regionIndex])[i].hitId[j];
  }
  return quadsInterface;
}

void CAHitQuadrupletGeneratorGPU::buildDoublets(HitsOnCPU const & hh, cudaStream_t stream) {
  auto nhits = hh.nHits;

  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize;
  int blocks = (3 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  gpuPixelDoublets::getDoubletsFromHisto<<<blocks, threadsPerBlock, 0, stream>>>(device_theCells_, device_nCells_, hh.gpu_d, device_isOuterHitOfCell_);
  cudaCheck(cudaGetLastError());
}
