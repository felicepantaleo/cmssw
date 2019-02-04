//
// Author: Felice Pantaleo, CERN
//

#include "HelixFitOnGPU.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/BrokenLine.h"

#include <cstdint>
#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"


using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;

using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;

using namespace Eigen;

__global__
void kernelBLFastFit(TuplesOnGPU::Container const * __restrict__ foundNtuplets,
    HitsOnGPU const * __restrict__ hhp,
    int hits_in_fit,
    double * __restrict__ phits,
    float * __restrict__ phits_ge,
    double * __restrict__ pfast_fit,
    uint32_t offset)
{

  assert(hits_in_fit==4); // FixMe later template

  assert(pfast_fit); assert(foundNtuplets);

  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  auto helix_start = local_start + offset;

  if (helix_start>=foundNtuplets->nbins()) return;
  if (foundNtuplets->size(helix_start)<hits_in_fit) {
    return;
  }

  Rfit::Map3x4d hits(phits+local_start);
  Rfit::Map4d   fast_fit(pfast_fit+local_start);
  Rfit::Map6x4f hits_ge(phits_ge+local_start);

#ifdef BL_DUMP_HITS
  __shared__ int done;
  done = 0;
  __syncthreads(); 
  bool dump =  (foundNtuplets->size(helix_start)==5 &&
                0 == atomicAdd(&done,1));
  auto imax = foundNtuplets->size(helix_start);
#else
  auto imax = hits_in_fit;
#endif

  // Prepare data structure
  auto const * hitId = foundNtuplets->begin(helix_start);
  for (unsigned int i = 0; i < imax; ++i) {
    auto hit = hitId[i];
    float ge[6];
    hhp->cpeParams->detParams(hhp->detInd_d[hit]).frame.toGlobal(hhp->xerr_d[hit], 0, hhp->yerr_d[hit], ge);
#ifdef BL_DUMP_HITS
    if (dump){
      printf("Hit global: %d: %d hits.col(%d) << %f,%f,%f\n", helix_start, hhp->detInd_d[hit],i,hhp->xg_d[hit],hhp->yg_d[hit],hhp->zg_d[hit]);
      printf("Error: %d: %d  hits_ge.col(%d) << %e,%e,%e,%e,%e,%e\n",helix_start,hhp->detInd_d[hit],i,ge[0],ge[1],ge[2],ge[3],ge[4],ge[5]);
    }
    if (i>=hits_in_fit) continue;
#endif
    hits.col(i) << hhp->xg_d[hit], hhp->yg_d[hit], hhp->zg_d[hit];
    hits_ge.col(i) << ge[0],ge[1],ge[2],ge[3],ge[4],ge[5];
  }
  BrokenLine::BL_Fast_fit(hits,fast_fit);

  // no NaN here....
  assert(fast_fit(0)==fast_fit(0));
  assert(fast_fit(1)==fast_fit(1));
  assert(fast_fit(2)==fast_fit(2));
  assert(fast_fit(3)==fast_fit(3));

}

__global__
void kernelBLFit(TuplesOnGPU::Container const * __restrict__ foundNtuplets,
    int hits_in_fit,
    double B,
    Rfit::helix_fit *results,
    double * __restrict__ phits,
    float * __restrict__ phits_ge,
    double * __restrict__ pfast_fit,
    uint32_t offset)
{

  assert(results); assert(pfast_fit);

  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  auto helix_start = local_start + offset;

  if (helix_start>=foundNtuplets->nbins()) return;
  if (foundNtuplets->size(helix_start)<hits_in_fit) {
    return;
  }

  Rfit::Map3x4d hits(phits+local_start);
  Rfit::Map4d   fast_fit(pfast_fit+local_start);
  Rfit::Map6x4f hits_ge(phits_ge+local_start);

  constexpr uint32_t N = Rfit::Map3x4d::ColsAtCompileTime;
  
  BrokenLine::PreparedBrokenLineData<N> data;
  Rfit::Matrix3d Jacob;

  BrokenLine::karimaki_circle_fit circle;
  Rfit::line_fit line;
 
  BrokenLine::prepareBrokenLineData(hits,fast_fit,B,data);
  BrokenLine::BL_Line_fit(hits_ge,fast_fit,B,data,line);
  BrokenLine::BL_Circle_fit(hits,hits_ge,fast_fit,B,data,circle);
  Jacob << 1,0,0,
    0,1,0,
    0,0,-B/std::copysign(Rfit::sqr(circle.par(2)),circle.par(2));
  circle.par(2)=B/std::abs(circle.par(2));
  circle.cov=Jacob*circle.cov*Jacob.transpose();


  // Grab helix_fit from the proper location in the output vector
  auto & helix = results[helix_start];
  helix.par << circle.par, line.par;

  helix.cov = Rfit::Matrix5d::Zero();
  helix.cov.block(0, 0, 3, 3) = circle.cov;
  helix.cov.block(3, 3, 2, 2) = line.cov;

  helix.q = circle.q;
  helix.chi2_circle = circle.chi2;
  helix.chi2_line = line.chi2;

#ifdef GPU_DEBUG
  printf("kernelBLFit circle.par(0,1,2): %d %f,%f,%f\n", helix_start,
         circle.par(0), circle.par(1), circle.par(2));
  printf("kernelLineFitAllHits line.par(0,1): %d %f,%f\n", helix_start, line.par(0),line.par(1));
  printf("kernelLineFitAllHits chi2 cov %f/%f %f,%f,%f,%f,%f\n",helix.chi2_circle,helix.chi2_line, 
         helix.cov(0,0),helix.cov(1,1),helix.cov(2,2),helix.cov(3,3),helix.cov(4,4));
#endif
}


void HelixFitOnGPU::launchBrokenLineKernels(HitsOnCPU const & hh, uint32_t nhits, uint32_t maxNumberOfTuples, cudaStream_t cudaStream)
{
    assert(tuples_d); assert(fast_fit_resultsGPU_);

    auto blockSize = 128;
    auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

    for (uint32_t offset=0; offset<maxNumberOfTuples; offset+=maxNumberOfConcurrentFits_) {
      kernelBLFastFit<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
          tuples_d, hh.gpu_d, 4,
          hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_,offset);
      cudaCheck(cudaGetLastError());

      kernelBLFit<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
             tuples_d, 4,  bField_, helix_fit_results_d,
             hitsGPU_, hits_geGPU_, fast_fit_resultsGPU_,
             offset);
      cudaCheck(cudaGetLastError());
    }
}
