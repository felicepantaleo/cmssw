// C++ headers
#include <algorithm>
#include <numeric>

// CUDA runtime
#include <cuda_runtime.h>

// CMSSW headers
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusteringConstants.h"
#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace {
  __global__
  void setHitsLayerStart(uint32_t const * __restrict__ hitsModuleStart, uint32_t* hitsLayerStart) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < 11) {
      hitsLayerStart[i] = hitsModuleStart[pixelCPEforGPU::layerStart[i]];
    }
  }
}

namespace pixelgpudetails {

  void PixelRecHitGPUKernel::makeHitsAsync(
                                           TrackingRecHit2DCUDA & hits_d,
                                           SiPixelDigisCUDA const& digis_d,
                                           SiPixelClustersCUDA const& clusters_d,
                                           BeamSpotCUDA const& bs_d,
                                           pixelCPEforGPU::ParamsOnGPU const * cpeParams,
                                           cuda::stream_t<>& stream
                                          ) const {


    int threadsPerBlock = 256;
    int blocks = digis_d.nModules(); // active modules (with digis)

#ifdef GPU_DEBUG
    std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
    if(blocks)  // protect from empty events
    gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0, stream.id()>>>(
      cpeParams,
      bs_d.data(),
      digis_d.moduleInd(),
      digis_d.xx(), digis_d.yy(), digis_d.adc(),
      clusters_d.moduleStart(),
      clusters_d.clusInModule(), clusters_d.moduleId(),
      digis_d.clus(),
      digis_d.nDigis(),
      clusters_d.clusModuleStart(),
      hits_d.view()
    );
    cudaCheck(cudaGetLastError());

    // assuming full warp of threads is better than a smaller number...
    setHitsLayerStart<<<1, 32, 0, stream.id()>>>(clusters_d.moduleStart(), hits_d.hitsLayerStart());
    cudaCheck(cudaGetLastError());

    auto nhits_ = clusters_d.nClusters();
    if (nhits_ >= TrackingRecHit2DSOAView::maxHits()) {
      edm::LogWarning("PixelRecHitGPUKernel" ) << "Hits Overflow " << nhits_  << " > " << TrackingRecHit2DSOAView::maxHits();
    } 

    if (nhits_)
    cudautils::fillManyFromVector(hits_d.phiBinner(), hits_d.phiBinnerWS(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nhits_, 256, stream.id());
    cudaCheck(cudaGetLastError());
  }
}
