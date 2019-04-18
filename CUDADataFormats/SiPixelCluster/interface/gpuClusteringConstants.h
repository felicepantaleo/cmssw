#ifndef CUDADataFormats_SiPixelCluster_gpuClusteringConstants_H
#define CUDADataFormats_SiPixelCluster_gpuClusteringConstants_H

#include <cstdint>

namespace pixelGPUConstants {
#ifdef GPU_SMALL_EVENTS
  constexpr uint32_t maxNumberOfHits = 24*1024;
#else
  constexpr uint32_t maxNumberOfHits = 48*1024; // data at pileup 50 has 18300 +/- 3500 hits; 40000 is around 6 sigma away
#endif
}

namespace gpuClustering {
  constexpr uint32_t MaxNumModules  = 2000;
  constexpr uint32_t MaxNumPixels   = 256 * 2000;   // this does not mean maxPixelPerModule == 256!
  constexpr uint32_t MaxNumClustersPerModules = 1024;
  constexpr uint32_t MaxNumClusters = pixelGPUConstants::maxNumberOfHits;
  constexpr uint16_t InvId          = 9999;         // must be > MaxNumModules

}

#endif 
