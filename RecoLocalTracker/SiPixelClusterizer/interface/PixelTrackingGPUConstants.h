#ifndef RecoLocalTracker_SiPixelClusterizer_interface_PixelTrackingGPUConstants_h
#define RecoLocalTracker_SiPixelClusterizer_interface_PixelTrackingGPUConstants_h

#include <cstdint>

namespace PixelGPUConstants {
#ifdef GPU_SMALL_EVENTS
  constexpr uint32_t maxNumberOfHits = 24*1024;
#else
  constexpr uint32_t maxNumberOfHits = 48*1024; // data at pileup 50 has 18300 +/- 3500 hits; 40000 is around 6 sigma away
#endif
}

#endif // RecoLocalTracker_SiPixelClusterizer_interface_PixelTrackingGPUConstants_h
