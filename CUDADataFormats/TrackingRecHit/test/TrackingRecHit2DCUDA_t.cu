#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"


namespace testTrackingRecHit2D {

  __global__ 
  void fill(TrackingRecHit2DSOAView * hits) {

  }


  __global__
  void verify(TrackingRecHit2DSOAView const * hits) {

  }

  void
  runKernels(TrackingRecHit2DSOAView * hits) {
    fill<<<1,1024>>>(hits);
    verify<<<1,1024>>>(hits);
  }

}


