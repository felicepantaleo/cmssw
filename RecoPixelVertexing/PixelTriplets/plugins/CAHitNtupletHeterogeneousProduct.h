#ifndef RecoPixelVertexing_PixelTriplets_CAHitNtupletHeterogeneousProduct_h
#define RecoPixelVertexing_PixelTriplets_CAHitNtupletHeterogeneousProduct_h
#include "FWCore/Utilities/interface/typedefs.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/PixelTrackingGPUConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "RecoPixelVertexing/PixelTriplets/interface/FakeRecoTrack.h"

#include "GPUCACell.h"


namespace CAHitNtupletHeterogeneousProduct {
  using CPUProduct = int;

  struct GPUProduct {
      Rfit::helix_fit* d_fitResults;
      FakeRecoTrack* d_recoTracks;
      std::vector<GPU::SimpleVector<Quadruplet>*> d_foundNtuplets;
      std::vector<Quadruplet*> d_foundNtupletsData;
  };

  using HeterogeneousGPUPixelTrack = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                            heterogeneous::GPUCudaProduct<GPUProduct> >;
}


#endif
