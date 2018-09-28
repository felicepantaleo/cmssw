#ifndef RecoPixelVertexing_PixelTriplets_CAHitNtupletHeterogeneousProduct_h
#define RecoPixelVertexing_PixelTriplets_CAHitNtupletHeterogeneousProduct_h
#include "FWCore/Utilities/interface/typedefs.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/PixelTrackingGPUConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"
#include "GPUCACell.h"


namespace CAHitNtupletHeterogeneousProduct {
  using CPUProduct = int;

  struct GPUPixelTrack {
      Rfit::helix_fit fitResults;
      Quadruplet quadruplet;
  };

  using GPUProduct = GPU::VecArray<GPUPixelTrack, PixelGPUConstants::maxNumberOfQuadruplets>;

  using HeterogeneousGPUPixelTrack = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                            heterogeneous::GPUCudaProduct<GPUProduct> >;
}


#endif
