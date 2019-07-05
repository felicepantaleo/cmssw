#ifndef CUDADataFormatsTrackTrackSoA_H
#define CUDADataFormatsTrackTrackSoA_H

#include "CUDADataFormats/Track/interface/TrajectoryStateSoA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include <cuda/api_wrappers.h>
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"



template <int32_t S>
class TrackSoA {
public:

  enum Quality : uint8_t { bad=0, dup, loose, strict, tight, highPurity };

  eigenSoA::ScalarSoA<Quality, S> quality;

  TrajectoryStateSoA<S> stateAtBS;
  eigenSoA::ScalarSoA<float, S> eta;

  TrajectoryStateSoA<S> stateAtOuterDet;

  using hindex_type = uint16_t;  
  using HitContainer = OneToManyAssoc<hindex_type, S, 5 * S>;

  HitContainter hitIndices;
  HitContainter detIndices;

};

class PixelTrackCUDA {
public:

#ifdef GPU_SMALL_EVENTS
  constexpr uint32_t maxNumber() { return 3 * 1024; }
#else
  constexpr uint32_t maxNumber() { return 24 * 1024; }
#endif

 using SoA = TrackSoA<maxNumber()>;


private:

  std::vector<uint32_t> m_indToEdm;  // index of    tuple in reco tracks....

  TrackingRecHit2DSOAView const* hitsOnGPU_d = nullptr;  // forwarding

  cudautils::device::unique_ptr<Soa> m_soa;
  cudautils::device::unique_ptr<AtomicPairCounter> m_apc;

  uint32_t m_nTuples;


}

#endif // CUDADataFormatsTrackTrackSoA_H

