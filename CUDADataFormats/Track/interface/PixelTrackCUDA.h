#ifndef CUDADataFormatsTrackTrackSoA_H
#define CUDADataFormatsTrackTrackSoA_H

#include "CUDADataFormats/Track/interface/TrajectoryStateSoA.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include <cuda/api_wrappers.h>
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"




enum TrackQuality : uint8_t { bad=0, dup, loose, strict, tight, highPurity };


template <int32_t S>
class TrackSoA {
public:

  static constexpr int32_t stride() { return S; }

  using Quality = TrackQuality;

  // Always check quality is at least loose!
  eigenSoA::ScalarSoA<Quality, S> quality;

  // this is chi2/ndof as not necessarely all hits are used in the fit  
  eigenSoA::ScalarSoA<float, S> chi2;

  int nHits(int i) const { return detIndices.size(i);}

  // State at the Beam spot
  // phi,tip,1/pt,cotan(theta),zip
  TrajectoryStateSoA<S> stateAtBS;
  eigenSoA::ScalarSoA<float, S> eta;
  eigenSoA::ScalarSoA<float, S> pt;
  float phi(int32_t i) const { return stateAtBS.state(i)(0); }
  float tip(int32_t i) const { return stateAtBS.state(i)(1); }
  float zip(int32_t i) const { return stateAtBS.state(i)(2); }

  // state at the detector of the outermost hit
  // representation to be decided...
  // not yet filled on GPU
  TrajectoryStateSoA<S> stateAtOuterDet;

  using hindex_type = uint16_t;  
  using HitContainer = OneToManyAssoc<hindex_type, S, 5 * S>;

  HitContainer hitIndices;
  HitContainer detIndices;
  
  // total number of tracks (including those not fitted)
  uint32_t m_nTracks;

};



class TrackingRecHit2DSOAView;

class PixelTrackCUDA {
public:

#ifdef GPU_SMALL_EVENTS
  static constexpr uint32_t maxNumber = 2 * 1024;
#else
  static constexpr uint32_t maxNumber = 32 * 1024;
#endif

  using SoA = TrackSoA<maxNumber>;
  using TrajectoryState = TrajectoryStateSoA<maxNumber>;
  using HitContainer = Soa::HitContainer;
  using Quality = TrackQuality;

  PixelTrackCUDA(){}
  PixelTrackCUDA(TrackingRecHit2DSOAView const* hhp, cuda::stream_t<> &stream);

  auto * soa() { return m_soa.get();}
  auto const * soa() const { return m_soa.get();}
  auto * apc() { return (AtomicPairCounter*)m_apc.get();}
  auto const * apc() const { return (AtomicPairCounter const*)m_apc.get();}

  TrackingRecHit2DSOAView const * hitsOnGPU() const { return hitsOnGPU_;}

  cudautils::host::unique_ptr<SoA> soaToHostAsync(cuda::stream_t<>& stream) const;

private:

  std::vector<uint32_t> m_indToEdm;  // index of    tuple in reco tracks....

  TrackingRecHit2DSOAView const* hitsOnGPU_ = nullptr;  // forwarding

  cudautils::device::unique_ptr<SoA> m_soa;
  cudautils::device::unique_ptr<AtomicPairCounter::c_type> m_apc;

  uint32_t m_nTracks;

};

#endif // CUDADataFormatsTrackTrackSoA_H

