#include "CUDADataFormats/Track/interface/PixelTrackCUDA.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


PixelTrackCUDA::PixelTrackCUDA(TrackingRecHit2DSOAView const* hhp,
                                           cuda::stream_t<> &stream):
                hitsOnGPU_(hhp), m_nTracks(0) {
  edm::Service<CUDAService> cs;
  m_soa = cs->make_device_unique<SoA>(stream);
}


cudautils::host::unique_ptr<PixelTrackCUDA::SoA> 
PixelTrackCUDA::soaToHostAsync(cuda::stream_t<>& stream) const {
  edm::Service<CUDAService> cs;
  auto ret = cs->make_host_unique<SoA>(stream);
  cudaMemcpyAsync(ret.get(), m_soa.get(), sizeof(SoA), cudaMemcpyDefault, stream.id());
  return ret;
}

