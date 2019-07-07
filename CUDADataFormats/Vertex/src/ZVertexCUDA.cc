#include "CUDADataFormats/Vertex/interface/ZVertexCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


ZVertexCUDA::ZVertexCUDA(TkSoA const *  trackOnGPU, cuda::stream_t<> &stream) :
             m_trackOnGPU(trackOnGPU) {
  edm::Service<CUDAService> cs;
  m_soa = cs->make_device_unique<SoA>(stream);
}


cudautils::host::unique_ptr<ZVertexCUDA::SoA> 
ZVertexCUDA::soaToHostAsync(cuda::stream_t<>& stream) const{
  edm::Service<CUDAService> cs;
  auto ret = cs->make_host_unique<SoA>(stream);
  cudaMemcpyAsync(ret.get(), m_soa.get(), sizeof(SoA), cudaMemcpyDefault, stream.id());
  return ret;
}

