#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"

BeamSpotCUDA::BeamSpotCUDA(cudautils::host::unique_ptr<Data> data_h, cuda::stream_t<>& stream) {
  edm::Service<CUDAService> cs;

  data_d_ = cs->make_device_unique<Data>(stream);
  cudautils::copyAsync(data_d_, data_h, stream);
}
