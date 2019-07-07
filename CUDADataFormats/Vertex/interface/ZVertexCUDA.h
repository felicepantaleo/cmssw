#ifndef CUDADataFormatsVertexZVertexCUDA_H
#define CUDADataFormatsVertexZVertexCUDA_H

#include "CUDADataFormats/Vertex/interface/ZVertexSoA.h"

#include <cuda/api_wrappers.h>
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"



class ZVertexCUDA {
public:
  using SoA = ZVertexSoA;

  ZVertexCUDA(){}
  ZVertexCUDA(cuda::stream_t<> &stream);

  auto * soa() { return m_soa.get();}
  auto const * soa() const { return m_soa.get();}


  cudautils::host::unique_ptr<SoA> soaToHostAsync(cuda::stream_t<>& stream) const;

private:

  cudautils::device::unique_ptr<SoA> m_soa;


};

#endif
