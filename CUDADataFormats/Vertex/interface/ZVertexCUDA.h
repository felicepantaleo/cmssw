#ifndef CUDADataFormatsVertexZVertexCUDA_H
#define CUDADataFormatsVertexZVertexCUDA_H

#include "CUDADataFormats/Vertex/interface/ZVertexSoA.h"
#include "CUDADataFormats/Track/interface/PixelTrackCUDA.h"

#include <cuda/api_wrappers.h>
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"



class ZVertexCUDA {
public:
  using SoA = ZVertexSoA;
  using TkSoA = PixelTrackCUDA::SoA;

  ZVertexCUDA(){}
  ZVertexCUDA(TkSoA const *  trackOnGPU, cuda::stream_t<> &stream);

  auto * soa() { return m_soa.get();}
  auto const * soa() const { return m_soa.get();}

  auto const * trackSoA() const { return m_trackOnGPU;}

  cudautils::host::unique_ptr<SoA> soaToHostAsync(cuda::stream_t<>& stream) const;

private:

  TkSoA const *  m_trackOnGPU = nullptr;  // fowarding
  cudautils::device::unique_ptr<SoA> m_soa;


};

#endif
