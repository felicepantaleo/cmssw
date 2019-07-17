#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include "CUDADataFormats/Vertex/interface/ZVertexSoA.h"
#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "CUDADataFormats/Track/interface/PixelTrackCUDA.h"


using ZVertexHeterogeneous = HeterogeneousSoA<ZVertexSoA>;
#ifndef __CUDACC__
#include "CUDADataFormats/Common/interface/CUDAProduct.h"
using ZVertexCUDAProduct =  CUDAProduct<ZVertexHeterogeneous>;
#endif

#endif
