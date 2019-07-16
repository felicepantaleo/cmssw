#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include "CUDADataFormats/Vertex/interface/ZVertexSoA.h"
#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"
#include "CUDADataFormats/Track/interface/PixelTrackCUDA.h"

using ZVertexGPU = HeterogeneousSoAGPU<ZVertexSoA>;
using ZVertexCUDA = HeterogeneousSoAGPU<ZVertexSoA>;
using ZVertexHost = HeterogeneousSoAHost<ZVertexSoA>;
using ZVertexCPU = HeterogeneousSoACPU<ZVertexSoA>;

#endif
