#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksByDensity.h"
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksDBSCAN.h"
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksIterative.h"

#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


// a macro SORRY
#define LOC_ZV(M) ((char*)(gpu_d) + offsetof(ZVertices, M))
#define LOC_WS(M) ((char*)(ws_d) + offsetof(WorkSpace, M))

namespace gpuVertexFinder {

  __global__ void loadTracks(TkSoA const* ptracks, ZVertexSoA * soa, WorkSpace* pws, float ptMin) {
    
    auto const & tracks = *ptracks;
    auto const & fit = tracks.stateAtBS;
    auto const* quality = tracks.qualityData();

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TkSoA::stride())
      return;

    auto nHits = tracks.nHits(idx);
    if (nHits == 0) return;  // this is a guard: maybe we need to move to nTracks...

    // initialize soa...
    soa->idv[idx]=-1;

    if (nHits < 4)
      return;  // no triplets
    if (quality[idx] != trackQuality::loose)
      return;
 
    auto pt = tracks.pt(idx);

    if (pt < ptMin)
      return;

    auto& data = *pws;
    auto it = atomicAdd(&data.ntrks, 1);
    data.itrk[it] = idx;
    data.zt[it] = tracks.zip(idx);
    data.ezt2[it] = fit.covariance(idx)(14);
    data.ptt2[it] = pt*pt;
  }

  ZVertexCUDA Producer::makeAsync(cuda::stream_t<>& stream, TkSoA const * tksoa, float ptMin) const {
    assert(tksoa);
    ZVertexCUDA vertices(stream);

    auto * soa = vertices.get();

    edm::Service<CUDAService> cs;
    auto ws_d = cs->make_device_unique<WorkSpace>(stream);
   
    init<<<1, 1, 0, stream.id()>>>(soa, ws_d.get());
    auto blockSize = 128;
    auto numberOfBlocks = (TkSoA::stride() + blockSize - 1) / blockSize;
    loadTracks<<<numberOfBlocks, blockSize, 0, stream.id()>>>(tksoa, soa, ws_d.get(), ptMin);
    cudaCheck(cudaGetLastError());
    if (useDensity_) {
      clusterTracksByDensity<<<1, 1024 - 256, 0, stream.id()>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
    } else if (useDBSCAN_) {
      clusterTracksDBSCAN<<<1, 1024 - 256, 0, stream.id()>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
    } else if (useIterative_) {
      clusterTracksIterative<<<1, 1024 - 256, 0, stream.id()>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
    }
    cudaCheck(cudaGetLastError());
    fitVertices<<<1, 1024 - 256, 0, stream.id()>>>(soa, ws_d.get(), 50.);
    cudaCheck(cudaGetLastError());

    splitVertices<<<1024, 128, 0, stream.id()>>>(soa, ws_d.get(), 9.f);
    cudaCheck(cudaGetLastError());
    fitVertices<<<1, 1024 - 256, 0, stream.id()>>>(soa, ws_d.get(), 5000.);
    cudaCheck(cudaGetLastError());

    sortByPt2<<<1, 256, 0, stream.id()>>>(soa, ws_d.get());
    cudaCheck(cudaGetLastError());

    return vertices;
  }

}  // namespace gpuVertexFinder

#undef FROM
