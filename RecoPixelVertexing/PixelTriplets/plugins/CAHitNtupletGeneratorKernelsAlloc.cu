#include "CAHitNtupletGeneratorKernels.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

void CAHitNtupletGeneratorKernels::deallocateOnGPU() {
  if (m_params.doStats_) {
    // crash on multi-gpu processes
    printCounters();
  }
  cudaFree(counters_);

}

void CAHitNtupletGeneratorKernels::allocateOnGPU(cuda::stream_t<>& stream) {
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  edm::Service<CUDAService> cs;


  cudaCheck(cudaMalloc(&counters_, sizeof(Counters)));
  cudaCheck(cudaMemset(counters_, 0, sizeof(Counters)));

  cudaCheck(cudaMalloc(&device_nCells_, sizeof(uint32_t)));

  /* not used at the moment 
  cudaCheck(cudaMalloc(&device_theCellNeighbors_, sizeof(CAConstants::CellNeighborsVector)));
  cudaCheck(cudaMemset(device_theCellNeighbors_, 0, sizeof(CAConstants::CellNeighborsVector)));
  cudaCheck(cudaMalloc(&device_theCellTracks_, sizeof(CAConstants::CellTracksVector)));
  cudaCheck(cudaMemset(device_theCellTracks_, 0, sizeof(CAConstants::CellTracksVector)));
  */

  device_hitToTuple_ = cs->make_device_unique<HitToTuple>(stream);
  device_hitToTuple_apcHolder_ = cs->make_device_unique<AtomicPairCounter::c_type>(stream);
  device_hitToTuple_apc_ = (AtomicPairCounter*)device_hitToTuple_apcHolder_.get();

  device_tupleMultiplicity_ = cs->make_device_unique<TupleMultiplicity>(stream);

  device_tmws_ = cs->make_device_unique<uint8_t[]>(std::max(TupleMultiplicity::wsSize(), HitToTuple::wsSize()),stream);

  cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), stream.id()));
  cudautils::launchZero(device_tupleMultiplicity_.get(), stream.id());
  cudautils::launchZero(device_hitToTuple_.get(), stream.id());  // we may wish to keep it in the edm...
}

