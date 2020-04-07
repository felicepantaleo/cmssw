#ifndef _HETEROGENEOUSHGCALPRODUCERMEMORYWRAPPER_H_
#define _HETEROGENEOUSHGCALPRODUCERMEMORYWRAPPER_H_

#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>
#include <type_traits>
#include <numeric>
#include <cuda_runtime.h>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCUncalibratedRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCConstant.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/ContextState.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "KernelManager.h"

namespace memory {
  namespace allocation {
    /*
    namespace {
      std::tuple<LENGTHSIZE, LENGTHSIZE, LENGTHSIZE, LENGTHSIZE> get_memory_sizes_(const std::vector<LENGTHSIZE>&, const LENGTHSIZE&, const LENGTHSIZE&, const LENGTHSIZE&);
    }
    */
    void host(KernelConstantData<HGCeeUncalibratedRecHitConstantData>*, cms::cuda::host::noncached::unique_ptr<double[]>&);
    void host(KernelConstantData<HGChefUncalibratedRecHitConstantData>*, cms::cuda::host::noncached::unique_ptr<double[]>&);
    void host(KernelConstantData<HGChebUncalibratedRecHitConstantData>*, cms::cuda::host::noncached::unique_ptr<double[]>&);
    void host(const int&, HGCUncalibratedRecHitSoA*, cms::cuda::host::noncached::unique_ptr<float[]>&);
    void host(const int&, HGCRecHitSoA*, cms::cuda::host::unique_ptr<float[]>&);
    void device(KernelConstantData<HGCeeUncalibratedRecHitConstantData>*, cms::cuda::device::unique_ptr<double[]>&);
    void device(KernelConstantData<HGChefUncalibratedRecHitConstantData>*, cms::cuda::device::unique_ptr<double[]>&);
    void device(KernelConstantData<HGChebUncalibratedRecHitConstantData>*, cms::cuda::device::unique_ptr<double[]>&);
    void device(const int&, HGCUncalibratedRecHitSoA*, HGCUncalibratedRecHitSoA*, HGCRecHitSoA*, cms::cuda::device::unique_ptr<float[]>&);
  }
}
							
#endif // _HETEROGENEOUSHGCALPRODUCERMEMORYWRAPPER_H_
