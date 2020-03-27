#ifndef _KERNELMANAGER_H
#define _KERNELMANAGER_H

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HGCalRecHitKernelImpl.cuh"
#include "Types.h"

#include <vector>
#include <algorithm> //std::swap  
#include <variant>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __CUDA_ARCH__
extern __constant__ uint32_t calo_rechit_masks[];
#endif

template <typename T>
class KernelConstantData {
 public:
 KernelConstantData(T& data_, HGCConstantVectorData& vdata_): data(data_), vdata(vdata_) {
    if( ! (std::is_same<T, HGCeeUncalibratedRecHitConstantData>::value or std::is_same<T, HGChefUncalibratedRecHitConstantData>::value or std::is_same<T, HGChebUncalibratedRecHitConstantData>::value ))
      {
	throw cms::Exception("WrongTemplateType") << "The KernelConstantData class does not support this type.";
      }
  }
  T data;
  HGCConstantVectorData vdata;
};

template <typename TYPE_IN, typename TYPE_OUT>
  class KernelModifiableData {
 public:
 KernelModifiableData(LENGTHSIZE nhits_, LENGTHSIZE stride_, TYPE_IN *h_in_, TYPE_IN *d_1_, TYPE_IN *d_2_, TYPE_OUT *d_out_, TYPE_OUT *h_out_):
  nhits(nhits_), stride(stride_), h_in(h_in_), d_1(d_1_), d_2(d_2_), d_out(d_out_), h_out(h_out_) {}

  LENGTHSIZE nhits;
  LENGTHSIZE stride;
  TYPE_IN *h_in;
  TYPE_IN *d_1, *d_2;
  TYPE_OUT *d_out;
  TYPE_OUT *h_out;
};

class KernelManagerHGCalRecHit {
 public:
  KernelManagerHGCalRecHit(KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA>*);
  ~KernelManagerHGCalRecHit();
  void run_kernels(const KernelConstantData<HGCeeUncalibratedRecHitConstantData>*, KernelConstantData<HGCeeUncalibratedRecHitConstantData>*);
  void run_kernels(const KernelConstantData<HGChefUncalibratedRecHitConstantData>*, KernelConstantData<HGChefUncalibratedRecHitConstantData>*);
  void run_kernels(const KernelConstantData<HGChebUncalibratedRecHitConstantData>*, KernelConstantData<HGChebUncalibratedRecHitConstantData>*);
  HGCRecHitSoA* get_output();

 private:
  void after_();
  LENGTHSIZE get_shared_memory_size_(const LENGTHSIZE&, const LENGTHSIZE&, const LENGTHSIZE&, const LENGTHSIZE&, const LENGTHSIZE&);
  void assign_and_transfer_to_device_();
  void assign_and_transfer_to_device_(const KernelConstantData<HGCeeUncalibratedRecHitConstantData>*, KernelConstantData<HGCeeUncalibratedRecHitConstantData>*);
  void assign_and_transfer_to_device_(const KernelConstantData<HGChefUncalibratedRecHitConstantData>*, KernelConstantData<HGChefUncalibratedRecHitConstantData>*);
  void assign_and_transfer_to_device_(const KernelConstantData<HGChebUncalibratedRecHitConstantData>*, KernelConstantData<HGChebUncalibratedRecHitConstantData>*);
  void transfer_to_host_and_synchronize_();
  void reuse_device_pointers_();

  LENGTHSIZE nbytes_host_;
  LENGTHSIZE nbytes_device_;
  KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA> *data_;
};

#endif //_KERNELMANAGER_H_
