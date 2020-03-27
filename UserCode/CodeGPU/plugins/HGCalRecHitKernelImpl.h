#ifndef HGCalRecHitKernelImpl_cuh
#define HGCalRecHitKernelImpl_cuh

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "Utils.h"

#include <algorithm> //std::swap
#include <cuda_runtime.h>

class KernelManagerBase  {
public:
  explicit KernelManagerBase(){};
  virtual ~KernelManagerBase(){};
  
  virtual void run_kernels() = 0;

protected:
  virtual void assign_and_transfer_to_device() = 0;
  virtual void transfer_to_host_and_synchronize() = 0;
  virtual void reuse_device_pointers() = 0;

  dim3 dimGrid;
  dim3 dimBlock;
};

//the class assumes that the sizes of the arrays pointed to and the size of the collection are constant
class KernelManagerHGCalRecHit: private KernelManagerBase {
public:
  explicit KernelManagerHGCalRecHit(const edm::SortedCollection<HGCUncalibratedRecHitCollection>&, const detectortype&);
  ~KernelManagerHGCalRecHit();
  void run_kernels();
  const edm::SortedCollection<HGCUncalibratedRecHitCollection>& get_new_collection();

private:
  void ee_step1_wrapper();
  void hef_step1_wrapper();
  void heb_step1_wrapper();
  void assign_and_transfer_to_device() override;
  void transfer_to_host_and_synchronize() override;
  void reuse_device_pointers() override;

  size_t shits_, sbytes_;
  const detectortype dtype_;
  edm::SortedCollection<HGCUncalibratedRecHit> oldhits_collection_;
  HGCUncalibratedRecHit *h_oldhits_, *h_newhits_; //host pointers
  HGCUncalibratedRecHit *d_oldhits_, *d_newhits_; //device pointers
};

__global__
void ee_step1(HGCUncalibratedRecHit *, HGCUncalibratedRecHit *, const size_t& length);
__global__
void hef_step1(HGCUncalibratedRecHit *, HGCUncalibratedRecHit *, const size_t& length);
__global__
void heb_step1(HGCUncalibratedRecHit *, HGCUncalibratedRecHit *, const size_t& length);


#endif //HGCalRecHitKernelImpl_cuh
