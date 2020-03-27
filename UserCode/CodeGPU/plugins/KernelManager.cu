#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "KernelManager.h"
#include "HGCalRecHitKernelImpl.cuh"

dim3 nblocks_;
constexpr dim3 nthreads_(256); //some kernels will potentially not allocate shared memory properly with a lower number

KernelManagerHGCalRecHit::KernelManagerHGCalRecHit(KernelModifiableData<HGCUncalibratedRecHitSoA, HGCRecHitSoA> *data):
  data_(data)
{
  nblocks_ = (data_->nhits + nthreads_.x - 1) / nthreads_.x;
  nbytes_host_ = (data_->h_out)->nbytes * data_->stride;
  nbytes_device_ = (data_->d_1)->nbytes * data_->stride;
}

KernelManagerHGCalRecHit::~KernelManagerHGCalRecHit()
{
}

void KernelManagerHGCalRecHit::assign_and_transfer_to_device_()
{
  cudaCheck( cudaMemcpyAsync((data_->d_1)->amplitude, (data_->h_in)->amplitude, nbytes_device_, cudaMemcpyHostToDevice) );
  after_();
}

void KernelManagerHGCalRecHit::assign_and_transfer_to_device_(const KernelConstantData<HGCeeUncalibratedRecHitConstantData> *h_kcdata, KernelConstantData<HGCeeUncalibratedRecHitConstantData> *d_kcdata)
{
  cudaCheck( cudaMemcpyAsync( d_kcdata->data.hgcEE_fCPerMIP_, h_kcdata->data.hgcEE_fCPerMIP_, h_kcdata->data.nbytes, cudaMemcpyHostToDevice) );
  after_();
}

void KernelManagerHGCalRecHit::assign_and_transfer_to_device_(const KernelConstantData<HGChefUncalibratedRecHitConstantData> *h_kcdata, KernelConstantData<HGChefUncalibratedRecHitConstantData> *d_kcdata)
{
  cudaCheck( cudaMemcpyAsync( d_kcdata->data.hgcHEF_fCPerMIP_, h_kcdata->data.hgcHEF_fCPerMIP_, h_kcdata->data.nbytes, cudaMemcpyHostToDevice) );
  after_();
}

void KernelManagerHGCalRecHit::assign_and_transfer_to_device_(const KernelConstantData<HGChebUncalibratedRecHitConstantData> *h_kcdata, KernelConstantData<HGChebUncalibratedRecHitConstantData> *d_kcdata)
{
  cudaCheck( cudaMemcpyAsync( d_kcdata->data.weights_, h_kcdata->data.weights_, h_kcdata->data.nbytes, cudaMemcpyHostToDevice) );
  after_();
}

void KernelManagerHGCalRecHit::transfer_to_host_and_synchronize_()
{
  cudaCheck( cudaMemcpyAsync((data_->h_out)->energy, (data_->d_out)->energy, nbytes_host_, cudaMemcpyDeviceToHost) );
  after_();
}

void KernelManagerHGCalRecHit::reuse_device_pointers_()
{
  std::swap(data_->d_1, data_->d_2); 
  after_();
}

int KernelManagerHGCalRecHit::get_shared_memory_size_(const int& nd, const int& nf, const int& nu, const int& ni, const int& nb) {
  int dmem = nd*sizeof(double);
  int fmem = nf*sizeof(float);
  int umem = nu*sizeof(uint32_t);
  int imem = ni*sizeof(int);
  int bmem = nb*sizeof(bool);
  return dmem + fmem + umem + imem + bmem;
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGCeeUncalibratedRecHitConstantData> *h_kcdata, KernelConstantData<HGCeeUncalibratedRecHitConstantData> *d_kcdata)
{
  assign_and_transfer_to_device_(h_kcdata, d_kcdata);
  assign_and_transfer_to_device_();

  printf("%d blocks being launched with %d threads (%d in total) for %d ee hits.\n", nblocks_.x, nthreads_.x, nblocks_.x*nthreads_.x, data_->nhits);
  int nbytes_shared = get_shared_memory_size_(h_kcdata->data.ndelem, h_kcdata->data.nfelem, h_kcdata->data.nuelem, h_kcdata->data.nielem, h_kcdata->data.nbelem);

  /*
  ee_step1<<<nblocks_, nthreads_>>>( *(data_->d_2), *(data_->d_1), d_kcdata->data, data_->nhits );
  after_();
  reuse_device_pointers_();
  */

  ee_to_rechit<<<nblocks_, nthreads_, nbytes_shared>>>( *(data_->d_out), *(data_->d_1), d_kcdata->data, data_->nhits );
  after_();

  transfer_to_host_and_synchronize_();
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGChefUncalibratedRecHitConstantData> *h_kcdata, KernelConstantData<HGChefUncalibratedRecHitConstantData> *d_kcdata)
{
  assign_and_transfer_to_device_(h_kcdata, d_kcdata);
  assign_and_transfer_to_device_();

  printf("%d blocks being launched with %d threads (%d in total) for %d hef hits.\n", nblocks_.x, nthreads_.x, nblocks_.x*nthreads_.x, data_->nhits);
  int nbytes_shared = get_shared_memory_size_(h_kcdata->data.ndelem, h_kcdata->data.nfelem, h_kcdata->data.nuelem, h_kcdata->data.nielem, h_kcdata->data.nbelem);

  /*
  hef_step1<<<nblocks_,nthreads_>>>( *(data_->d_2), *(data_->d_1), d_kcdata->data, data_->nhits);
  after_();
  reuse_device_pointers_();
  */

  hef_to_rechit<<<nblocks_,nthreads_, nbytes_shared>>>( *(data_->d_out), *(data_->d_1), d_kcdata->data, data_->nhits );
  after_();
  transfer_to_host_and_synchronize_();
}

void KernelManagerHGCalRecHit::run_kernels(const KernelConstantData<HGChebUncalibratedRecHitConstantData> *h_kcdata, KernelConstantData<HGChebUncalibratedRecHitConstantData> *d_kcdata)
{
  assign_and_transfer_to_device_(h_kcdata, d_kcdata);
  assign_and_transfer_to_device_();

  printf("%d blocks being launched with %d threads (%d in total) for %d heb hits.\n", nblocks_.x, nthreads_.x, nblocks_.x*nthreads_.x, data_->nhits);
  int nbytes_shared = get_shared_memory_size_(h_kcdata->data.ndelem, h_kcdata->data.nfelem, h_kcdata->data.nuelem, h_kcdata->data.nielem, h_kcdata->data.nbelem);

  /*
  heb_step1<<<nblocks_, nthreads_>>>( *(data_->d_2), *(data_->d_1), d_kcdata->data, data_->nhits);
  after_();
  reuse_device_pointers_();
  */

  heb_to_rechit<<<nblocks_, nthreads_, nbytes_shared>>>( *(data_->d_out), *(data_->d_1), d_kcdata->data, data_->nhits );
  after_();

  transfer_to_host_and_synchronize_();
}

void KernelManagerHGCalRecHit::after_() {
  cudaCheck( cudaDeviceSynchronize() );
  cudaCheck( cudaGetLastError() );
}

HGCRecHitSoA* KernelManagerHGCalRecHit::get_output()
{
  return data_->h_out;
}
