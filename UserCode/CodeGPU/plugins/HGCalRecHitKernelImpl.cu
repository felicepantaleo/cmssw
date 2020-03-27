#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "HGCalRecHitKernelImpl.cuh"

__device__
int wafer(uint32_t id)
{
  static const int kHGCalWaferOffset = 8;
  static const int kHGCalWaferMask = 0x3FF;
  return (id >> kHGCalWaferOffset) & kHGCalWaferMask; 
}

__device__
int layer(uint32_t id, unsigned int offset)
{
  static const int kHGCalLayerOffset = 20;
  static const int kHGCalLayerMask = 0x1F;
  int layer = (id >> kHGCalLayerOffset) & kHGCalLayerMask; 
  return layer + offset;
}

__device__ 
double get_weight_from_layer(const int& padding, const int& layer, double*& sd)
{
  return sd[padding + layer];
}

__device__
void make_rechit(unsigned int tid, HGCRecHitSoA& dst_soa, HGCUncalibratedRecHitSoA& src_soa, const bool &heb_flag, 
		 const double &weight, const double &rcorr, const double &cce_correction, const double &sigmaNoiseGeV, float *& sf)
{
  dst_soa.id[tid] = src_soa.id[tid];
  dst_soa.energy[tid] = src_soa.amplitude[tid] * weight * 0.001f;
  if(!heb_flag)
    dst_soa.energy[tid] *=  __fdividef(rcorr, cce_correction);
  dst_soa.time[tid] = src_soa.jitter[tid];
  dst_soa.flagBits[tid] |= (0x1 << HGCRecHit::kGood);
  float son = __fdividef( dst_soa.energy[tid], sigmaNoiseGeV);
  float son_norm = fminf(32.f, son) / 32.f * ((1 << 8)-1);
  long int son_round = lroundf( son_norm );
  dst_soa.son[tid] = static_cast<uint8_t>( son_round );

  if(heb_flag==0)
    {
      //get time resolution
      float max = fmaxf(son, sf[0]); //this max trick avoids if...elseif...else condition
      float aterm = sf[2];
      float cterm = sf[3];
      dst_soa.timeError[tid] = sqrt( __fdividef(aterm,max)*__fdividef(aterm,max) + cterm*cterm );
    }
  else
    dst_soa.timeError[tid] = -1;
}

__device__ 
double get_thickness_correction(const int& padding, double *& sd, const HGCeeUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL];
}
__device__ 
double get_thickness_correction(const int& padding, double *& sd, const HGChefUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL];
}

__device__
double get_noise(const int& padding, double *& sd, const HGCeeUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}
__device__
double get_noise(const int& padding, double *& sd, const HGChefUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}

__device__
double get_cce_correction(const int& padding, double *& sd, const HGCeeUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}
__device__
double get_cce_correction(const int& padding, double *& sd, const HGChefUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained fro the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}

__device__ 
double get_fCPerMIP(const int& padding, double *& sd, const HGCeeUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained from the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}
__device__ 
double get_fCPerMIP(const int& padding, double *& sd, const HGChefUncalibratedRecHitConstantData& cdata)
{
  int waferTypeL = cdata.waferTypeL_[1]; //this should be obtained from the DetId and the wafer() __device__ function.
  return sd[padding + waferTypeL - 1];
}

__device__ 
void set_shared_memory(int tid, double*& sd, float*& sf, uint32_t*& su, int*& si, bool*& sb, const HGCeeUncalibratedRecHitConstantData& cdata, const int& size1, const int& size2, const int& size3, const int& size4, const int& size5, const int& size6)
{
  const int initial_pad = 2;
  if(tid == 0)
    sd[tid] = cdata.hgcEE_keV2DIGI_;
  else if(tid == 1)
    sd[tid] = cdata.hgceeUncalib2GeV_;
  else if(tid >= initial_pad && tid < size1)
    sd[tid] = cdata.hgcEE_fCPerMIP_[tid-2];
  else if(tid >= size1 && tid < size2)
    sd[tid] = cdata.hgcEE_cce_[tid-size1];
  else if(tid >= size2 && tid < size3)
    sd[tid] = cdata.hgcEE_noise_fC_[tid-size2];
  else if(tid >= size3 && tid < size4)
    sd[tid] = cdata.rcorr_[tid - size3];
  else if(tid >= size4 && tid < size5)
    sd[tid] = cdata.weights_[tid - size4];
  else if(tid >= size5 && tid < size6)
    si[tid - size5] = cdata.waferTypeL_[tid - size5];
  else if(tid == size6)
    sf[0] = (cdata.xmin_ > 0) ? cdata.xmin_ : 0.1;
  else if(tid == size6 + 1)
    sf[1] = cdata.xmax_;
  else if(tid == size6 + 2)
    sf[2] = cdata.aterm_;
  else if(tid == size6 + 3)
    sf[3] = cdata.cterm_;
  else if(tid == size6 + 4)
    su[0] = cdata.rangeMatch_;
  else if(tid == size6 + 5)
    su[1] = cdata.rangeMask_;
  else if(tid == size6 + 6)
    sb[0] = cdata.hgcEE_isSiFE_;

  __syncthreads();
}

__device__ 
void set_shared_memory(int tid, double*& sd, float*& sf, uint32_t*& su, int*& si, bool*& sb, const HGChefUncalibratedRecHitConstantData& cdata, const int& size1, const int& size2, const int& size3, const int& size4, const int& size5, const int& size6)
{
  const int initial_pad = 2;
  if(tid == 0)
    sd[tid] = cdata.hgcHEF_keV2DIGI_;
  else if(tid == 1)
    sd[tid] = cdata.hgchefUncalib2GeV_;
  else if(tid >= initial_pad && tid < size1)
    sd[tid] = cdata.hgcHEF_fCPerMIP_[tid-initial_pad];
  else if(tid >= size1 && tid < size2)
    sd[tid] = cdata.hgcHEF_cce_[tid-size1];
  else if(tid >= size2 && tid < size3)
    sd[tid] = cdata.hgcHEF_noise_fC_[tid-size2];  
  else if(tid >= size3 && tid < size4)
    sd[tid] = cdata.rcorr_[tid - size3];
  else if(tid >= size4 && tid < size5)
    sd[tid] = cdata.weights_[tid - size4];
  else if(tid >= size5 && tid < size6)
    si[tid - size5] = cdata.waferTypeL_[tid - size5];
  else if(tid == size6)
    sf[0] = (cdata.xmin_ > 0) ? cdata.xmin_ : 0.1;
  else if(tid == size6 + 1)
    sf[1] = cdata.xmax_;
  else if(tid == size6 + 2)
    sf[2] = cdata.aterm_;
  else if(tid == size6 + 3)
    sf[3] = cdata.cterm_;
  else if(tid == size6 + 4)
    su[0] = cdata.rangeMatch_;
  else if(tid == size6 + 5)
    su[1] = cdata.rangeMask_;
  /*
  else if(tid == size6 + 6)
    su[2] = cdata.fhOffset_;
  */
  else if(tid == size6 + 6) //CHANGE
    sb[0] = cdata.hgcHEF_isSiFE_;

  __syncthreads();
}

__device__ 
void set_shared_memory(int tid, double*& sd, float*& sf, uint32_t*& su, bool*& sb, const HGChebUncalibratedRecHitConstantData& cdata, const int& size1)
{
  const int initial_pad = 3;
  if(tid == 0)
    sd[tid] = cdata.hgcHEB_keV2DIGI_;
  else if(tid == 1)
    sd[tid] = cdata.hgchebUncalib2GeV_;
  else if(tid == 2)
    sd[tid] = cdata.hgcHEB_noise_MIP_;
  else if(tid >= initial_pad && tid < size1)
    sd[tid] = cdata.weights_[tid - initial_pad];
  else if(tid == size1)
    su[0] = cdata.rangeMatch_;
  else if(tid == size1 + 1)
    su[1] = cdata.rangeMask_;
  else if(tid == size1 + 2)
    su[2] = cdata.fhOffset_;
  else if(tid == size1 + 3)
    sb[0] = cdata.hgcHEB_isSiFE_;

  __syncthreads();
}

__global__
void ee_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGCeeUncalibratedRecHitConstantData cdata, int length)
{
  /*
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
      dst_soa.amplitude[i] = src_soa.amplitude[i];
    }
  */
}

__global__
void hef_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChefUncalibratedRecHitConstantData cdata, int length)
{
}

__global__
void heb_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChebUncalibratedRecHitConstantData cdata, int length)
{
}

__global__
void ee_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGCeeUncalibratedRecHitConstantData cdata, int length)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  int size1 = cdata.s_hgcEE_fCPerMIP_ + 2;
  int size2 = cdata.s_hgcEE_cce_      + size1;
  int size3 = cdata.s_hgcEE_noise_fC_ + size2;
  int size4 = cdata.s_rcorr_          + size3; 
  int size5 = cdata.s_weights_        + size4; 
  int size6 = cdata.s_waferTypeL_     + size5; 

  extern __shared__ double s[];
  double   *sd = s;
  float    *sf = (float*)   (sd + cdata.ndelem);
  uint32_t *su = (uint32_t*)(sf + cdata.nfelem);
  int      *si = (int*)     (su + cdata.nuelem);
  bool     *sb = (bool*)    (si + cdata.nielem);
  set_shared_memory(threadIdx.x, sd, sf, su, si, sb, cdata, size1, size2, size3, size4, size5, size6);

  for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
      double l = layer(src_soa.id[tid], 0); //no offset
      double weight = get_weight_from_layer(size4, l, sd);
      double rcorr = get_thickness_correction(size3, sd, cdata);
      double noise = get_noise(size2, sd, cdata);
      double cce_correction = get_cce_correction(size1, sd, cdata);
      double fCPerMIP = get_fCPerMIP(2, sd, cdata);
      double sigmaNoiseGeV = 1e-3 * weight * rcorr * __fdividef( noise, fCPerMIP );
      make_rechit(i, dst_soa, src_soa, false, weight, rcorr, cce_correction, sigmaNoiseGeV, sf);
    }
}

__global__
void hef_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChefUncalibratedRecHitConstantData cdata, int length)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  int size1 = cdata.s_hgcHEF_fCPerMIP_ + 2;
  int size2 = cdata.s_hgcHEF_cce_      + size1;
  int size3 = cdata.s_hgcHEF_noise_fC_ + size2;
  int size4 = cdata.s_rcorr_           + size3; 
  int size5 = cdata.s_weights_         + size4; 
  int size6 = cdata.s_waferTypeL_      + size5;

  extern __shared__ double s[];
  double   *sd = s;
  float    *sf = (float*)   (sd + cdata.ndelem);
  uint32_t *su = (uint32_t*)(sf + cdata.nfelem);
  int      *si = (int*)     (su + cdata.nuelem);
  bool     *sb = (bool*)    (si + cdata.nielem);

  set_shared_memory(threadIdx.x, sd, sf, su, si, sb, cdata, size1, size2, size3, size4, size5, size6);

  for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
      double l = layer(src_soa.id[tid], 0); //no offset
      double weight = get_weight_from_layer(size4, l, sd);
      double rcorr = get_thickness_correction(size3, sd, cdata);
      double noise = get_noise(size2, sd, cdata);
      double cce_correction = get_cce_correction(size1, sd, cdata);
      double fCPerMIP = get_fCPerMIP(2, sd, cdata);
      double sigmaNoiseGeV = 1e-3 * weight * rcorr * __fdividef( noise,  fCPerMIP );
      printf("before %u,%u,%u,%u \n", dst_soa.son[110], dst_soa.son[13], dst_soa.son[650], dst_soa.son[114]);
      make_rechit(i, dst_soa, src_soa, false, weight, rcorr, cce_correction, sigmaNoiseGeV, sf);
      printf("after %u,%u,%u,%u \n", dst_soa.son[110], dst_soa.son[13], dst_soa.son[650], dst_soa.son[114]);
    }
}

__global__
void heb_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, const HGChebUncalibratedRecHitConstantData cdata, int length)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  int size1 = cdata.s_weights_ + 3; 

  extern __shared__ double s[];
  double   *sd = s;
  float    *sf = (float*)   (sd + cdata.ndelem);
  uint32_t *su = (uint32_t*)(sf + cdata.nfelem);
  bool     *sb = (bool*)    (su + cdata.nielem);
  set_shared_memory(threadIdx.x, sd, sf, su, sb, cdata, size1);

  for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x)
    {
      double l = layer(src_soa.id[tid], su[2]);
      double weight = get_weight_from_layer(3, l, sd);
      double noise = sd[2];
      double sigmaNoiseGeV = 1e-3 * noise * weight;
      make_rechit(i, dst_soa, src_soa, true, weight, 0., 0., sigmaNoiseGeV, sf);
    }
}

/*
=======
>>>>>>> b5bfc7e2f47f926abb3dcd21cdf5e2094e53dd3f
//declared as extern in DataFormats/CaloRecHit/interface/CaloRecHit.h
#ifdef __CUDA_ARCH__
__constant__ uint32_t calo_rechit_masks[] = {0x00000000u, 0x00000001u, 0x00000003u, 0x00000007u, 0x0000000fu, 0x0000001fu,
					     0x0000003fu, 0x0000007fu, 0x000000ffu, 0x000001ffu, 0x000003ffu, 0x000007ffu,
					     0x00000fffu, 0x00001fffu, 0x00003fffu, 0x00007fffu, 0x0000ffffu, 0x0001ffffu,
					     0x0003ffffu, 0x0007ffffu, 0x000fffffu, 0x001fffffu, 0x003fffffu, 0x007fffffu,
					     0x00ffffffu, 0x01ffffffu, 0x03ffffffu, 0x07ffffffu, 0x0fffffffu, 0x1fffffffu,
					     0x3fffffffu, 0x7fffffffu, 0xffffffffu};
#endif
<<<<<<< HEAD
=======
<<<<<<< HEAD
*/
