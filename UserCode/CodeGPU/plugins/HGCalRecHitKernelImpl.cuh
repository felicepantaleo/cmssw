#ifndef _HGCALRECHITKERNELIMPL_CUH_
#define _HGCALRECHITKERNELIMPL_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDADataFormats/HGCal/interface/HGCUncalibratedRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCRecHitSoA.h"
#include "CUDADataFormats/HGCal/interface/HGCConstant.h"

__global__
void ee_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGCeeUncalibratedRecHitConstantData cdata, int length);

__global__
void hef_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGChefUncalibratedRecHitConstantData cdata, int length);

__global__
void heb_step1(HGCUncalibratedRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGChebUncalibratedRecHitConstantData cdata, int length);

__global__
void ee_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGCeeUncalibratedRecHitConstantData cdata, int length);

__global__
void hef_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGChefUncalibratedRecHitConstantData cdata, int length);

__global__
void heb_to_rechit(HGCRecHitSoA dst_soa, HGCUncalibratedRecHitSoA src_soa, HGChebUncalibratedRecHitConstantData cdata, int length);

#endif /* _HGCALRECHITKERNELIMPL_H_ */
