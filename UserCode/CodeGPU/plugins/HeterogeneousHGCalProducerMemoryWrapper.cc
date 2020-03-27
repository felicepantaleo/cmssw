#include "HeterogeneousHGCalProducerMemoryWrapper.h"
#include "Types.h"

namespace memory {
  namespace allocation {    
    namespace {
      //returns total number of bytes, number of 'double' elements and number of 'float' elements
      std::tuple<int, int, int, int> get_memory_sizes_(const std::vector<int>& fixed_sizes, const int& ndoubles, const int& nfloats, const int& nints)
      {
	const int size1 = sizeof(double);
	const int size2 = sizeof(float);
	const int size3 = sizeof(int);
	int nelements1_tot = std::accumulate( fixed_sizes.begin(), fixed_sizes.begin() + ndoubles, 0);
	int nelements2_tot = std::accumulate( fixed_sizes.begin() + ndoubles, fixed_sizes.begin() + ndoubles + nfloats, 0);
	int nelements3_tot = std::accumulate( fixed_sizes.begin() + ndoubles + nfloats, fixed_sizes.end(), 0);
	assert( fixed_sizes.begin() + ndoubles + nfloats + nints == fixed_sizes.end() );
	int size_tot = nelements1_tot*size1+nelements2_tot*size2+nelements3_tot*size3;
	return std::make_tuple(size_tot, nelements1_tot, nelements2_tot, nelements3_tot);
      }
    }

    void device(KernelConstantData<HGCeeUncalibratedRecHitConstantData> *kcdata, cms::cuda::device::unique_ptr<double[]>& mem) {
      const std::vector<int> nelements = {kcdata->data.s_hgcEE_fCPerMIP_, kcdata->data.s_hgcEE_cce_, kcdata->data.s_hgcEE_noise_fC_, kcdata->data.s_rcorr_, kcdata->data.s_weights_, kcdata->data.s_waferTypeL_};
      auto memsizes = get_memory_sizes_(nelements, 5, 0, 1);
      mem = cms::cuda::make_device_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data.hgcEE_fCPerMIP_ = mem.get();
      kcdata->data.hgcEE_cce_      = kcdata->data.hgcEE_fCPerMIP_ + nelements[0];
      kcdata->data.hgcEE_noise_fC_ = kcdata->data.hgcEE_cce_ + nelements[1];
      kcdata->data.rcorr_          = kcdata->data.hgcEE_noise_fC_ + nelements[2];
      kcdata->data.weights_        = kcdata->data.rcorr_ + nelements[3];
      kcdata->data.waferTypeL_     = reinterpret_cast<int*>(kcdata->data.weights_ + nelements[4]);
      kcdata->data.nbytes = std::get<0>(memsizes);
      kcdata->data.ndelem = std::get<1>(memsizes) + 2;
      kcdata->data.nfelem = std::get<2>(memsizes) + 4;
      kcdata->data.nielem = std::get<3>(memsizes) + 0;
      kcdata->data.nuelem = 2;
      kcdata->data.nbelem = 1;
    }

    void device(KernelConstantData<HGChefUncalibratedRecHitConstantData> *kcdata, cms::cuda::device::unique_ptr<double[]>& mem) {
      const std::vector<int> nelements = {kcdata->data.s_hgcHEF_fCPerMIP_, kcdata->data.s_hgcHEF_cce_, kcdata->data.s_hgcHEF_noise_fC_, kcdata->data.s_rcorr_, kcdata->data.s_weights_, kcdata->data.s_waferTypeL_};
      auto memsizes = get_memory_sizes_(nelements, 5, 0, 1);
      mem = cms::cuda::make_device_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data.hgcHEF_fCPerMIP_ = mem.get();
      kcdata->data.hgcHEF_cce_      = kcdata->data.hgcHEF_fCPerMIP_ + nelements[0];
      kcdata->data.hgcHEF_noise_fC_ = kcdata->data.hgcHEF_cce_ + nelements[1];
      kcdata->data.rcorr_           = kcdata->data.hgcHEF_noise_fC_ + nelements[2];
      kcdata->data.weights_         = kcdata->data.rcorr_ + nelements[3];
      kcdata->data.waferTypeL_      = reinterpret_cast<int*>(kcdata->data.weights_ + nelements[4]);
      kcdata->data.nbytes = std::get<0>(memsizes);
      kcdata->data.ndelem = std::get<1>(memsizes) + 2;
      kcdata->data.nfelem = std::get<2>(memsizes) + 4;
      kcdata->data.nielem = std::get<3>(memsizes) + 0;
      kcdata->data.nuelem = 3;
      kcdata->data.nbelem = 1;
    }

    void device(KernelConstantData<HGChebUncalibratedRecHitConstantData> *kcdata, cms::cuda::device::unique_ptr<double[]>& mem) {
      const std::vector<int> nelements = {kcdata->data.s_weights_};
      auto memsizes = get_memory_sizes_(nelements, 1, 0, 0);

      mem = cms::cuda::make_device_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data.weights_  = mem.get();
      kcdata->data.nbytes = std::get<0>(memsizes);
      kcdata->data.ndelem = std::get<1>(memsizes) + 3;
      kcdata->data.nfelem = std::get<2>(memsizes) + 0;
      kcdata->data.nielem = std::get<3>(memsizes) + 0;
      kcdata->data.nuelem = 3;
      kcdata->data.nbelem = 1;
    }

    void device(const int& nhits, HGCUncalibratedRecHitSoA* soa1, HGCUncalibratedRecHitSoA* soa2, HGCRecHitSoA* soa3, cms::cuda::device::unique_ptr<float[]>& mem)
    {
      std::vector<int> sizes = {6*sizeof(float), 3*sizeof(uint32_t),                     //soa1
				6*sizeof(float), 3*sizeof(uint32_t),                     //soa2
				3*sizeof(float), 2*sizeof(uint32_t), 1*sizeof(uint8_t)}; //soa3
      int size_tot = std::accumulate( sizes.begin(), sizes.end(), 0);
      mem = cms::cuda::make_device_unique<float[]>(nhits * size_tot, 0);

      soa1->amplitude     = mem.get();
      soa1->pedestal      = soa1->amplitude    + nhits;
      soa1->jitter        = soa1->pedestal     + nhits;
      soa1->chi2          = soa1->jitter       + nhits;
      soa1->OOTamplitude  = soa1->chi2         + nhits;
      soa1->OOTchi2       = soa1->OOTamplitude + nhits;
      soa1->flags         = reinterpret_cast<uint32_t*>(soa1->OOTchi2 + nhits);
      soa1->aux           = soa1->flags        + nhits;
      soa1->id            = soa1->aux          + nhits;

      soa2->amplitude     = reinterpret_cast<float*>(soa1->id + nhits);
      soa2->pedestal      = soa2->amplitude    + nhits;
      soa2->jitter        = soa2->pedestal     + nhits;
      soa2->chi2          = soa2->jitter       + nhits;
      soa2->OOTamplitude  = soa2->chi2         + nhits;
      soa2->OOTchi2       = soa2->OOTamplitude + nhits;
      soa2->flags         = reinterpret_cast<uint32_t*>(soa2->OOTchi2 + nhits);
      soa2->aux           = soa2->flags        + nhits;
      soa2->id            = soa2->aux          + nhits;
  
      soa3->energy        = reinterpret_cast<float*>(soa2->id + nhits);
      soa3->time          = soa3->energy       + nhits;
      soa3->timeError     = soa3->time         + nhits;
      soa3->id            = reinterpret_cast<uint32_t*>(soa3->timeError + nhits);
      soa3->flagBits      = soa3->id           + nhits;
      soa3->son           = reinterpret_cast<uint8_t*>(soa3->flagBits + nhits);

      soa1->nbytes = std::accumulate(sizes.begin(), sizes.begin()+2, 0);
      soa2->nbytes = std::accumulate(sizes.begin()+2, sizes.begin()+4, 0);
      soa3->nbytes = std::accumulate(sizes.begin()+4, sizes.end(), 0);
    }

    void host(KernelConstantData<HGCeeUncalibratedRecHitConstantData>* kcdata, cms::cuda::host::noncached::unique_ptr<double[]>& mem)
    {
      const std::vector<int> nelements = {kcdata->data.s_hgcEE_fCPerMIP_, kcdata->data.s_hgcEE_cce_, kcdata->data.s_hgcEE_noise_fC_, kcdata->data.s_rcorr_, kcdata->data.s_weights_, kcdata->data.s_waferTypeL_};
      auto memsizes = get_memory_sizes_(nelements, 5, 0, 1);
      mem = cms::cuda::make_host_noncached_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data.hgcEE_fCPerMIP_ = mem.get();
      kcdata->data.hgcEE_cce_      = kcdata->data.hgcEE_fCPerMIP_ + nelements[0];
      kcdata->data.hgcEE_noise_fC_ = kcdata->data.hgcEE_cce_ + nelements[1];
      kcdata->data.rcorr_          = kcdata->data.hgcEE_noise_fC_ + nelements[2];
      kcdata->data.weights_        = kcdata->data.rcorr_ + nelements[3];
      kcdata->data.waferTypeL_     = reinterpret_cast<int*>(kcdata->data.weights_ + nelements[4]);
      kcdata->data.nbytes = std::get<0>(memsizes);
      kcdata->data.ndelem = std::get<1>(memsizes) + 2;
      kcdata->data.nfelem = std::get<2>(memsizes) + 0;
      kcdata->data.nielem = std::get<3>(memsizes) + 0;
      kcdata->data.nuelem = 2;
      kcdata->data.nbelem = 1;
    }

    void host(KernelConstantData<HGChefUncalibratedRecHitConstantData>* kcdata, cms::cuda::host::noncached::unique_ptr<double[]>& mem)
    {
      const std::vector<int> nelements = {kcdata->data.s_hgcHEF_fCPerMIP_, kcdata->data.s_hgcHEF_cce_, kcdata->data.s_hgcHEF_noise_fC_, kcdata->data.s_rcorr_, kcdata->data.s_weights_, kcdata->data.s_waferTypeL_};
      auto memsizes = get_memory_sizes_(nelements, 5, 0, 1);
      mem = cms::cuda::make_host_noncached_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data.hgcHEF_fCPerMIP_ = mem.get();
      kcdata->data.hgcHEF_cce_      = kcdata->data.hgcHEF_fCPerMIP_ + nelements[0];
      kcdata->data.hgcHEF_noise_fC_ = kcdata->data.hgcHEF_cce_ + nelements[1];
      kcdata->data.rcorr_           = kcdata->data.hgcHEF_noise_fC_ + nelements[2];
      kcdata->data.weights_         = kcdata->data.rcorr_ + nelements[3];
      kcdata->data.waferTypeL_      = reinterpret_cast<int*>(kcdata->data.weights_ + nelements[4]);
      kcdata->data.nbytes = std::get<0>(memsizes);
      kcdata->data.ndelem = std::get<1>(memsizes) + 2;
      kcdata->data.nfelem = std::get<2>(memsizes) + 0;
      kcdata->data.nielem = std::get<3>(memsizes) + 0;
      kcdata->data.nuelem = 3;
      kcdata->data.nbelem = 1;
    }

    void host(KernelConstantData<HGChebUncalibratedRecHitConstantData>* kcdata, cms::cuda::host::noncached::unique_ptr<double[]>& mem)
    {
      const std::vector<int> nelements = {kcdata->data.s_weights_};
      auto memsizes = get_memory_sizes_(nelements, 1, 0, 0);
      mem = cms::cuda::make_host_noncached_unique<double[]>(std::get<0>(memsizes), 0);

      kcdata->data.weights_ = mem.get();
      kcdata->data.nbytes = std::get<0>(memsizes);
      kcdata->data.ndelem = std::get<1>(memsizes) + 3;
      kcdata->data.nfelem = std::get<2>(memsizes) + 0;
      kcdata->data.nielem = std::get<3>(memsizes) + 0;
      kcdata->data.nuelem = 3;
      kcdata->data.nbelem = 1;
    }

    void host(const int& nhits, HGCUncalibratedRecHitSoA* soa, cms::cuda::host::noncached::unique_ptr<float[]>& mem)
    {
      std::vector<int> sizes = { 6*sizeof(float), 3*sizeof(uint32_t) };
      int size_tot = std::accumulate(sizes.begin(), sizes.end(), 0);
      mem = cms::cuda::make_host_noncached_unique<float[]>(nhits * size_tot, 0);

      soa->amplitude     = mem.get();
      soa->pedestal      = soa->amplitude    + nhits;
      soa->jitter        = soa->pedestal     + nhits;
      soa->chi2          = soa->jitter       + nhits;
      soa->OOTamplitude  = soa->chi2         + nhits;
      soa->OOTchi2       = soa->OOTamplitude + nhits;
      soa->flags         = reinterpret_cast<uint32_t*>(soa->OOTchi2 + nhits);
      soa->aux           = soa->flags        + nhits;
      soa->id            = soa->aux          + nhits;
      soa->nbytes = size_tot;
    }

    void host(const int& nhits, HGCRecHitSoA* soa, cms::cuda::host::unique_ptr<float[]>& mem)
    {
      std::vector<int> sizes = { 3*sizeof(float), 2*sizeof(uint32_t), sizeof(uint8_t) };
      int size_tot = std::accumulate(sizes.begin(), sizes.end(), 0);
      mem = cms::cuda::make_host_unique<float[]>(nhits * size_tot, 0);

      soa->energy     = mem.get();
      soa->time       = soa->energy     + nhits;
      soa->timeError  = soa->time       + nhits;
      soa->id         = reinterpret_cast<uint32_t*>(soa->timeError + nhits);
      soa->flagBits   = soa->id         + nhits;
      soa->son        = reinterpret_cast<uint8_t*>(soa->flagBits + nhits);
      soa->nbytes = size_tot;
    }
  }
}
