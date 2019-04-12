#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"


TrackingRecHit2DCUDA::TrackingRecHit2DCUDA(uint32_t nHits, cuda::stream_t<>& stream) : m_nHits(nHits) {
  edm::Service<CUDAService> cs;

  m_store16 = cs->make_device_unique<uint16_t[]>(nHits*n16,stream);
  m_store32 = cs->make_device_unique<float[]>(nHits*n32+11+(1+TrackingRecHit2DSOAView::Hist::wsSize())/sizeof(float),stream);
  m_HistStore = cs->make_device_unique<TrackingRecHit2DSOAView::Hist>(stream);
   
  auto get16 = [&](int i) { return m_store16.get()+i*nHits;};
  auto get32 = [&](int i) { return m_store32.get()+i*nHits;};

  auto view = cs->make_host_unique<TrackingRecHit2DSOAView>(stream);

 // copy all the pointers
  view->m_hist = m_HistStore.get();
  view->m_hws = (uint8_t *)(get32(n32)+11);

  view->m_xl = get32(0);
  view->m_yl = get32(1);
  view->m_xerr = get32(2);
  view->m_yerr = get32(3);

  view->m_xg = get32(4);
  view->m_yg = get32(5);
  view->m_zg = get32(6);
  view->m_rg = get32(7);

  view->m_iphi = (int16_t *)get16(0);

  view->m_charge = (int32_t *)get32(8);
  view->m_xsize = (int16_t *)get16(2);
  view->m_ysize = (int16_t *)get16(3);
  view->m_detInd = get16(1);

  view->m_hitsLayerStart = (uint32_t *)get32(n32);

  // transfer veiw
  m_view = cs->make_device_unique<TrackingRecHit2DSOAView>(stream);
  cudautils::copyAsync(m_view, view, stream);

}
