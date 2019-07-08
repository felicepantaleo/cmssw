#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"


#include "CUDADataFormats/Track/interface/PixelTrackCUDA.h"

class PixelTrackSoAFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit PixelTrackSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~PixelTrackSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;


  edm::EDGetTokenT<CUDAProduct<PixelTrackCUDA>> tokenCUDA_;
  edm::EDPutTokenT<PixelTrackCUDA::SoA> tokenSOA_;

  cudautils::host::unique_ptr<PixelTrackCUDA::SoA> m_soa;

};

PixelTrackSoAFromCUDA::PixelTrackSoAFromCUDA(const edm::ParameterSet& iConfig) :
  tokenCUDA_(consumes<CUDAProduct<PixelTrackCUDA>>(iConfig.getParameter<edm::InputTag>("src"))),
  tokenSOA_(produces<PixelTrackCUDA::SoA>())
{}


void PixelTrackSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;

   desc.add<edm::InputTag>("src", edm::InputTag("caHitNtupletCUDA"));
   descriptions.add("pixelTrackSoA", desc);

}


void PixelTrackSoAFromCUDA::acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  CUDAProduct<PixelTrackCUDA> const& inputDataWrapped = iEvent.get(tokenCUDA_);
  CUDAScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_soa = inputData.soaToHostAsync(ctx.stream());

}

void PixelTrackSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {

  // I suspect this is wrong
  //std::unique_ptr<PixelTrackCUDA::SoA> output(m_soa.release());
  //we need to make a copy to use standard destructor
  auto output = std::make_unique<PixelTrackCUDA::SoA>(*m_soa);
  iEvent.put(std::move(output));

}


DEFINE_FWK_MODULE(PixelTrackSoAFromCUDA);
