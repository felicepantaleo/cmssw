#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "CAHitQuadrupletGeneratorGPU.h"
#include "CUDADataFormats/Track/interface/PixelTrackCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"


class CAHitNtupletCUDA : public edm::global::EDProducer<> {
public:
  explicit CAHitNtupletCUDA(const edm::ParameterSet& iConfig);
  ~CAHitNtupletCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<CUDAProduct<TrackingRecHit2DCUDA>> tokenHit_;
  edm::EDPutTokenT<CUDAProduct<PixelTrackCUDA>> tokenTrack_;

  CAHitQuadrupletGeneratorGPU gpuAlgo_;

  const bool useRiemannFit_;

};

CAHitNtupletCUDA::CAHitNtupletCUDA(const edm::ParameterSet& iConfig) :
      tokenHit_(consumes<CUDAProduct<TrackingRecHit2DCUDA>>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      tokenTrack_(produces<CUDAProduct<PixelTrackCUDA>>()),
      gpuAlgo_(iConfig, consumesCollector()),
      useRiemannFit_(iConfig.getParameter<bool>("useRiemannFit")) {

}


void CAHitNtupletCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsCUDAPreSplitting"));
  desc.add<bool>("useRiemannFit", false)->setComment("true for Riemann, false for BrokenLine");

  CAHitQuadrupletGeneratorGPU::fillDescriptions(desc);
  auto label = "caHitNtupletCUDA";
  descriptions.add(label, desc);
}

void CAHitNtupletCUDA::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& es) const {

  edm::Handle<CUDAProduct<TrackingRecHit2DCUDA>>  hHits;
  iEvent.getByToken(tokenHit_, hHits);

  CUDAScopedContextProduce ctx{*hHits};
  auto const& hits = ctx.get(*hHits);

  ctx.emplace(
      iEvent,
      tokenTrack_,
      std::move(gpuAlgo_.makeTuplesAsync(hits, ctx.stream())));

}



DEFINE_FWK_MODULE(CAHitNtupletCUDA);
