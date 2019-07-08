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

#include "gpuVertexFinder.h"

class PixelVertexProducerCUDA : public edm::global::EDProducer<> {
public:
  explicit PixelVertexProducerCUDA(const edm::ParameterSet& iConfig);
  ~PixelVertexProducerCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<CUDAProduct<PixelTrackCUDA>> tokenTrack_;
  edm::EDPutTokenT<CUDAProduct<ZVertexCUDA>> tokenVertex_;

  const gpuVertexFinder::Producer m_gpuAlgo;

  // Tracking cuts before sending tracks to vertex algo
  const float m_ptMin;

};

PixelVertexProducerCUDA::PixelVertexProducerCUDA(const edm::ParameterSet& conf) :
     tokenTrack_(consumes<CUDAProduct<PixelTrackCUDA>>(conf.getParameter<edm::InputTag>("pixelTrackSrc"))),
     tokenVertex_(produces<CUDAProduct<ZVertexCUDA>>()),
     m_gpuAlgo(conf.getParameter<bool>("useDensity"),
                conf.getParameter<bool>("useDBSCAN"),
                conf.getParameter<bool>("useIterative"),
                conf.getParameter<int>("minT"),
                conf.getParameter<double>("eps"),
                conf.getParameter<double>("errmax"),
                conf.getParameter<double>("chi2max")),
     m_ptMin(conf.getParameter<double>("PtMin"))  // 0.5 GeV
{}


void PixelVertexProducerCUDA::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  // Only one of these three algos can be used at once.
  // Maybe this should become a Plugin Factory
  desc.add<bool>("useDensity", true);
  desc.add<bool>("useDBSCAN", false);
  desc.add<bool>("useIterative", false);

  desc.add<int>("minT", 2);          // min number of neighbours to be "core"
  desc.add<double>("eps", 0.07);     // max absolute distance to cluster
  desc.add<double>("errmax", 0.01);  // max error to be "seed"
  desc.add<double>("chi2max", 9.);   // max normalized distance to cluster

  desc.add<double>("PtMin", 0.5);
  desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("caHitNtupletCUDA"));

  auto label = "pixelVertexCUDA";
  descriptions.add(label, desc);
}


void PixelVertexProducerCUDA::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {

  edm::Handle<CUDAProduct<PixelTrackCUDA>>  hTracks;
  iEvent.getByToken(tokenTrack_, hTracks);

  CUDAScopedContextProduce ctx{*hTracks};
  auto const& tracks = ctx.get(*hTracks);

  auto const * soa = tracks.soa();
  assert(soa);

  ctx.emplace(
        iEvent,
        tokenVertex_,
        std::move(m_gpuAlgo.makeAsync(ctx.stream(),soa,m_ptMin))
      );

}


DEFINE_FWK_MODULE(PixelVertexProducerCUDA);
