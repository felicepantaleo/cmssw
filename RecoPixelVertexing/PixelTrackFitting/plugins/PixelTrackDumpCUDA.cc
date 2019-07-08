#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
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
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "CUDADataFormats/Vertex/interface/ZVertexCUDA.h"
#include "CUDADataFormats/Track/interface/PixelTrackCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"


class PixelTrackDumpCUDA : public edm::global::EDAnalyzer<> {
public:
  explicit PixelTrackDumpCUDA(const edm::ParameterSet& iConfig);
  ~PixelTrackDumpCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, edm::Event const & iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<CUDAProduct<PixelTrackCUDA>> tokenTrack_;
  edm::EDGetTokenT<CUDAProduct<ZVertexCUDA>> tokenVertex_;

};

PixelTrackDumpCUDA::PixelTrackDumpCUDA(const edm::ParameterSet& iConfig) :
  tokenTrack_(consumes<CUDAProduct<PixelTrackCUDA>>(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"))),
  tokenVertex_(consumes<CUDAProduct<ZVertexCUDA>>(iConfig.getParameter<edm::InputTag>("pixelVertexSrc")))

{}

void PixelTrackDumpCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

   desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("caHitNtupletCUDA"));
   desc.add<edm::InputTag>("pixelVertexSrc", edm::InputTag("pixelVertexCUDA"));
   descriptions.add("pixelTrackDumpCUDA", desc);
}

void PixelTrackDumpCUDA::analyze(edm::StreamID streamID, edm::Event const & iEvent, const edm::EventSetup& iSetup) const {

  edm::Handle<CUDAProduct<PixelTrackCUDA>>  hTracks;
  iEvent.getByToken(tokenTrack_, hTracks);

  CUDAScopedContextProduce ctx{*hTracks};
  auto const& tracks = ctx.get(*hTracks);

  auto const * tsoa = tracks.soa();
  assert(tsoa);

  edm::Handle<CUDAProduct<ZVertexCUDA>>  hVertices;
  iEvent.getByToken(tokenVertex_, hVertices);

  auto const& vertices = ctx.get(*hVertices);

  auto const * vsoa = vertices.soa();
  assert(vsoa);


}


DEFINE_FWK_MODULE(PixelTrackDumpCUDA);

