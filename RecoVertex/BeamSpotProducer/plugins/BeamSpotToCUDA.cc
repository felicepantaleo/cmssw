#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"


class BeamSpotToCUDA: public edm::global::EDProducer<> {
public:
  explicit BeamSpotToCUDA(const edm::ParameterSet& iConfig);
  ~BeamSpotToCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  edm::EDGetTokenT<reco::BeamSpot> bsGetToken_;
  edm::EDPutTokenT<CUDAProduct<BeamSpotCUDA>> bsPutToken_;
};

BeamSpotToCUDA::BeamSpotToCUDA(const edm::ParameterSet& iConfig):
  bsGetToken_{consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("src"))},
  bsPutToken_{produces<CUDAProduct<BeamSpotCUDA>>()}
{}

void BeamSpotToCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("offlineBeamSpot"));
  descriptions.addWithDefaultLabel(desc);
}

void BeamSpotToCUDA::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  CUDAScopedContext ctx{streamID};

  const reco::BeamSpot& bs = iEvent.get(bsGetToken_);

  edm::Service<CUDAService> cs;
  auto bsHost = cs->make_host_unique<BeamSpotCUDA::Data>(ctx.stream());
  bsHost->x = bs.x0();
  bsHost->y = bs.y0();
  bsHost->z = bs.z0();

  bsHost->sigmaZ = bs.sigmaZ();
  bsHost->beamWidthX = bs.BeamWidthX();
  bsHost->beamWidthY = bs.BeamWidthY();
  bsHost->dxdz = bs.dxdz();
  bsHost->dydz = bs.dydz();
  bsHost->emittanceX = bs.emittanceX();
  bsHost->emittanceY = bs.emittanceY();
  bsHost->betaStar = bs.betaStar();

  ctx.emplace(iEvent, bsPutToken_, std::move(bsHost), ctx.stream());
}

DEFINE_FWK_MODULE(BeamSpotToCUDA);

