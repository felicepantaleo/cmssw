// user include files
#include <unordered_map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"
#include "CommonTools/RecoAlgos/interface/MultiCollectionManager.h"

class RecHitMapProducer : public edm::global::EDProducer<> {
public:
  RecHitMapProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<HGCRecHitCollection> hits_ee_token_;
  const edm::EDGetTokenT<HGCRecHitCollection> hits_fh_token_;
  const edm::EDGetTokenT<HGCRecHitCollection> hits_bh_token_;
  const edm::EDGetTokenT<MultiCollectionManager<HGCRecHitCollection>> hgcalToken_;
  const edm::EDGetTokenT<reco::PFRecHitCollection> hits_eb_token_;
  const edm::EDGetTokenT<reco::PFRecHitCollection> hits_hb_token_;
  const edm::EDGetTokenT<reco::PFRecHitCollection> hits_ho_token_;
  bool hgcalOnly_;
};

DEFINE_FWK_MODULE(RecHitMapProducer);

using DetIdRecHitMap = std::unordered_map<DetId, const unsigned int>;

RecHitMapProducer::RecHitMapProducer(const edm::ParameterSet& ps)
    : hgcalToken_{consumes<MultiCollectionManager<HGCRecHitCollection>>(
          ps.getParameter<edm::InputTag>("HGCalMultiRecHits"))},
      hits_eb_token_(consumes<reco::PFRecHitCollection>(ps.getParameter<edm::InputTag>("EBInput"))),
      hits_hb_token_(consumes<reco::PFRecHitCollection>(ps.getParameter<edm::InputTag>("HBInput"))),
      hits_ho_token_(consumes<reco::PFRecHitCollection>(ps.getParameter<edm::InputTag>("HOInput"))),
      hgcalOnly_(ps.getParameter<bool>("hgcalOnly")) {
  produces<DetIdRecHitMap>("hgcalRecHitMap");
  if (!hgcalOnly_)
    produces<DetIdRecHitMap>("barrelRecHitMap");
}

void RecHitMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HGCalMultiRecHits", {"hgcalRecHitMultiCollectionProducer", ""});
  desc.add<edm::InputTag>("EBInput", {"particleFlowRecHitECAL", ""});
  desc.add<edm::InputTag>("HBInput", {"particleFlowRecHitHBHE", ""});
  desc.add<edm::InputTag>("HOInput", {"particleFlowRecHitHO", ""});
  desc.add<bool>("hgcalOnly", true);
  descriptions.add("recHitMapProducer", desc);
}

void RecHitMapProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {
  auto hitMapHGCal = std::make_unique<DetIdRecHitMap>();
  auto const& mgr = evt.get(hgcalToken_);
  auto flat = mgr.makeFlatView();  // by value
  for (unsigned int i = 0; i < flat.size(); ++i) {
    hitMapHGCal->emplace(flat[i].detid(), i);
  }

  // Add the HGCal rechit map to the event
  evt.put(std::move(hitMapHGCal), "hgcalRecHitMap");

  if (!hgcalOnly_) {
    auto hitMapBarrel = std::make_unique<DetIdRecHitMap>();
    MultiVectorManager<reco::PFRecHit> barrelRechitManager;
    barrelRechitManager.addVector(evt.get(hits_eb_token_));
    barrelRechitManager.addVector(evt.get(hits_hb_token_));
    barrelRechitManager.addVector(evt.get(hits_ho_token_));
    for (unsigned int i = 0; i < barrelRechitManager.size(); ++i) {
      const auto recHitDetId = barrelRechitManager[i].detId();
      hitMapBarrel->emplace(recHitDetId, i);
    }
    evt.put(std::move(hitMapBarrel), "barrelRecHitMap");
  }
}
