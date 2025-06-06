// Author: Felice Pantaleo (CERN), 2025, felice.pantaleo@cern.ch
//
// Produce a MultiCollectionManager<HGCRecHitCollection> that bundles the EE, FH
// and BH rec‑hit branches into one flat, persistent object.  Analysis modules
// such as RecHitMapProducer can consume the manager and access the hits via the
// usual MultiVectorManager API.
//
// Configuration example:
// cms.EDProducer("HGCalRecHitMultiCollectionProducer",
//                EEInput = cms.InputTag("HGCalRecHit", "HGCEERecHits"),
//                FHInput = cms.InputTag("HGCalRecHit", "HGCHEFRecHits"),
//                BHInput = cms.InputTag("HGCalRecHit", "HGCHEBRecHits"))

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "CommonTools/RecoAlgos/interface/MultiCollectionManager.h"

class HGCalRecHitMultiCollectionProducer : public edm::global::EDProducer<> {
public:
  explicit HGCalRecHitMultiCollectionProducer(edm::ParameterSet const& ps)
      : eeToken_{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("EEInput"))},
        fhToken_{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("FHInput"))},
        bhToken_{consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("BHInput"))} {
    produces<MultiCollectionManager<HGCRecHitCollection>>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("EEInput", {"HGCalRecHit", "HGCEERecHits"});
    desc.add<edm::InputTag>("FHInput", {"HGCalRecHit", "HGCHEFRecHits"});
    desc.add<edm::InputTag>("BHInput", {"HGCalRecHit", "HGCHEBRecHits"});
    descriptions.add("hgcalRecHitMultiCollectionProducer", desc);
  }

  void produce(edm::StreamID, edm::Event& evt, edm::EventSetup const&) const override {
    // Retrieve input collections
    auto const& ee = evt.getHandle(eeToken_);
    auto const& fh = evt.getHandle(fhToken_);
    auto const& bh = evt.getHandle(bhToken_);

    if (!ee.isValid() || !fh.isValid() || !bh.isValid()) {
      edm::LogWarning("HGCalRecHitMultiCollectionProducer")
          << "At least one HGCal rechit collection is missing. Producing an empty manager.";
      evt.put(std::make_unique<MultiCollectionManager<HGCRecHitCollection>>());
      return;
    }

    auto manager = std::make_unique<MultiCollectionManager<HGCRecHitCollection>>();
    manager->addCollection(edm::RefProd<HGCRecHitCollection>(ee));
    manager->addCollection(edm::RefProd<HGCRecHitCollection>(fh));
    manager->addCollection(edm::RefProd<HGCRecHitCollection>(bh));

    evt.put(std::move(manager));
  }

private:
  const edm::EDGetTokenT<HGCRecHitCollection> eeToken_;
  const edm::EDGetTokenT<HGCRecHitCollection> fhToken_;
  const edm::EDGetTokenT<HGCRecHitCollection> bhToken_;
};

DEFINE_FWK_MODULE(HGCalRecHitMultiCollectionProducer);
