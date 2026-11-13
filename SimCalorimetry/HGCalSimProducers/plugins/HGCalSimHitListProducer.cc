#include <string>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCalSimHitListBuilder.h"

// Collects the HGCAL Geant4 hits of the event, without pileup, into a hit list for the portable digitizer.
class HGCalSimHitListProducer : public edm::stream::EDProducer<> {
public:
  explicit HGCalSimHitListProducer(edm::ParameterSet const&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  HGCalSimHitListBuilder builder_;
  const edm::EDPutTokenT<hgcaldigi::HGCalSimHitListHost> listToken_;
  edm::EDPutTokenT<hgcaldigi::HGCalSimHitAccumulatorMonitorCollection> monitorToken_;
  const bool produceExtra_;
};

HGCalSimHitListProducer::HGCalSimHitListProducer(edm::ParameterSet const& iConfig)
    : builder_(iConfig, consumesCollector()),
      listToken_(produces(iConfig.getParameter<std::string>("instanceLabel"))),
      produceExtra_(iConfig.getParameter<bool>("produceExtra")) {
  if (produceExtra_) {
    monitorToken_ = produces(iConfig.getParameter<std::string>("instanceLabel"));
  }
}

void HGCalSimHitListProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  HGCalSimHitListBuilder::fillPSetDescription(desc);
  desc.add<std::string>("instanceLabel", "HGCalSimHitList");
  desc.add<bool>("produceExtra", false)->setComment("Also put the hit counters");
  descriptions.addWithDefaultLabel(desc);
}

void HGCalSimHitListProducer::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  builder_.beginEvent(iSetup);
  auto const& sources = builder_.sources();
  for (std::size_t i = 0; i < sources.size(); ++i) {
    auto const handle = iEvent.getHandle(sources[i].token);
    if (handle.isValid()) {
      builder_.add(*handle, i);
    }
  }
  iEvent.emplace(listToken_, builder_.makeList());
  if (produceExtra_) {
    iEvent.emplace(monitorToken_, builder_.monitors());
  }
}

DEFINE_FWK_MODULE(HGCalSimHitListProducer);
