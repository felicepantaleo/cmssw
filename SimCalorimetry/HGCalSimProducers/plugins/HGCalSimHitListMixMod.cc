#include <memory>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCalSimHitListBuilder.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

// Mixing Module plugin: collects the HGCAL Geant4 hits of the signal and of every pileup event
// into one hit list for the portable digitizer. The hit times include the bunch crossing offset.
class HGCalSimHitListMixMod : public DigiAccumulatorMixMod {
public:
  HGCalSimHitListMixMod(edm::ParameterSet const&, edm::ProducesCollector, edm::ConsumesCollector&);

  void initializeEvent(edm::Event const&, edm::EventSetup const&) override;
  void accumulate(edm::Event const&, edm::EventSetup const&) override;
  void accumulate(PileUpEventPrincipal const&, edm::EventSetup const&, edm::StreamID const&) override;
  void finalizeEvent(edm::Event&, edm::EventSetup const&) override;

private:
  HGCalSimHitListBuilder builder_;
  const std::string instance_;
  const bool produceExtra_;
};

HGCalSimHitListMixMod::HGCalSimHitListMixMod(edm::ParameterSet const& iConfig,
                                             edm::ProducesCollector producesCollector,
                                             edm::ConsumesCollector& iC)
    : builder_(iConfig, iC),
      instance_(iConfig.getParameter<std::string>("instanceLabel")),
      produceExtra_(iConfig.getParameter<bool>("produceExtra")) {
  producesCollector.produces<hgcaldigi::HGCalSimHitListHost>(instance_);
  if (produceExtra_) {
    producesCollector.produces<hgcaldigi::HGCalSimHitAccumulatorMonitorCollection>(instance_);
  }
}

void HGCalSimHitListMixMod::initializeEvent(edm::Event const&, edm::EventSetup const& iSetup) {
  builder_.beginEvent(iSetup);
}

void HGCalSimHitListMixMod::accumulate(edm::Event const& iEvent, edm::EventSetup const&) {
  auto const& sources = builder_.sources();
  for (std::size_t i = 0; i < sources.size(); ++i) {
    auto const handle = iEvent.getHandle(sources[i].token);
    if (handle.isValid()) {
      builder_.add(*handle, i);
    }
  }
}

void HGCalSimHitListMixMod::accumulate(PileUpEventPrincipal const& pileupEvent,
                                       edm::EventSetup const&,
                                       edm::StreamID const&) {
  auto const& sources = builder_.sources();
  for (std::size_t i = 0; i < sources.size(); ++i) {
    edm::Handle<edm::PCaloHitContainer> handle;
    if (pileupEvent.getByLabel(sources[i].tag, handle)) {
      builder_.add(*handle, i);
    }
  }
}

void HGCalSimHitListMixMod::finalizeEvent(edm::Event& iEvent, edm::EventSetup const&) {
  iEvent.put(std::make_unique<hgcaldigi::HGCalSimHitListHost>(builder_.makeList()), instance_);
  if (produceExtra_) {
    iEvent.put(std::make_unique<hgcaldigi::HGCalSimHitAccumulatorMonitorCollection>(builder_.monitors()), instance_);
  }
}

DEFINE_DIGI_ACCUMULATOR(HGCalSimHitListMixMod);
