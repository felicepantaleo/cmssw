#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerFillRcd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class LHCPhaseShiftProducer : public edm::global::EDProducer<> {
public:
  explicit LHCPhaseShiftProducer(const edm::ParameterSet& iConfig)
      : lhcInfoToken_(esConsumes(iConfig.getParameter<edm::ESInputTag>("lhcInfo"))) {
    produces<float>();
    produces<int>("bunchCrossing");
  };
  ~LHCPhaseShiftProducer() override {};

  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd> lhcInfoToken_;
};

void LHCPhaseShiftProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& lhcInfo = iSetup.getData(lhcInfoToken_);

  // get bunch crossing for phase shift
  auto bunchCrossing = std::make_unique<int>(iEvent.bunchCrossing());
  if (*bunchCrossing < 0) {
    const auto collidingBunchBitset = lhcInfo.bunchBitsetForBeam1() & lhcInfo.bunchBitsetForBeam2();
    const auto n = collidingBunchBitset.any() ? iEvent.id().event() % collidingBunchBitset.count() : -1;
    auto pos = collidingBunchBitset._Find_first();
    for (size_t i = 0; i < n; i++)
      pos = collidingBunchBitset._Find_next(pos);
    *bunchCrossing = pos < collidingBunchBitset.size() ? pos : -1;
  }

  // get phase shift
  auto phaseShift = std::make_unique<float>(0.);
  if (*bunchCrossing >= 0)
    *phaseShift = lhcInfo.phaseShift()[*bunchCrossing];

  iEvent.put(std::move(phaseShift));
  iEvent.put(std::move(bunchCrossing), "bunchCrossing");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void LHCPhaseShiftProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::ESInputTag>("lhcInfo", edm::ESInputTag("lhcPhaseShiftESSource", "lhcPhaseShift"));
  descriptions.add("lhcPhaseShift", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LHCPhaseShiftProducer);
