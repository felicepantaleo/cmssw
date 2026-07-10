// Diagnostic: dumps the per-sub-event hit count of the accumulator's merged HGCal
// PCaloHit collection, to check pileup hit coverage. Not for production.
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include <map>
#include <iostream>

class MergedHitDumpAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit MergedHitDumpAnalyzer(edm::ParameterSet const& p)
      : tok_(consumes<std::vector<PCaloHit>>(p.getParameter<edm::InputTag>("src"))) {}
  void analyze(edm::Event const& e, edm::EventSetup const&) override {
    auto const& hits = e.get(tok_);
    std::map<std::pair<int,int>, unsigned> byId;
    for (auto const& h : hits) { auto ee = h.eventId(); byId[{ee.bunchCrossing(), ee.event()}]++; }
    unsigned sig = 0, pu = 0, npu = 0;
    for (auto const& kv : byId) {
      if (kv.first == std::make_pair(0,0)) sig += kv.second;
      else { pu += kv.second; ++npu; }
    }
    std::cout << "HITDUMP evt: total=" << hits.size() << " signal(0,0)=" << sig
              << " PU_hits=" << pu << " PU_subevents=" << npu
              << " hits_per_PU_subevent=" << (npu ? double(pu)/npu : 0) << std::endl;
  }
private:
  edm::EDGetTokenT<std::vector<PCaloHit>> tok_;
};
DEFINE_FWK_MODULE(MergedHitDumpAnalyzer);
