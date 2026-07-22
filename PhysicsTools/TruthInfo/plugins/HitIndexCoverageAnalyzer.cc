// Diagnostic: per-particle HGCal hit coverage of the logical-graph hit index,
// split by signal vs pileup eventId. Answers whether pileup branches carry hits.
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include <iostream>

class HitIndexCoverageAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HitIndexCoverageAnalyzer(edm::ParameterSet const& p)
      : gTok_(consumes<truth::Graph>(p.getParameter<edm::InputTag>("graph"))),
        hTok_(consumes<truth::LogicalGraphHitIndex>(p.getParameter<edm::InputTag>("hitIndex"))) {}
  void analyze(edm::Event const& e, edm::EventSetup const&) override {
    auto const& g = e.get(gTok_);
    auto const& hi = e.get(hTok_);
    unsigned nSig = 0, nPU = 0, sigHit = 0, puHit = 0;
    unsigned long sigHitCount = 0, puHitCount = 0;
    for (unsigned pid = 0; pid < g.nParticles(); ++pid) {
      bool sig = (g.particle(pid).eventId() == 0ull);
      auto direct = hi.directHits(truth::HitChannel::HGCalCalo, pid);
      auto sub = hi.subgraphHits(truth::HitChannel::HGCalCalo, pid);
      unsigned nh = direct.size() + sub.size();
      if (sig) {
        ++nSig;
        if (nh)
          ++sigHit;
        sigHitCount += nh;
      } else {
        ++nPU;
        if (nh)
          ++puHit;
        puHitCount += nh;
      }
    }
    std::cout << "COVERAGE evt: SIG particles=" << nSig << " withHits=" << sigHit << " hits=" << sigHitCount
              << " | PU particles=" << nPU << " withHits=" << puHit << " hits=" << puHitCount << std::endl;
  }

private:
  edm::EDGetTokenT<truth::Graph> gTok_;
  edm::EDGetTokenT<truth::LogicalGraphHitIndex> hTok_;
};
DEFINE_FWK_MODULE(HitIndexCoverageAnalyzer);
