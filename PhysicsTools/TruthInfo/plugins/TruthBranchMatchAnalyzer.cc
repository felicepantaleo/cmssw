// Splits each reco trackster's truth-branch match into signal-branch (eventId 0)
// vs pileup-branch, and reports signal-branch reconstruction efficiency, using the
// pileup-aware mixed truth graph. Diagnostic for the PU200 truth-branch study.
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include <unordered_set>
#include <iostream>

class TruthBranchMatchAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TruthBranchMatchAnalyzer(edm::ParameterSet const& p)
      : gTok_(consumes<truth::Graph>(p.getParameter<edm::InputTag>("graph"))),
        tsTok_(consumes<std::vector<ticl::Trackster>>(p.getParameter<edm::InputTag>("tracksters"))),
        mTok_(consumes<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>>(
            p.getParameter<edm::InputTag>("association"))),
        eMin_(p.getParameter<double>("minSharedEnergy")) {}
  void analyze(edm::Event const& e, edm::EventSetup const&) override {
    auto const& g = e.get(gTok_);
    auto const& ts = e.get(tsTok_);
    auto const& m = e.get(mTok_);
    std::unordered_set<unsigned> matchedSignalRoots;
    unsigned n = ts.size(), sigT = 0, puT = 0, fakeT = 0;
    for (unsigned t = 0; t < n && t < m.size(); ++t) {
      float bestE = -1.f;
      unsigned bestIdx = 0;
      for (auto const& el : m[t])
        if (el.sharedEnergy() > bestE) { bestE = el.sharedEnergy(); bestIdx = el.index(); }
      if (bestE > eMin_) {
        bool sig = (bestIdx < g.nParticles() && g.particles()[bestIdx].eventId == 0ull);
        if (sig) { ++sigT; matchedSignalRoots.insert(bestIdx); } else ++puT;
      } else ++fakeT;
    }
    // signal-branch denominator: eventId 0, not backscattered, crossed calo (checkpoint 0)
    unsigned sigBr = 0, sigBrMatched = 0;
    for (unsigned i = 0; i < g.nParticles(); ++i) {
      auto const& pd = g.particles()[i];
      if (pd.eventId != 0ull || pd.backscattered)
        continue;
      bool crossed = false;
      for (auto const& cp : pd.checkpoints)
        if (cp.checkpointId == 0) { crossed = true; break; }
      if (!crossed)
        continue;
      ++sigBr;
      if (matchedSignalRoots.count(i))
        ++sigBrMatched;
    }
    totT_ += n; sigT_ += sigT; puT_ += puT; fakeT_ += fakeT;
    sigBr_ += sigBr; sigBrMatched_ += sigBrMatched; ++nev_;
  }
  void endJob() override {
    if (!totT_) return;
    std::cout << "TBMATCH tracksters/evt=" << double(totT_) / nev_
              << " matchSignal=" << double(sigT_) / totT_
              << " matchPU=" << double(puT_) / totT_
              << " fake=" << double(fakeT_) / totT_
              << " | signalBranches/evt=" << double(sigBr_) / nev_
              << " signalBranchEff=" << (sigBr_ ? double(sigBrMatched_) / sigBr_ : 0) << std::endl;
  }
private:
  edm::EDGetTokenT<truth::Graph> gTok_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> tsTok_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>> mTok_;
  double eMin_;
  unsigned long totT_ = 0, sigT_ = 0, puT_ = 0, fakeT_ = 0, sigBr_ = 0, sigBrMatched_ = 0, nev_ = 0;
};
DEFINE_FWK_MODULE(TruthBranchMatchAnalyzer);
