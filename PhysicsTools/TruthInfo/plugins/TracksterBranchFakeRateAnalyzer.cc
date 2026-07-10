// Prints the trackster-to-branch fake rate: fraction of reco tracksters that share
// no truth-branch energy above threshold. With the pileup-aware mixed truth graph
// (pileup branches carry hits), pileup tracksters can match, so this measures a
// genuine PU-aware fake rate. Diagnostic for the pileup-truth study.
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include <atomic>

class TracksterBranchFakeRateAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TracksterBranchFakeRateAnalyzer(edm::ParameterSet const& p)
      : tsToken_(consumes<std::vector<ticl::Trackster>>(p.getParameter<edm::InputTag>("tracksters"))),
        mapToken_(consumes<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>>(
            p.getParameter<edm::InputTag>("association"))),
        eMin_(p.getParameter<double>("minSharedEnergy")) {}
  void analyze(edm::Event const& e, edm::EventSetup const&) override {
    auto const& ts = e.get(tsToken_);
    auto const& m = e.get(mapToken_);
    unsigned n = ts.size(), matched = 0;
    for (unsigned t = 0; t < n && t < m.size(); ++t) {
      bool has = false;
      for (auto const& el : m[t])
        if (el.sharedEnergy() > eMin_) {
          has = true;
          break;
        }
      if (has)
        ++matched;
    }
    tot_ += n;
    match_ += matched;
    ++nev_;
  }
  void endJob() override {
    if (tot_)
      edm::LogSystem("FakeRate") << "TRACKSTER_BRANCH: tracksters/evt=" << double(tot_) / nev_
                                 << " matched=" << double(match_) / tot_
                                 << " FAKE_RATE=" << double(tot_ - match_) / tot_;
    std::cout << "FAKERATE_RESULT tracksters_per_evt=" << (nev_ ? double(tot_) / nev_ : 0)
              << " matched=" << (tot_ ? double(match_) / tot_ : 0)
              << " fake_rate=" << (tot_ ? double(tot_ - match_) / tot_ : 0) << std::endl;
  }

private:
  edm::EDGetTokenT<std::vector<ticl::Trackster>> tsToken_;
  edm::EDGetTokenT<ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>> mapToken_;
  double eMin_;
  unsigned long tot_ = 0, match_ = 0, nev_ = 0;
};
DEFINE_FWK_MODULE(TracksterBranchFakeRateAnalyzer);
