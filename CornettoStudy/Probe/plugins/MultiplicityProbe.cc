// Study probe, not for integration. Per event and energy threshold: the number of calo-boundary particles
// that enter HGCAL (signal and pileup apart) and the number of tracksters of each collection.

#include <string>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

#include "TTree.h"

class MultiplicityProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit MultiplicityProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  static constexpr int kNThr = 6;
  static constexpr int kMaxColl = 8;
  const std::vector<double> thresholds_;
  const unsigned int minHits_;
  edm::EDGetTokenT<truth::Graph> graphToken_;
  edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  std::vector<edm::EDGetTokenT<std::vector<ticl::Trackster>>> collTokens_;
  TTree* tree_;
  int event_;
  int nSig_[kNThr], nPu_[kNThr], nTs_[kMaxColl * kNThr];
};

MultiplicityProbe::MultiplicityProbe(const edm::ParameterSet& cfg)
    : thresholds_(cfg.getParameter<std::vector<double>>("thresholds")),
      minHits_(cfg.getParameter<unsigned int>("minHGCalHits")),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))) {
  usesResource(TFileService::kSharedResource);
  for (auto const& n : cfg.getParameter<std::vector<std::string>>("collections"))
    collTokens_.push_back(consumes<std::vector<ticl::Trackster>>(edm::InputTag(n)));
  if (thresholds_.size() != kNThr || collTokens_.size() > kMaxColl)
    throw cms::Exception("Configuration")
        << "need " << kNThr << " thresholds and at most " << kMaxColl << " collections";
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("mult", "mult");
  tree_->Branch("event", &event_);
  tree_->Branch("nSig", nSig_, ("nSig[" + std::to_string(kNThr) + "]/I").c_str());
  tree_->Branch("nPu", nPu_, ("nPu[" + std::to_string(kNThr) + "]/I").c_str());
  tree_->Branch("nTs", nTs_, ("nTs[" + std::to_string(kMaxColl * kNThr) + "]/I").c_str());
}

void MultiplicityProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  event_ = event.id().event();
  std::fill(nSig_, nSig_ + kNThr, 0);
  std::fill(nPu_, nPu_ + kNThr, 0);
  std::fill(nTs_, nTs_ + kMaxColl * kNThr, 0);

  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (uint32_t p = 0; p < graph.particles().size(); ++p) {
    auto const& pd = graph.particles()[p];
    if (!pd.isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    unsigned int nHgc = 0;
    for (auto const& h : hits) {
      const auto det = DetId(h.detId).det();
      if (det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc)
        ++nHgc;
    }
    if (nHgc < minHits_)
      continue;
    const double e = pd.momentum.E();
    for (int k = 0; k < kNThr; ++k)
      if (e >= thresholds_[k])
        ++(pd.isSignal() ? nSig_[k] : nPu_[k]);
  }
  for (unsigned int c = 0; c < collTokens_.size(); ++c)
    for (auto const& ts : event.get(collTokens_[c]))
      for (int k = 0; k < kNThr; ++k)
        if (ts.raw_energy() >= thresholds_[k])
          ++nTs_[c * kNThr + k];
  tree_->Fill();
}

DEFINE_FWK_MODULE(MultiplicityProbe);
