// Study probe, not for integration. One row per layer cluster claimed by the tracks in the global arbitration: its
// features and the share of its simulated energy by truth class: particles of tracked charged candidates (and
// descendants), untracked charged hadrons, EM particles (photons, electrons, pi0), and other neutrals; and the signal share.

#include <algorithm>
#include <cmath>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
}  // namespace

class ClaimTruthProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ClaimTruthProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<std::vector<int>> claimedToken_;
  const edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<MapType> r2tToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  TTree* tree_;
  int event_, nHits_, det_;
  float e_, z_, r_, tracked_, untrackedCharged_, em_, otherNeutral_, signal_;
};

ClaimTruthProbe::ClaimTruthProbe(const edm::ParameterSet& cfg)
    : claimedToken_(consumes<std::vector<int>>(cfg.getParameter<edm::InputTag>("claimed"))),
      candToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("candidates"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      trackToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      r2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("claimed", "claimed");
  tree_->Branch("event", &event_);
  tree_->Branch("nHits", &nHits_);
  tree_->Branch("det", &det_);
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{{"E", &e_},
                                                                 {"z", &z_},
                                                                 {"r", &r_},
                                                                 {"tracked", &tracked_},
                                                                 {"untrackedCharged", &untrackedCharged_},
                                                                 {"em", &em_},
                                                                 {"otherNeutral", &otherNeutral_},
                                                                 {"signal", &signal_}})
    tree_->Branch(n, v);
}

void ClaimTruthProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& claimed = event.get(claimedToken_);
  auto const& cands = event.get(candToken_);
  auto const& lcs = event.get(lcToken_);
  const auto tkH = event.getHandle(trackToken_);
  auto const& r2t = event.get(r2tToken_).getMap();
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  event_ = event.id().event();
  const unsigned int nP = graph.particles().size();

  std::vector<char> tracked(nP, 0);
  for (auto const& c : cands) {
    if (c.charge() == 0 || c.trackPtrs().empty() || c.trackPtrs()[0].isNull() || c.trackPtrs()[0].id() != tkH.id())
      continue;
    const size_t t = c.trackPtrs()[0].key();
    if (t >= r2t.size() || r2t[t].empty())
      continue;
    auto best = std::min_element(
        r2t[t].begin(), r2t[t].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
    const unsigned int u = best->index();
    if (u >= nP || tracked[u])
      continue;
    tracked[u] = 1;
    for (auto const& d : truth::Particle(&graph, u).descendants())
      tracked[d.id()] = 1;
  }
  // Class of each calo-boundary particle: 0 tracked, 1 untracked charged hadron, 2 EM, 3 other neutral.
  std::unordered_map<uint32_t, std::vector<std::tuple<int, bool, float>>> cellTruth;
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (unsigned int p = 0; p < nP; ++p) {
    auto const& pd = graph.particles()[p];
    if (!pd.isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    const int a = std::abs(pd.pdgId);
    const int cls = tracked[p]                                       ? 0
                    : (a == 211 || a == 321 || a == 2212 || a == 13) ? 1
                    : (a == 22 || a == 11 || a == 111)               ? 2
                                                                     : 3;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    for (auto const& h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f)
        cellTruth[h.detId].emplace_back(cls, pd.eventId == 0, h.energy);
  }
  for (int v : claimed) {
    if (v < 0 || static_cast<size_t>(v) >= lcs.size())
      continue;
    auto const& lc = lcs[v];
    float byClass[4] = {0.f, 0.f, 0.f, 0.f}, sig = 0.f, tot = 0.f;
    for (auto const& [id, f] : lc.hitsAndFractions()) {
      auto it = cellTruth.find(id.rawId());
      if (it == cellTruth.end())
        continue;
      for (auto const& [cls, isSignal, e] : it->second) {
        byClass[cls] += f * e;
        tot += f * e;
        if (isSignal)
          sig += f * e;
      }
    }
    e_ = lc.energy();
    z_ = std::abs(static_cast<float>(lc.z()));
    r_ = std::hypot(lc.x(), lc.y());
    nHits_ = lc.hitsAndFractions().size();
    det_ = lc.hitsAndFractions().empty() ? -1 : static_cast<int>(lc.hitsAndFractions()[0].first.det());
    tracked_ = tot > 0.f ? byClass[0] / tot : -1.f;
    untrackedCharged_ = tot > 0.f ? byClass[1] / tot : -1.f;
    em_ = tot > 0.f ? byClass[2] / tot : -1.f;
    otherNeutral_ = tot > 0.f ? byClass[3] / tot : -1.f;
    signal_ = tot > 0.f ? sig / tot : -1.f;
    tree_->Fill();
  }
}

DEFINE_FWK_MODULE(ClaimTruthProbe);
