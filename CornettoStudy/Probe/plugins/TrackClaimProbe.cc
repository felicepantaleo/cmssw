// Study probe, not for integration. The track claim of TICLInterpretationProducer against the truth. One row per group
// of claiming tracks: the tracks in mode charged hadron, jet member or recovery that share a footprint trackster (or one
// track without a footprint), with deficit = sum of expectedDeposit(p) - raw energy of the footprint above 0.
// The raw energy of a layer cluster is split by its truth: the simulated HGCAL energy of its hits times the hit
// fractions, per calo-boundary particle. Classes of a particle: own (the deep match of a track of the group,
// DeepTrackMatch.h, and its descendants), other tracked (the deep match of another selected track), untracked neutral
// or untracked charged (by its ancestor at the generator final-state level, ReconstructableFinalState, else its own pdg
// id). Pile-up: eventId not 0. Own raw energy: the raw energy of every layer cluster times its own truth fraction.
// Tree claimEvent, one row per event: the raw energy of signal untracked neutral particles in all final tracksters and
// in all claim tracksters.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CornettoStudy/Probe/plugins/DeepTrackMatch.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
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
  // 1 neutral, 2 charged, 0 in neither list.
  int chargeClass(int pdg) {
    switch (std::abs(pdg)) {
      case 22:
      case 111:
      case 130:
      case 310:
      case 2112:
      case 3122:
      case 3212:
      case 3322:
        return 1;
      case 11:
      case 13:
      case 211:
      case 321:
      case 2212:
      case 3112:
      case 3222:
      case 3312:
      case 3334:
        return 2;
      default:
        return 0;
    }
  }
}  // namespace

class TrackClaimProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TrackClaimProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken_;
  const edm::EDGetTokenT<std::vector<int>> trackToTracksterToken_, trackToClaimToken_, trackModeToken_, neutralIdxToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<MapType> tkR2tToken_;
  const edm::EDGetTokenT<std::vector<unsigned int>> assignableToken_;
  const double response_, logSlope_;
  TTree* tree_;
  TTree* eventTree_;
  float sigNeutralInTs_, sigNeutralClaimed_;
  int event_, nTk_, nClaimLC_, ownSignal_;
  float p_, pLead_, eta_, expected_, footRaw_, deficit_, claimedRaw_, maxClaimLC_, claimCEE_, claimOwn_,
      claimOtherTracked_, claimNeutral_, claimCharged_, claimNoTruth_, claimPU_, ownTotal_, ownFoot_, ownClaim_,
      ownNeutral_, ownCEE_, ownTrueE_, claimSigNeutral_, claimSigCharged_;
};

TrackClaimProbe::TrackClaimProbe(const edm::ParameterSet& cfg)
    : trackToken_(consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"))),
      tracksterToken_(consumes<std::vector<ticl::Trackster>>(cfg.getParameter<edm::InputTag>("interpretations"))),
      trackToTracksterToken_(consumes<std::vector<int>>(
          edm::InputTag(cfg.getParameter<edm::InputTag>("interpretations").label(), "trackToTrackster"))),
      trackToClaimToken_(consumes<std::vector<int>>(
          edm::InputTag(cfg.getParameter<edm::InputTag>("interpretations").label(), "trackToClaimTrackster"))),
      trackModeToken_(consumes<std::vector<int>>(
          edm::InputTag(cfg.getParameter<edm::InputTag>("interpretations").label(), "trackMode"))),
      neutralIdxToken_(consumes<std::vector<int>>(
          edm::InputTag(cfg.getParameter<edm::InputTag>("interpretations").label(), "neutralIdx"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      tkR2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      assignableToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("assignableRoots"))),
      response_(cfg.getParameter<double>("expectedDepositResponse")),
      logSlope_(cfg.getParameter<double>("expectedDepositLogSlope")) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("claim", "claim");
  for (auto [n, v] : std::vector<std::pair<const char*, int*>>{
           {"event", &event_}, {"nTk", &nTk_}, {"nClaimLC", &nClaimLC_}, {"ownSignal", &ownSignal_}})
    tree_->Branch(n, v);
  for (auto [n, v] : std::vector<std::pair<const char*, float*>>{{"p", &p_},
                                                                {"pLead", &pLead_},
                                                                {"eta", &eta_},
                                                                {"expected", &expected_},
                                                                {"footRaw", &footRaw_},
                                                                {"deficit", &deficit_},
                                                                {"claimedRaw", &claimedRaw_},
                                                                {"maxClaimLC", &maxClaimLC_},
                                                                {"claimCEE", &claimCEE_},
                                                                {"claimOwn", &claimOwn_},
                                                                {"claimOtherTracked", &claimOtherTracked_},
                                                                {"claimNeutral", &claimNeutral_},
                                                                {"claimCharged", &claimCharged_},
                                                                {"claimNoTruth", &claimNoTruth_},
                                                                {"claimPU", &claimPU_},
                                                                {"ownTotal", &ownTotal_},
                                                                {"ownFoot", &ownFoot_},
                                                                {"ownClaim", &ownClaim_},
                                                                {"ownNeutral", &ownNeutral_},
                                                                {"ownCEE", &ownCEE_},
                                                                {"ownTrueE", &ownTrueE_},
                                                                {"claimSigNeutral", &claimSigNeutral_},
                                                                {"claimSigCharged", &claimSigCharged_}})
    tree_->Branch(n, v);
  eventTree_ = fs->make<TTree>("claimEvent", "claimEvent");
  eventTree_->Branch("event", &event_);
  eventTree_->Branch("sigNeutralInTs", &sigNeutralInTs_);
  eventTree_->Branch("sigNeutralClaimed", &sigNeutralClaimed_);
}

void TrackClaimProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& tracks = event.get(trackToken_);
  auto const& tracksters = event.get(tracksterToken_);
  auto const& toTs = event.get(trackToTracksterToken_);
  auto const& toClaim = event.get(trackToClaimToken_);
  auto const& mode = event.get(trackModeToken_);
  auto const& neutralIdx = event.get(neutralIdxToken_);
  auto const& lcs = event.get(lcToken_);
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& tkR2t = event.get(tkR2tToken_).getMap();
  event_ = event.id().event();
  const unsigned int nP = graph.particles().size();

  // Truth composition of each layer cluster: (calo-boundary particle, simulated energy x hit fraction).
  std::unordered_map<uint32_t, std::vector<std::pair<unsigned int, float>>> cellTruth;
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (unsigned int p = 0; p < nP; ++p) {
    if (!graph.particles()[p].isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    for (auto const& h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f)
        cellTruth[h.detId].emplace_back(p, h.energy);
  }
  std::vector<std::vector<std::pair<unsigned int, float>>> lcTruth(lcs.size());
  std::vector<float> lcTruthSum(lcs.size(), 0.f);
  std::unordered_map<unsigned int, float> comp;
  for (size_t v = 0; v < lcs.size(); ++v) {
    comp.clear();
    for (auto const& [id, frac] : lcs[v].hitsAndFractions()) {
      auto it = cellTruth.find(id.rawId());
      if (it == cellTruth.end())
        continue;
      for (auto const& [p, e] : it->second)
        comp[p] += frac * e;
    }
    for (auto const& [p, e] : comp) {
      lcTruth[v].emplace_back(p, e);
      lcTruthSum[v] += e;
    }
  }

  // Groups of claiming tracks, keyed by the footprint trackster (or -1 - track for a track without a footprint).
  std::map<int, std::vector<unsigned int>> groups;
  for (unsigned int t = 0; t < tracks.size() && t < mode.size(); ++t)
    if (mode[t] == 2 || mode[t] == 4 || mode[t] == 5)
      groups[t < toTs.size() && toTs[t] >= 0 ? toTs[t] : -1 - static_cast<int>(t)].push_back(t);

  // Deep matches: own particles per group, and particles of any selected track.
  const cornettostudy::DeepTrackMatch deepMatch(graph, tkR2t, event.get(assignableToken_));
  std::vector<int> ownGroup(nP, -1);
  std::vector<char> tracked(nP, 0);
  std::vector<int> deepOf(tracks.size(), -1);
  for (unsigned int t = 0; t < tracks.size() && t < mode.size(); ++t) {
    if (mode[t] < 0)
      continue;
    float purity;
    deepOf[t] = deepMatch(t, purity);
    if (deepOf[t] < 0)
      continue;
    tracked[deepOf[t]] = 1;
    for (auto const& d : truth::Particle(&graph, deepOf[t]).descendants())
      tracked[d.id()] = 1;
  }
  int g = 0;
  std::vector<int> groupKey;
  for (auto const& [key, tks] : groups) {
    for (unsigned int t : tks) {
      if (deepOf[t] < 0)
        continue;
      ownGroup[deepOf[t]] = g;
      for (auto const& d : truth::Particle(&graph, deepOf[t]).descendants())
        ownGroup[d.id()] = g;
    }
    groupKey.push_back(key);
    ++g;
  }
  // Class of an untracked particle by its generator final-state ancestor.
  std::vector<int> cls(nP, -1);
  auto classOf = [&](unsigned int p) {
    if (cls[p] >= 0)
      return cls[p];
    unsigned int q = p;
    bool found = false;
    for (int depth = 0; depth < 256; ++depth) {
      if (graph.particles()[q].isAtLevel(truth::LevelFlag::ReconstructableFinalState)) {
        found = true;
        break;
      }
      const auto parents = truth::Particle(&graph, q).parents();
      if (parents.empty())
        break;
      q = parents.front().id();
    }
    cls[p] = chargeClass(graph.particles()[found ? q : p].pdgId);
    return cls[p];
  };

  // Own raw energy per group: in all layer clusters, in CE-E, and in the clusters of the neutral final tracksters.
  std::vector<char> inNeutral(lcs.size(), 0);
  for (int n : neutralIdx)
    if (n >= 0 && n < static_cast<int>(tracksters.size()))
      for (auto v : tracksters[n].vertices())
        if (v < lcs.size())
          inNeutral[v] = 1;
  std::vector<float> ownTotal(g, 0.f), ownCEE(g, 0.f), ownNeutral(g, 0.f);
  for (size_t v = 0; v < lcs.size(); ++v) {
    if (!(lcTruthSum[v] > 0.f))
      continue;
    const bool cee = lcs[v].seed().det() == DetId::HGCalEE;
    for (auto const& [p, e] : lcTruth[v]) {
      if (ownGroup[p] < 0)
        continue;
      const float raw = lcs[v].energy() * e / lcTruthSum[v];
      ownTotal[ownGroup[p]] += raw;
      if (cee)
        ownCEE[ownGroup[p]] += raw;
      if (inNeutral[v])
        ownNeutral[ownGroup[p]] += raw;
    }
  }
  auto ownIn = [&](unsigned int v, int grp) {
    float own = 0.f;
    if (lcTruthSum[v] > 0.f)
      for (auto const& [p, e] : lcTruth[v])
        if (ownGroup[p] == grp)
          own += lcs[v].energy() * e / lcTruthSum[v];
    return own;
  };

  // Signal untracked neutral raw energy in all final tracksters and in the claim tracksters of all tracks.
  std::vector<char> inClaim(lcs.size(), 0);
  for (int c : toClaim)
    if (c >= 0 && c < static_cast<int>(tracksters.size()))
      for (auto v : tracksters[c].vertices())
        if (v < lcs.size())
          inClaim[v] = 1;
  std::vector<char> inTs(lcs.size(), 0);
  for (auto const& ts : tracksters)
    for (auto v : ts.vertices())
      if (v < lcs.size())
        inTs[v] = 1;
  sigNeutralInTs_ = sigNeutralClaimed_ = 0.f;
  for (size_t v = 0; v < lcs.size(); ++v) {
    if (!inTs[v] || !(lcTruthSum[v] > 0.f))
      continue;
    for (auto const& [p, e] : lcTruth[v]) {
      if (tracked[p] || graph.particles()[p].eventId != 0 || classOf(p) == 2)
        continue;
      const float raw = lcs[v].energy() * e / lcTruthSum[v];
      sigNeutralInTs_ += raw;
      if (inClaim[v])
        sigNeutralClaimed_ += raw;
    }
  }
  eventTree_->Fill();

  g = 0;
  for (auto const& [key, tks] : groups) {
    const int grp = g++;
    nTk_ = tks.size();
    p_ = pLead_ = expected_ = 0.f;
    eta_ = 0.f;
    for (unsigned int t : tks) {
      const float p = tracks[t].p();
      p_ += p;
      expected_ += (logSlope_ == 0. ? response_ : response_ + logSlope_ * std::log(p)) * p;
      if (p > pLead_) {
        pLead_ = p;
        eta_ = tracks[t].eta();
      }
    }
    footRaw_ = key >= 0 && key < static_cast<int>(tracksters.size()) ? tracksters[key].raw_energy() : 0.f;
    deficit_ = expected_ - footRaw_;
    if (!(deficit_ > 0.f))
      continue;
    ownFoot_ = 0.f;
    if (key >= 0 && key < static_cast<int>(tracksters.size()))
      for (auto v : tracksters[key].vertices())
        if (v < lcs.size())
          ownFoot_ += ownIn(v, grp);
    claimedRaw_ = maxClaimLC_ = claimCEE_ = claimOwn_ = claimOtherTracked_ = claimNeutral_ = claimCharged_ =
        claimNoTruth_ = claimPU_ = ownClaim_ = claimSigNeutral_ = claimSigCharged_ = 0.f;
    nClaimLC_ = 0;
    for (unsigned int t : tks) {
      if (t >= toClaim.size() || toClaim[t] < 0 || toClaim[t] >= static_cast<int>(tracksters.size()))
        continue;
      for (auto v : tracksters[toClaim[t]].vertices()) {
        if (v >= lcs.size())
          continue;
        const float e = lcs[v].energy();
        ++nClaimLC_;
        claimedRaw_ += e;
        maxClaimLC_ = std::max(maxClaimLC_, e);
        if (lcs[v].seed().det() == DetId::HGCalEE)
          claimCEE_ += e;
        if (!(lcTruthSum[v] > 0.f)) {
          claimNoTruth_ += e;
          continue;
        }
        for (auto const& [p, te] : lcTruth[v]) {
          const float raw = e * te / lcTruthSum[v];
          if (graph.particles()[p].eventId != 0)
            claimPU_ += raw;
          if (ownGroup[p] == grp)
            claimOwn_ += raw;
          else if (tracked[p])
            claimOtherTracked_ += raw;
          else if (classOf(p) == 2) {
            claimCharged_ += raw;
            if (graph.particles()[p].eventId == 0)
              claimSigCharged_ += raw;
          } else {
            claimNeutral_ += raw;
            if (graph.particles()[p].eventId == 0)
              claimSigNeutral_ += raw;
          }
        }
      }
    }
    ownClaim_ = claimOwn_;
    ownTotal_ = ownTotal[grp];
    ownCEE_ = ownCEE[grp];
    ownNeutral_ = ownNeutral[grp];
    ownTrueE_ = 0.f;
    ownSignal_ = 0;
    for (unsigned int t : tks)
      if (deepOf[t] >= 0) {
        ownTrueE_ += graph.particles()[deepOf[t]].momentum.energy();
        ownSignal_ |= graph.particles()[deepOf[t]].eventId == 0 ? 1 : 0;
      }
    tree_->Fill();
  }
}

DEFINE_FWK_MODULE(TrackClaimProbe);
