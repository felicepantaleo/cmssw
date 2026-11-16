// Study probe, not for integration. One row per arbitration hypothesis (dumped by TICLInterpretationProducer with
// dumpHypotheses = True): features, the model features x of the producer, the truth label from the truth graph, and
// the conflict structure. A second tree lists the conflicting pairs: same track, or a shared layer-cluster energy above
// maxSharedFraction of the smaller footprint. A third tree has one row per neutral final trackster: the inputs of the
// neutral species model and the truth species of its layer clusters.

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTICL/Interpretation/interface/ModelFeatures.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

#include "TString.h"
#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
  bool isChargedHadron(int pdg) {
    const int a = std::abs(pdg);
    return a == 211 || a == 321 || a == 2212;
  }
  bool isNeutralHadron(int pdg) {
    const int a = std::abs(pdg);
    return a == 130 || a == 2112 || a == 310;
  }
  bool isNeutral(int pdg) { return isNeutralHadron(pdg) || pdg == 22 || pdg == 111; }
}  // namespace

class HypothesisTruthDumper : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HypothesisTruthDumper(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  template <typename T>
  edm::EDGetTokenT<std::vector<T>> hyp(const std::string& label, const std::string& instance) {
    return consumes<std::vector<T>>(edm::InputTag(label, instance));
  }
  edm::EDGetTokenT<std::vector<int>> typeT_, trackT_, accT_, lcOffT_, lcsT_, jetOffT_, jetTkT_;
  edm::EDGetTokenT<std::vector<float>> scoreT_, rawET_, rawEmET_, xT_, yT_, zT_, pidT_, weightT_, featuresT_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> finalT_;
  edm::EDGetTokenT<std::vector<int>> neutralIdxT_, neutralPdgT_, neutralFromPhotonT_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  edm::EDGetTokenT<MapType> tkR2tToken_;
  edm::EDGetTokenT<truth::Graph> graphToken_;
  edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  edm::EDGetTokenT<std::vector<unsigned int>> legsToken_;
  const double maxSharedFraction_;

  TTree* tree_;
  TTree* pairs_;
  int event_, lumi_, idx_, type_, track_, gsf_, accepted_, nLC_, nJetTracks_, label_, objPdg_, trackObjPdg_, signal_;
  int comp_, compSize_, trackSameObj_, nMuonTracks_, nElectronTracks_, species_;
  float score_, rawE_, rawEmE_, eta_, phi_, pid_[8], trackP_, trackPt_, trackEta_, eOverP_, dR_, purity_, objFrac_,
      neutralFrac_, speciesFrac_, completeness_, trackPurity_, weight_;
  // The model features of the row, as the producer computes them.
  float x_[ticl::modelfeatures::hypothesis::kN];
  // Neutral final tracksters: the neutral species model inputs and the truth species.
  TTree* neutrals_;
  int nIdx_, nPred_, nFromPhoton_, nSpecies_, nObjPdg_, nSignal_;
  float nRawE_, nRegE_, nSpeciesFrac_, nPurity_, nX_[ticl::modelfeatures::neutralSpecies::kN];
  int pEvent_, pLumi_, pI_, pJ_;
};

HypothesisTruthDumper::HypothesisTruthDumper(const edm::ParameterSet& cfg)
    : maxSharedFraction_(cfg.getParameter<double>("maxSharedFraction")) {
  usesResource(TFileService::kSharedResource);
  const auto m = cfg.getParameter<std::string>("hypothesisModule");
  typeT_ = hyp<int>(m, "hypType");
  trackT_ = hyp<int>(m, "hypTrack");
  accT_ = hyp<int>(m, "hypAccepted");
  lcOffT_ = hyp<int>(m, "hypLCOffsets");
  lcsT_ = hyp<int>(m, "hypLCs");
  jetOffT_ = hyp<int>(m, "hypJetOffsets");
  jetTkT_ = hyp<int>(m, "hypJetTracks");
  scoreT_ = hyp<float>(m, "hypScore");
  rawET_ = hyp<float>(m, "hypRawE");
  rawEmET_ = hyp<float>(m, "hypRawEmE");
  xT_ = hyp<float>(m, "hypX");
  yT_ = hyp<float>(m, "hypY");
  zT_ = hyp<float>(m, "hypZ");
  pidT_ = hyp<float>(m, "hypPid");
  weightT_ = hyp<float>(m, "hypWeight");
  featuresT_ = hyp<float>(m, "hypFeatures");
  finalT_ = consumes<std::vector<ticl::Trackster>>(edm::InputTag(m));
  neutralIdxT_ = hyp<int>(m, "neutralIdx");
  neutralPdgT_ = hyp<int>(m, "neutralPdg");
  neutralFromPhotonT_ = hyp<int>(m, "neutralFromPhoton");
  lcToken_ = consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"));
  trackToken_ = consumes<std::vector<reco::Track>>(cfg.getParameter<edm::InputTag>("tracks"));
  tkR2tToken_ = consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"));
  graphToken_ = consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"));
  hitIndexToken_ = consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"));
  legsToken_ = consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("legs"));

  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("hyps", "hyps");
  for (auto [n, p] : std::vector<std::pair<const char*, int*>>{{"event", &event_},
                                                               {"lumi", &lumi_},
                                                               {"idx", &idx_},
                                                               {"type", &type_},
                                                               {"track", &track_},
                                                               {"gsf", &gsf_},
                                                               {"accepted", &accepted_},
                                                               {"nLC", &nLC_},
                                                               {"nJetTracks", &nJetTracks_},
                                                               {"label", &label_},
                                                               {"objPdg", &objPdg_},
                                                               {"trackObjPdg", &trackObjPdg_},
                                                               {"signal", &signal_},
                                                               {"comp", &comp_},
                                                               {"compSize", &compSize_},
                                                               {"trackSameObj", &trackSameObj_},
                                                               {"nMuonTracks", &nMuonTracks_},
                                                               {"nElectronTracks", &nElectronTracks_},
                                                               {"species", &species_}})
    tree_->Branch(n, p);
  for (auto [n, p] : std::vector<std::pair<const char*, float*>>{{"score", &score_},
                                                                 {"rawE", &rawE_},
                                                                 {"rawEmE", &rawEmE_},
                                                                 {"eta", &eta_},
                                                                 {"phi", &phi_},
                                                                 {"trackP", &trackP_},
                                                                 {"trackPt", &trackPt_},
                                                                 {"trackEta", &trackEta_},
                                                                 {"eOverP", &eOverP_},
                                                                 {"dR", &dR_},
                                                                 {"purity", &purity_},
                                                                 {"objFrac", &objFrac_},
                                                                 {"neutralFrac", &neutralFrac_},
                                                                 {"speciesFrac", &speciesFrac_},
                                                                 {"completeness", &completeness_},
                                                                 {"trackPurity", &trackPurity_},
                                                                 {"weight", &weight_}})
    tree_->Branch(n, p);
  tree_->Branch("pid", pid_, "pid[8]/F");
  tree_->Branch("x", x_, Form("x[%d]/F", ticl::modelfeatures::hypothesis::kN));
  neutrals_ = fs->make<TTree>("neutrals", "neutral final tracksters");
  for (auto [n, p] : std::vector<std::pair<const char*, int*>>{{"event", &event_},
                                                               {"lumi", &lumi_},
                                                               {"idx", &nIdx_},
                                                               {"pdgPred", &nPred_},
                                                               {"fromPhoton", &nFromPhoton_},
                                                               {"species", &nSpecies_},
                                                               {"objPdg", &nObjPdg_},
                                                               {"signal", &nSignal_}})
    neutrals_->Branch(n, p);
  for (auto [n, p] : std::vector<std::pair<const char*, float*>>{
           {"rawE", &nRawE_}, {"regE", &nRegE_}, {"speciesFrac", &nSpeciesFrac_}, {"purity", &nPurity_}})
    neutrals_->Branch(n, p);
  neutrals_->Branch("x", nX_, Form("x[%d]/F", ticl::modelfeatures::neutralSpecies::kN));
  pairs_ = fs->make<TTree>("pairs", "conflicting hypothesis pairs");
  pairs_->Branch("event", &pEvent_);
  pairs_->Branch("lumi", &pLumi_);
  pairs_->Branch("i", &pI_);
  pairs_->Branch("j", &pJ_);
}

void HypothesisTruthDumper::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& type = event.get(typeT_);
  const unsigned int nH = type.size();
  if (nH == 0)
    return;
  auto const& track = event.get(trackT_);
  auto const& acc = event.get(accT_);
  auto const& lcOff = event.get(lcOffT_);
  auto const& lcs = event.get(lcsT_);
  auto const& jetOff = event.get(jetOffT_);
  auto const& jetTk = event.get(jetTkT_);
  std::unordered_set<int> muonTrack, electronTrack;
  for (unsigned int h = 0; h < nH; ++h)
    if (track[h] >= 0 && type[h] == 0)
      muonTrack.insert(track[h]);
    else if (track[h] >= 0 && type[h] == 1)
      electronTrack.insert(track[h]);
  auto const& score = event.get(scoreT_);
  auto const& rawE = event.get(rawET_);
  auto const& rawEmE = event.get(rawEmET_);
  auto const& xs = event.get(xT_);
  auto const& ys = event.get(yT_);
  auto const& zs = event.get(zT_);
  auto const& pid = event.get(pidT_);
  auto const& weight = event.get(weightT_);  // empty in greedy mode
  auto const& features = event.get(featuresT_);
  constexpr unsigned int kNx = ticl::modelfeatures::hypothesis::kN;
  if (features.size() != nH * kNx)
    throw cms::Exception("LogicError") << "hypFeatures has " << features.size() << " values for " << nH
                                       << " hypotheses";
  auto const& layerClusters = event.get(lcToken_);
  auto const& tracks = event.get(trackToken_);
  auto const& tkR2t = event.get(tkR2tToken_).getMap();
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& legs = event.get(legsToken_);
  event_ = event.id().event();
  lumi_ = event.id().luminosityBlock();

  // Truth object of a particle: its signal leg when it descends from one, else the particle itself.
  const unsigned int nP = graph.particles().size();
  std::vector<int> objectOf(nP, -1);
  for (unsigned int leg : legs) {
    if (leg >= nP || !graph.particles()[leg].isSignal())
      continue;
    objectOf[leg] = static_cast<int>(leg);
    for (auto const& d : truth::Particle(&graph, leg).descendants())
      objectOf[d.id()] = static_cast<int>(leg);
  }
  auto objectFor = [&](unsigned int p) { return objectOf[p] >= 0 ? objectOf[p] : static_cast<int>(p); };
  auto pdgOf = [&](int obj) { return obj >= 0 && obj < static_cast<int>(nP) ? graph.particles()[obj].pdgId : 0; };

  // Simulated HGCAL energy per cell and truth object, from the calo-boundary particles.
  std::unordered_map<uint32_t, std::vector<std::pair<int, float>>> cellTruth;
  // EM origin per cell: (pi0 parent or -1, photon daughter of the pi0 or the EM particle itself, energy); hadronic
  // energy has photon -1.
  std::unordered_map<uint32_t, std::vector<std::tuple<int, int, float>>> cellSpecies;
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  for (unsigned int p = 0; p < nP; ++p) {
    if (!graph.particles()[p].isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    const int obj = objectFor(p);
    // Walk up the photon / electron chain to a pi0 parent.
    int pi0 = -1, photon = -1;
    const int a = std::abs(graph.particles()[p].pdgId);
    if (a == 22 || a == 11) {
      photon = static_cast<int>(p);
      int cur = static_cast<int>(p);
      for (int step = 0; step < 8; ++step) {
        const auto parents = truth::Particle(&graph, cur).parents();
        if (parents.empty())
          break;
        const int q = static_cast<int>(parents.front().id());
        const int aq = std::abs(graph.particles()[q].pdgId);
        if (aq == 111) {
          pi0 = q;
          photon = cur;
          break;
        }
        if (aq != 22 && aq != 11)
          break;
        cur = q;
      }
    }
    for (auto const& h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f) {
        cellTruth[h.detId].emplace_back(obj, h.energy);
        cellSpecies[h.detId].emplace_back(pi0, photon, h.energy);
      }
  }
  // Footprint species: 111 when one pi0 dominates the EM energy with at least two photons above 10% of its energy, 22
  // for other EM energy, 130 when hadronic energy dominates. speciesFrac is the share of the dominant species.
  auto speciesOf = [&](std::span<const int> lcList, float& frac) {
    std::map<int, std::map<int, double>> byPi0;
    double em = 0., had = 0.;
    for (const int lcIdx : lcList) {
      const auto lc = static_cast<unsigned int>(lcIdx);
      if (lc >= layerClusters.size())
        continue;
      for (auto const& [id, f] : layerClusters[lc].hitsAndFractions()) {
        auto it = cellSpecies.find(id.rawId());
        if (it == cellSpecies.end())
          continue;
        for (auto const& [pi0, photon, e] : it->second) {
          if (photon < 0) {
            had += f * e;
            continue;
          }
          em += f * e;
          if (pi0 >= 0)
            byPi0[pi0][photon] += f * e;
        }
      }
    }
    frac = 0.f;
    if (em + had <= 0.)
      return 0;
    if (had > em) {
      frac = had / (em + had);
      return 130;
    }
    double bestE = 0.;
    int nPhotons = 0;
    for (auto const& [pi0, photons] : byPi0) {
      double e = 0.;
      for (auto const& [ph, eph] : photons)
        e += eph;
      if (e > bestE) {
        bestE = e;
        nPhotons = 0;
        for (auto const& [ph, eph] : photons)
          nPhotons += eph >= 0.1 * e ? 1 : 0;
      }
    }
    if (nPhotons >= 2 && bestE >= 0.5 * em) {
      frac = bestE / (em + had);
      return 111;
    }
    frac = em / (em + had);
    return 22;
  };
  auto lcsOf = [&](unsigned int h) { return std::span<const int>(lcs.data() + lcOff[h], lcOff[h + 1] - lcOff[h]); };
  auto species = [&](unsigned int h, float& frac) { return speciesOf(lcsOf(h), frac); };

  // Footprint composition by truth object (simulated energy weighted by the layer-cluster hit fraction).
  auto compositionOf = [&](std::span<const int> lcList) {
    std::map<int, double> comp;
    for (const int lcIdx : lcList) {
      const auto lc = static_cast<unsigned int>(lcIdx);
      if (lc >= layerClusters.size())
        continue;
      for (auto const& [id, frac] : layerClusters[lc].hitsAndFractions()) {
        auto it = cellTruth.find(id.rawId());
        if (it == cellTruth.end())
          continue;
        for (auto const& [obj, e] : it->second)
          comp[obj] += frac * e;
      }
    }
    return comp;
  };
  auto composition = [&](unsigned int h) { return compositionOf(lcsOf(h)); };
  // Truth object of a track: the best Fixed reco-to-truth match, mapped to its object.
  auto trackObject = [&](int t, float& purity) {
    purity = 0.f;
    if (t < 0 || t >= static_cast<int>(tkR2t.size()) || tkR2t[t].empty())
      return -1;
    auto best = std::min_element(
        tkR2t[t].begin(), tkR2t[t].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
    purity = 1.f - best->score();
    return best->index() < nP ? objectFor(best->index()) : -1;
  };
  // Truth particle of a track (not mapped to an object), -1 when none.
  auto trackParticle = [&](int t) {
    if (t < 0 || t >= static_cast<int>(tkR2t.size()) || tkR2t[t].empty())
      return -1;
    auto best = std::min_element(
        tkR2t[t].begin(), tkR2t[t].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
    return best->index() < nP ? static_cast<int>(best->index()) : -1;
  };
  // The photon that a track comes from through a chain of e+-: the first photon above its truth particle, which must
  // be an e+-; -1 when none.
  auto conversionPhoton = [&](int t) {
    int cur = trackParticle(t);
    if (cur < 0 || std::abs(graph.particles()[cur].pdgId) != 11)
      return -1;
    for (int step = 0; step < 8; ++step) {
      const auto parents = truth::Particle(&graph, cur).parents();
      if (parents.empty())
        return -1;
      const int q = static_cast<int>(parents.front().id());
      const int aq = std::abs(graph.particles()[q].pdgId);
      if (aq == 22)
        return q;
      if (aq != 11)
        return -1;
      cur = q;
    }
    return -1;
  };
  // True when particle p is the photon g or descends from it through e+- and photons.
  auto fromPhoton = [&](int p, int g) {
    for (int step = 0; step < 8 && p >= 0; ++step) {
      if (p == g)
        return true;
      const int ap = std::abs(graph.particles()[p].pdgId);
      if (ap != 11 && ap != 22)
        return false;
      const auto parents = truth::Particle(&graph, p).parents();
      if (parents.empty())
        return false;
      p = static_cast<int>(parents.front().id());
    }
    return false;
  };
  // Completeness: share of the HGCAL energy of a particle and its descendants that the footprint of hypothesis h holds.
  std::unordered_map<uint32_t, float> subtree;
  auto completeness = [&](unsigned int h, int u) {
    if (u < 0)
      return 0.f;
    subtree.clear();
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, u, hits);
    double total = 0.;
    for (auto const& hit : hits)
      if (isHGCal(hit.detId) && hit.energy > 0.f) {
        subtree[hit.detId] += hit.energy;
        total += hit.energy;
      }
    if (total <= 0.)
      return 0.f;
    double held = 0.;
    for (int k = lcOff[h]; k < lcOff[h + 1]; ++k) {
      const auto lc = static_cast<unsigned int>(lcs[k]);
      if (lc >= layerClusters.size())
        continue;
      for (auto const& [id, f] : layerClusters[lc].hitsAndFractions())
        if (auto it = subtree.find(id.rawId()); it != subtree.end())
          held += f * it->second;
    }
    return static_cast<float>(held / total);
  };

  // Conflict graph: shared track, or shared layer-cluster energy above maxSharedFraction of the smaller footprint.
  std::vector<double> footE(nH, 0.);
  std::unordered_map<int, std::vector<unsigned int>> lcOwners;
  for (unsigned int h = 0; h < nH; ++h)
    for (int k = lcOff[h]; k < lcOff[h + 1]; ++k) {
      const auto lc = static_cast<unsigned int>(lcs[k]);
      if (lc < layerClusters.size()) {
        footE[h] += layerClusters[lc].energy();
        lcOwners[lcs[k]].push_back(h);
      }
    }
  std::map<std::pair<unsigned int, unsigned int>, double> shared;
  for (auto const& [lc, owners] : lcOwners)
    for (unsigned int a = 0; a < owners.size(); ++a)
      for (unsigned int b = a + 1; b < owners.size(); ++b)
        shared[{std::min(owners[a], owners[b]), std::max(owners[a], owners[b])}] += layerClusters[lc].energy();
  std::vector<std::vector<int>> tracksOf(nH);
  for (unsigned int h = 0; h < nH; ++h) {
    if (track[h] >= 0)
      tracksOf[h].push_back(track[h]);
    for (int k = jetOff[h]; k < jetOff[h + 1]; ++k)
      tracksOf[h].push_back(jetTk[k]);
  }
  std::unordered_map<int, std::vector<unsigned int>> trackOwners;
  for (unsigned int h = 0; h < nH; ++h)
    for (int t : tracksOf[h])
      trackOwners[t].push_back(h);
  std::vector<std::pair<unsigned int, unsigned int>> conflicts;
  for (auto const& [ab, e] : shared) {
    const double smaller = std::min(footE[ab.first], footE[ab.second]);
    if (smaller > 0. && e / smaller > maxSharedFraction_)
      conflicts.push_back(ab);
  }
  for (auto const& [t, owners] : trackOwners)
    for (unsigned int a = 0; a < owners.size(); ++a)
      for (unsigned int b = a + 1; b < owners.size(); ++b)
        conflicts.emplace_back(std::min(owners[a], owners[b]), std::max(owners[a], owners[b]));
  std::sort(conflicts.begin(), conflicts.end());
  conflicts.erase(std::unique(conflicts.begin(), conflicts.end()), conflicts.end());
  std::vector<unsigned int> comp(nH);
  std::iota(comp.begin(), comp.end(), 0u);
  auto find = [&comp](unsigned int x) {
    while (comp[x] != x)
      x = comp[x] = comp[comp[x]];
    return x;
  };
  // Track-only hypotheses (type 6, no footprint) do not enter the component sizes, as in the producer.
  auto trackOnly = [&](unsigned int h) { return type[h] == 6 && lcOff[h + 1] == lcOff[h]; };
  for (auto const& [a, b] : conflicts) {
    if (trackOnly(a) || trackOnly(b))
      continue;
    const auto ra = find(a), rb = find(b);
    if (ra != rb)
      comp[std::max(ra, rb)] = std::min(ra, rb);
  }
  std::vector<int> compSize(nH, 0);
  for (unsigned int h = 0; h < nH; ++h)
    ++compSize[find(h)];

  pEvent_ = event_;
  pLumi_ = lumi_;
  for (auto const& [a, b] : conflicts) {
    pI_ = a;
    pJ_ = b;
    pairs_->Fill();
  }

  for (unsigned int h = 0; h < nH; ++h) {
    idx_ = h;
    type_ = type[h];
    track_ = track[h];
    gsf_ = -1;  // no GSF input
    accepted_ = acc[h];
    nLC_ = lcOff[h + 1] - lcOff[h];
    nJetTracks_ = jetOff[h + 1] - jetOff[h];
    // Tracks of the hypothesis that also have a muon or an electron hypothesis.
    nMuonTracks_ = nElectronTracks_ = 0;
    auto countLeptonHyps = [&](int t) {
      if (t < 0)
        return;
      nMuonTracks_ += muonTrack.count(t);
      nElectronTracks_ += electronTrack.count(t);
    };
    if (nJetTracks_ > 0)
      for (int k = jetOff[h]; k < jetOff[h + 1]; ++k)
        countLeptonHyps(jetTk[k]);
    else
      countLeptonHyps(track_);
    score_ = score[h];
    weight_ = h < weight.size() ? weight[h] : -999.f;
    std::copy_n(features.begin() + h * kNx, kNx, x_);
    rawE_ = rawE[h];
    rawEmE_ = rawEmE[h];
    for (int k = 0; k < 8; ++k)
      pid_[k] = pid[8 * h + k];
    const math::XYZVectorF bary(xs[h], ys[h], zs[h]);
    eta_ = rawE_ > 0.f ? bary.eta() : -99.f;
    phi_ = rawE_ > 0.f ? bary.phi() : -99.f;
    comp_ = find(h);
    compSize_ = compSize[comp_];

    // Footprint: dominant truth object and its fraction of the footprint.
    const auto c = composition(h);
    double tot = 0.;
    int domObj = -1;
    double domE = 0.;
    for (auto const& [obj, e] : c) {
      tot += e;
      if (e > domE) {
        domE = e;
        domObj = obj;
      }
    }
    purity_ = tot > 0. ? domE / tot : -1.f;
    species_ = species(h, speciesFrac_);
    double neutralE = 0.;
    for (auto const& [obj, e] : c)
      if (isNeutral(pdgOf(obj)))
        neutralE += e;
    neutralFrac_ = tot > 0. ? neutralE / tot : 0.f;
    objPdg_ = pdgOf(domObj);
    signal_ = (domObj >= 0 && domObj < static_cast<int>(nP) && graph.particles()[domObj].isSignal()) ? 1 : 0;

    // Track: kinematics, truth object, and whether the footprint is dominated by the same object.
    trackP_ = trackPt_ = trackEta_ = eOverP_ = dR_ = -1.f;
    completeness_ = 0.f;
    trackPurity_ = 0.f;
    int tObj = -1;
    if (track_ >= 0 && track_ < static_cast<int>(tracks.size())) {
      auto const& tk = tracks[track_];
      trackP_ = tk.p();
      trackPt_ = tk.pt();
      trackEta_ = tk.eta();
      eOverP_ = rawE_ > 0.f && trackP_ > 0.f ? rawE_ / trackP_ : -1.f;
      const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
      dR_ = rawE_ > 0.f ? reco::deltaR(dir.eta(), dir.phi(), eta_, phi_) : -1.f;
      tObj = trackObject(track_, trackPurity_);
      completeness_ = rawE_ > 0.f ? completeness(h, trackParticle(track_)) : 0.f;
    } else if (nJetTracks_ > 0) {
      // Jet: scalar sums of p and pt, direction of the summed outer momenta.
      math::XYZVector sum;
      trackP_ = trackPt_ = 0.f;
      for (int k = jetOff[h]; k < jetOff[h + 1]; ++k) {
        auto const& tk = tracks[jetTk[k]];
        trackP_ += tk.p();
        trackPt_ += tk.pt();
        sum += tk.outerOk() ? tk.outerMomentum() : tk.momentum();
      }
      trackEta_ = sum.eta();
      eOverP_ = rawE_ > 0.f && trackP_ > 0.f ? rawE_ / trackP_ : -1.f;
      dR_ = rawE_ > 0.f ? reco::deltaR(sum.eta(), sum.phi(), eta_, phi_) : -1.f;
    }
    trackObjPdg_ = pdgOf(tObj);
    trackSameObj_ = (tObj >= 0 && tObj == domObj) ? 1 : 0;
    objFrac_ = 0.f;

    // Label: the hypothesis type matches the species of its truth object, and its footprint (if any) is dominated by
    // that object, or for a charged hadron holds most of the energy of its track particle.
    const bool footOK = rawE_ <= 0.f || (purity_ >= 0.5f && (tObj < 0 || tObj == domObj));
    switch (type_) {
      case 0:  // Muon
        label_ = (std::abs(trackObjPdg_) == 13 && (rawE_ <= 0.f || tObj == domObj)) ? 1 : 0;
        break;
      case 1:  // Electron
        label_ = (std::abs(trackObjPdg_) == 11 && footOK && rawE_ > 0.f) ? 1 : 0;
        break;
      case 3:  // ChargedHadron
      case 6:  // RecoveryChargedHadron
        // The energy comes from the track and the excess becomes a neutral residual: the footprint must hold most of
        // the energy of the track particle, and may hold other energy.
        label_ = (isChargedHadron(trackObjPdg_) && (rawE_ <= 0.f || completeness_ >= 0.5f)) ? 1 : 0;
        break;
      case 2:  // Photon
        if (nJetTracks_ == 0) {
          label_ = ((objPdg_ == 111 || objPdg_ == 22) && purity_ >= 0.5f) ? 1 : 0;
        } else {
          // Converted photon: every track comes from the same photon, whose object dominates the footprint (a signal
          // leg holds the photon; a pileup footprint is dominated by the photon or one of its descendants).
          int photon = -1;
          bool sameConversion = true;
          for (int k = jetOff[h]; k < jetOff[h + 1]; ++k) {
            const int g = conversionPhoton(jetTk[k]);
            sameConversion &= g >= 0 && (photon < 0 || g == photon);
            photon = g;
          }
          label_ = (sameConversion && photon >= 0 && purity_ >= 0.5f && domObj >= 0 &&
                    (domObj == objectFor(photon) || fromPhoton(domObj, photon)))
                       ? 1
                       : 0;
        }
        break;
      case 4:  // NeutralHadron
        label_ = (isNeutralHadron(objPdg_) && purity_ >= 0.5f) ? 1 : 0;
        break;
      case 5: {  // Jet: at least two tracks from distinct charged hadrons. Their objects hold a quarter of the footprint,
        // and with the neutral objects (the neutral residual) at least half of it. No member track is a lepton.
        std::vector<int> objs;
        bool lepton = false;
        for (int k = jetOff[h]; k < jetOff[h + 1]; ++k) {
          float pur;
          const int o = trackObject(jetTk[k], pur);
          const int a = std::abs(pdgOf(o));
          lepton |= (o >= 0 && (a == 11 || a == 13));
          if (o >= 0 && isChargedHadron(pdgOf(o)))
            objs.push_back(o);
        }
        std::sort(objs.begin(), objs.end());
        objs.erase(std::unique(objs.begin(), objs.end()), objs.end());
        double held = 0.;
        for (int o : objs)
          if (c.count(o))
            held += c.at(o);
        objFrac_ = tot > 0. ? held / tot : 0.f;
        label_ = (!lepton && objs.size() >= 2 && objFrac_ >= 0.25f && objFrac_ + neutralFrac_ >= 0.5f) ? 1 : 0;
        break;
      }
      default:
        label_ = 0;
    }
    tree_->Fill();
  }

  // Neutral final tracksters: the inputs of the neutral species model, the type that the producer gave, and the truth
  // species and dominant object of the layer clusters.
  auto const& finalTs = event.get(finalT_);
  auto const& neutralIdx = event.get(neutralIdxT_);
  auto const& neutralPdg = event.get(neutralPdgT_);
  auto const& neutralFromPhoton = event.get(neutralFromPhotonT_);
  std::vector<int> lcList;
  std::array<float, ticl::modelfeatures::neutralSpecies::kN> speciesX;
  for (size_t k = 0; k < neutralIdx.size(); ++k) {
    const auto& ts = finalTs[neutralIdx[k]];
    nIdx_ = neutralIdx[k];
    nPred_ = neutralPdg[k];
    nFromPhoton_ = neutralFromPhoton[k];
    nRawE_ = ts.raw_energy();
    nRegE_ = ts.regressed_energy();
    speciesX.fill(0.f);
    ticl::modelfeatures::fillNeutralSpeciesFeatures(ts, speciesX);
    std::copy(speciesX.begin(), speciesX.end(), nX_);
    lcList.assign(ts.vertices().begin(), ts.vertices().end());
    nSpecies_ = speciesOf(lcList, nSpeciesFrac_);
    const auto c = compositionOf(lcList);
    double tot = 0., domE = 0.;
    int domObj = -1;
    for (auto const& [obj, e] : c) {
      tot += e;
      if (e > domE) {
        domE = e;
        domObj = obj;
      }
    }
    nPurity_ = tot > 0. ? domE / tot : -1.f;
    nObjPdg_ = pdgOf(domObj);
    nSignal_ = (domObj >= 0 && domObj < static_cast<int>(nP) && graph.particles()[domObj].isSignal()) ? 1 : 0;
    neutrals_->Fill();
  }
}

DEFINE_FWK_MODULE(HypothesisTruthDumper);
