// Study probe, not for integration. The simulated HGCAL energy of truth jets in the final tracksters and their
// candidates. A truth jet is an anti-kT jet (radius jetR) of the signal reconstructableFinalState members of the truth
// graph (a pi0 is one object, no neutrinos); its energy is the subgraph energy of its constituents. The partonJets
// subgraphs are not used: they hold the string fragments up to the beam remnants (|eta| 6 to 7 on QCD).
//   jets:  one row per jet with |eta| in [minAbsEta, maxAbsEta] and pT >= minJetPt: the momentum, the constituent
//          energy by species (EM: photon, electron, pi0, eta; neutral hadron; charged with a track; charged without a
//          track), the HGCAL deposit by species of the calo-boundary particle, and the part of each in a final trackster.
//   ts:    one row per final trackster with energy from a selected jet: energy, PID, fate (class of the interpretation:
//          0 neutral candidate, 1 neutral without candidate, 2 tracks and a neutral residual, 3 tracks only, 4 claim
//          trackster), the energy of its neutral candidates (N) and of its charged candidates (Q), and its footprint
//          truth energy split into signal not in a selected jet (with a track or not) and pileup.
//   pairs: one row per (jet, trackster) with the shared energy by species.
// A particle in the subgraphs of the constituents of n jets gives 1/n of its energy to each. "With a track": the
// particle, or an ancestor, is the best Fixed truth match of the track of a charged candidate (the energy that the track
// already carries).
// With assocStudy: the jet deposit in cells with a rechit and in layer clusters of 1 hit and of 2 or more hits, and per
// trackster the AdaptiveNominal truth root and its footprint share with three cell weights: the rechit energy of every
// cell (A, as the production associator), only the HGCAL cells in layer clusters of 2 or more hits plus the barrel (B),
// and only those HGCAL cells (C).

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/Branch.h"
#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/RecoHitAdapters.h"

#include "TTree.h"
#include "fastjet/ClusterSequence.hh"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
  // Species of a particle: 0 EM (photon, pi0, eta, electron without a track), 1 neutral hadron, 2 charged with a track,
  // 3 charged without a track. A muon counts as charged.
  enum Species { kEM = 0, kH0 = 1, kChTrk = 2, kChNoTrk = 3, kNSpecies = 4 };
  // trackMode value of a jet member (ticl::TrackMode::kJetMember, RecoTICL/Interpretation/interface/AssignmentMaps.h)
  constexpr int kJetMember = 4;
}  // namespace

class JetTruthProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit JetTruthProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken_;
  const edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  const edm::EDGetTokenT<std::vector<int>> trackModeToken_;
  const edm::EDGetTokenT<std::vector<int>> trackToTracksterToken_;
  const edm::EDGetTokenT<std::vector<int>> neutralIdxToken_;
  const edm::EDGetTokenT<MapType> tkR2tToken_;
  const double minAbsEta_, maxAbsEta_, minJetPt_, jetR_;
  const bool assocStudy_;
  edm::EDGetTokenT<std::vector<unsigned int>> selectedRootsToken_, assignableRootsToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcalRecHitTokens_;
  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> pfRecHitTokens_;
  edm::EDGetTokenT<MapType> tsR2tToken_;
  static constexpr int kNWeights = 3;

  TTree *jetTree_, *tsTree_, *pairTree_;
  int event_, lumi_, jet_, nConst_, ts_, cls_, hadronic_;
  float jetPt_, jetEta_, jetPhi_, jetE_;
  float constE_[kNSpecies], dep_[kNSpecies], depTs_[kNSpecies];
  float tsE_, tsRawE_, tsEta_, tsPhi_, tsN_, tsQ_, footE_, sigOtherTrk_, sigOtherNoTrk_, puE_, pidEM_;
  float share_[kNSpecies];
  float depRH_[kNSpecies], depLC2_[kNSpecies], depLC1_[kNSpecies];
  int rootProd_, root_[kNWeights];
  float target_[kNWeights], trkE_;
};

JetTruthProbe::JetTruthProbe(const edm::ParameterSet& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      tracksterToken_(consumes<std::vector<ticl::Trackster>>(cfg.getParameter<edm::InputTag>("tracksters"))),
      candToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("candidates"))),
      trackModeToken_(consumes<std::vector<int>>(cfg.getParameter<edm::InputTag>("trackMode"))),
      trackToTracksterToken_(consumes<std::vector<int>>(cfg.getParameter<edm::InputTag>("trackToTrackster"))),
      neutralIdxToken_(consumes<std::vector<int>>(cfg.getParameter<edm::InputTag>("neutralIdx"))),
      tkR2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      minAbsEta_(cfg.getParameter<double>("minAbsEta")),
      maxAbsEta_(cfg.getParameter<double>("maxAbsEta")),
      minJetPt_(cfg.getParameter<double>("minJetPt")),
      jetR_(cfg.getParameter<double>("jetR")),
      assocStudy_(cfg.getUntrackedParameter<bool>("assocStudy", false)) {
  if (assocStudy_) {
    selectedRootsToken_ = consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("selectedRoots"));
    assignableRootsToken_ = consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("assignableRoots"));
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("hgcalRecHits"))
      hgcalRecHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
    for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("pfRecHits"))
      pfRecHitTokens_.push_back(consumes<reco::PFRecHitCollection>(tag));
    tsR2tToken_ = consumes<MapType>(cfg.getParameter<edm::InputTag>("tracksterRecoToTruth"));
  }
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  jetTree_ = fs->make<TTree>("jets", "anti-kT truth jets and their HGCAL deposit");
  tsTree_ = fs->make<TTree>("ts", "final tracksters with energy from a selected jet");
  pairTree_ = fs->make<TTree>("pairs", "shared energy of a jet and a final trackster by species");
  for (auto* t : {jetTree_, tsTree_, pairTree_}) {
    t->Branch("event", &event_);
    t->Branch("lumi", &lumi_);
  }
  jetTree_->Branch("jet", &jet_);
  jetTree_->Branch("nConst", &nConst_);
  jetTree_->Branch("pt", &jetPt_);
  jetTree_->Branch("eta", &jetEta_);
  jetTree_->Branch("phi", &jetPhi_);
  jetTree_->Branch("e", &jetE_);
  jetTree_->Branch("constE", constE_, "constE[4]/F");
  jetTree_->Branch("dep", dep_, "dep[4]/F");
  jetTree_->Branch("depTs", depTs_, "depTs[4]/F");
  jetTree_->Branch("depRH", depRH_, "depRH[4]/F");
  jetTree_->Branch("depLC2", depLC2_, "depLC2[4]/F");
  jetTree_->Branch("depLC1", depLC1_, "depLC1[4]/F");
  tsTree_->Branch("ts", &ts_);
  tsTree_->Branch("cls", &cls_);
  tsTree_->Branch("hadronic", &hadronic_);
  tsTree_->Branch("e", &tsE_);
  tsTree_->Branch("rawE", &tsRawE_);
  tsTree_->Branch("eta", &tsEta_);
  tsTree_->Branch("phi", &tsPhi_);
  tsTree_->Branch("pidEM", &pidEM_);
  tsTree_->Branch("N", &tsN_);
  tsTree_->Branch("Q", &tsQ_);
  tsTree_->Branch("footE", &footE_);
  tsTree_->Branch("sigOtherTrk", &sigOtherTrk_);
  tsTree_->Branch("sigOtherNoTrk", &sigOtherNoTrk_);
  tsTree_->Branch("puE", &puE_);
  tsTree_->Branch("trkE", &trkE_);
  tsTree_->Branch("rootProd", &rootProd_);
  tsTree_->Branch("root", root_, "root[3]/I");
  tsTree_->Branch("target", target_, "target[3]/F");
  pairTree_->Branch("jet", &jet_);
  pairTree_->Branch("ts", &ts_);
  pairTree_->Branch("share", share_, "share[4]/F");
}

void JetTruthProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& layerClusters = event.get(lcToken_);
  auto const& tracksters = event.get(tracksterToken_);
  auto const& cands = event.get(candToken_);
  auto const& trackMode = event.get(trackModeToken_);
  auto const& trackToTrackster = event.get(trackToTracksterToken_);
  auto const& neutralIdx = event.get(neutralIdxToken_);
  auto const& tkR2t = event.get(tkR2tToken_).getMap();
  event_ = event.id().event();
  lumi_ = event.id().luminosityBlock();
  const unsigned int nP = graph.particles().size();

  // Particles whose energy a charged candidate's track carries: the best Fixed match of the track and its descendants.
  std::vector<bool> withTrack(nP, false);
  for (auto const& c : cands) {
    if (c.charge() == 0 || c.trackPtr().isNull())
      continue;
    const auto key = c.trackPtr().key();
    if (key >= tkR2t.size() || tkR2t[key].empty())
      continue;
    auto best = std::min_element(
        tkR2t[key].begin(), tkR2t[key].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
    if (best->index() >= nP || withTrack[best->index()])
      continue;
    withTrack[best->index()] = true;
    for (auto const& d : graph.particle(best->index()).descendants())
      withTrack[d.id()] = true;
  }
  auto speciesOf = [&](unsigned int p) {
    const int a = std::abs(graph.particles()[p].pdgId);
    if (a == 22 || a == 111 || a == 221)
      return kEM;
    if (a == 11)
      return withTrack[p] ? kChTrk : kEM;
    if (std::abs(graph.particle(p).charge()) > 0.5)
      return withTrack[p] ? kChTrk : kChNoTrk;
    return kH0;
  };

  // Truth jets: anti-kT jets of the signal reconstructableFinalState members.
  std::vector<fastjet::PseudoJet> inputs;
  for (unsigned int p = 0; p < nP; ++p) {
    auto const& pd = graph.particles()[p];
    if (!pd.isSignal() || !pd.isAtLevel(truth::LevelFlag::ReconstructableFinalState))
      continue;
    fastjet::PseudoJet pj(pd.momentum.px(), pd.momentum.py(), pd.momentum.pz(), pd.momentum.e());
    pj.set_user_index(static_cast<int>(p));
    inputs.push_back(pj);
  }
  fastjet::ClusterSequence cs(inputs, fastjet::JetDefinition(fastjet::antikt_algorithm, jetR_));
  std::vector<fastjet::PseudoJet> jets;
  for (auto const& j : fastjet::sorted_by_pt(cs.inclusive_jets(minJetPt_)))
    if (std::abs(j.eta()) >= minAbsEta_ && std::abs(j.eta()) <= maxAbsEta_)
      jets.push_back(j);
  if (jets.empty())
    return;
  // Hit count of the layer cluster of each HGCAL cell; a cell in no layer cluster is absent.
  std::unordered_map<uint32_t, int> cellLC;
  for (auto const& lc : layerClusters) {
    const int n = lc.hitsAndFractions().size();
    for (auto const& [id, frac] : lc.hitsAndFractions()) {
      auto& v = cellLC[id.rawId()];
      v = std::max(v, n);
    }
  }
  auto lcHits = [&cellLC](uint32_t id) {
    auto it = cellLC.find(id);
    return it == cellLC.end() ? 0 : it->second;
  };
  // Cell weights of the three associations, and the HGCAL cells with a rechit.
  std::array<truth::CellEnergyTable, kNWeights> tables;
  std::unordered_set<uint32_t> rechitCells;
  if (assocStudy_) {
    for (auto const& token : pfRecHitTokens_)
      for (auto const& hit : event.get(token)) {
        tables[0].add(hit.detId(), hit.energy());
        tables[1].add(hit.detId(), hit.energy());
      }
    for (auto const& token : hgcalRecHitTokens_)
      for (auto const& hit : event.get(token)) {
        const uint32_t id = hit.id().rawId();
        rechitCells.insert(id);
        tables[0].add(id, hit.energy());
        if (lcHits(id) >= 2) {
          tables[1].add(id, hit.energy());
          tables[2].add(id, hit.energy());
        }
      }
    for (auto& t : tables)
      t.finalize();
  }
  // Jets of each particle: the constituents and their descendants, for the 1/n split of a particle in two subgraphs.
  std::vector<std::vector<int>> jetsOf(nP);
  std::vector<std::array<double, kNSpecies>> constE(jets.size(), std::array<double, kNSpecies>{});
  for (unsigned int j = 0; j < jets.size(); ++j)
    for (auto const& c : jets[j].constituents()) {
      const auto id = static_cast<unsigned int>(c.user_index());
      constE[j][speciesOf(id)] += c.e();
      jetsOf[id].push_back(static_cast<int>(j));
      for (auto const& d : graph.particle(id).descendants())
        jetsOf[d.id()].push_back(static_cast<int>(j));
    }
  for (auto& js : jetsOf) {
    std::sort(js.begin(), js.end());
    js.erase(std::unique(js.begin(), js.end()), js.end());
  }

  // Simulated HGCAL energy per cell and calo-boundary particle, and the deposit of each jet by species.
  std::unordered_map<uint32_t, std::vector<std::pair<int, float>>> cellTruth;
  std::vector<std::array<double, kNSpecies>> dep(jets.size(), std::array<double, kNSpecies>{});
  std::vector<std::array<double, kNSpecies>> depRH(jets.size(), std::array<double, kNSpecies>{});
  std::vector<std::array<double, kNSpecies>> depLC2(jets.size(), std::array<double, kNSpecies>{});
  std::vector<std::array<double, kNSpecies>> depLC1(jets.size(), std::array<double, kNSpecies>{});
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  std::vector<int> species(nP, -1);
  for (unsigned int p = 0; p < nP; ++p) {
    auto const& pd = graph.particles()[p];
    if (!pd.isAtLevel(truth::LevelFlag::CaloBoundary))
      continue;
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    species[p] = speciesOf(p);
    for (auto const& h : hits) {
      if (!isHGCal(h.detId) || !(h.energy > 0.f))
        continue;
      cellTruth[h.detId].emplace_back(static_cast<int>(p), h.energy);
      const int nLC = lcHits(h.detId);
      const bool withRechit = rechitCells.count(h.detId) > 0;
      for (int j : jetsOf[p]) {
        const double w = h.energy / jetsOf[p].size();
        dep[j][species[p]] += w;
        depRH[j][species[p]] += withRechit ? w : 0.;
        depLC2[j][species[p]] += nLC >= 2 ? w : 0.;
        depLC1[j][species[p]] += nLC == 1 ? w : 0.;
      }
    }
  }

  // Fate of the final tracksters: neutral and charged candidate energy, tracks, neutralIdx.
  const unsigned int nTs = tracksters.size();
  std::vector<double> N(nTs, 0.), Q(nTs, 0.);
  std::vector<int> nTrk(nTs, 0);
  std::vector<bool> isNeutral(nTs, false);
  for (int k : neutralIdx)
    if (k >= 0 && static_cast<unsigned int>(k) < nTs)
      isNeutral[k] = true;
  for (auto const& c : cands) {
    auto const& tsRefs = c.tracksters();
    if (c.charge() == 0) {
      for (auto const& t : tsRefs)
        N[t.key()] += c.energy() / tsRefs.size();
      continue;
    }
    int k = -1;
    if (!tsRefs.empty()) {
      k = static_cast<int>(tsRefs[0].key());
    } else if (c.trackPtr().isNonnull()) {
      const auto tk = c.trackPtr().key();
      if (tk < trackMode.size() && trackMode[tk] == kJetMember)
        k = trackToTrackster[tk];
    }
    if (k >= 0 && static_cast<unsigned int>(k) < nTs) {
      Q[k] += c.energy();
      ++nTrk[k];
    }
  }

  // The three AdaptiveNominal associations on the production candidate and assignable roots.
  std::vector<uint32_t> generations;
  std::vector<std::unique_ptr<truth::BranchHitAssociator>> associators;
  std::vector<uint8_t> isAssignable;
  MapType const* tsMap = nullptr;
  if (assocStudy_) {
    generations = truth::particleGenerations(graph);
    auto const& selected = event.get(selectedRootsToken_);
    const std::vector<uint32_t> roots(selected.begin(), selected.end());
    const uint32_t hgcalMask = (1u << DetId::HGCalEE) | (1u << DetId::HGCalHSi) | (1u << DetId::HGCalHSc);
    for (auto const& t : tables)
      associators.push_back(
          std::make_unique<truth::BranchHitAssociator>(index,
                                                       roots,
                                                       truth::BranchHitAssociator::Metric::SharedEnergy,
                                                       truth::HitChannel::Calo,
                                                       false,
                                                       hgcalMask,
                                                       &t,
                                                       generations));
    isAssignable.assign(nP, 0);
    for (const unsigned int id : event.get(assignableRootsToken_))
      if (id < nP)
        isAssignable[id] = 1;
    tsMap = &event.get(tsR2tToken_);
  }
  std::vector<truth::BranchMatch> assignable;
  std::unordered_map<uint32_t, float> rootCells;

  // Shared energy of each final trackster with each jet by species, and its other truth.
  std::vector<std::array<double, kNSpecies>> depTs(jets.size(), std::array<double, kNSpecies>{});
  std::unordered_map<int, std::array<double, kNSpecies>> perJet;
  for (unsigned int k = 0; k < nTs; ++k) {
    auto const& ts = tracksters[k];
    perJet.clear();
    double foot = 0., otherTrk = 0., otherNoTrk = 0., pu = 0., trk = 0.;
    for (size_t i = 0; i < ts.vertices().size(); ++i) {
      const auto lc = ts.vertices()[i];
      if (lc >= layerClusters.size())
        continue;
      const float share = 1.f / std::max<float>(1.f, ts.vertex_multiplicity()[i]);
      for (auto const& [id, frac] : layerClusters[lc].hitsAndFractions()) {
        auto it = cellTruth.find(id.rawId());
        if (it == cellTruth.end())
          continue;
        for (auto const& [p, e] : it->second) {
          const double w = share * frac * e;
          foot += w;
          trk += withTrack[p] ? w : 0.;
          if (!jetsOf[p].empty()) {
            for (int j : jetsOf[p])
              perJet[j][species[p]] += w / jetsOf[p].size();
          } else if (graph.particles()[p].isSignal()) {
            (species[p] == kChTrk ? otherTrk : otherNoTrk) += w;
          } else {
            pu += w;
          }
        }
      }
    }
    if (perJet.empty())
      continue;
    ts_ = static_cast<int>(k);
    if (nTrk[k] > 0)
      cls_ = N[k] > 0. ? 2 : 3;
    else if (isNeutral[k])
      cls_ = N[k] > 0. ? 0 : 1;
    else
      cls_ = 4;
    hadronic_ = ts.isHadronic() ? 1 : 0;
    tsE_ = ts.regressed_energy();
    tsRawE_ = ts.raw_energy();
    tsEta_ = ts.barycenter().eta();
    tsPhi_ = ts.barycenter().phi();
    pidEM_ = ts.id_probabilities(0) + ts.id_probabilities(1);
    tsN_ = N[k];
    tsQ_ = Q[k];
    footE_ = foot;
    sigOtherTrk_ = otherTrk;
    sigOtherNoTrk_ = otherNoTrk;
    puE_ = pu;
    trkE_ = trk;
    rootProd_ = -1;
    std::fill(root_, root_ + kNWeights, -1);
    std::fill(target_, target_ + kNWeights, -1.f);
    if (assocStudy_) {
      auto const& row = tsMap->getMap()[k];
      if (!row.empty())
        rootProd_ = std::min_element(row.begin(), row.end(), [](auto const& a, auto const& b) {
                      return a.score() < b.score();
                    })->index();
      const auto recoHits = truth::recoHits(ts, layerClusters);
      for (int v = 0; v < kNWeights; ++v) {
        assignable.clear();
        for (auto const& m : associators[v]->bestBranches(recoHits))
          if (m.rootParticleId < nP && isAssignable[m.rootParticleId] != 0)
            assignable.push_back(m);
        const auto best =
            truth::BranchHitAssociator::bestAdaptiveBranch(std::span<const truth::BranchMatch>(assignable), 1.f, 1.f);
        if (best.rootParticleId == truth::BranchMatch::kInvalidRoot)
          continue;
        root_[v] = static_cast<int>(best.rootParticleId);
        if (v > 0 && root_[v] == root_[v - 1]) {
          target_[v] = target_[v - 1];
          continue;
        }
        // Footprint share of the root: its subgraph energy in the cells of the trackster over the footprint energy.
        rootCells.clear();
        hits.clear();
        index.appendSubgraphHits(truth::HitChannel::Calo, best.rootParticleId, hits);
        for (auto const& h : hits)
          if (isHGCal(h.detId) && h.energy > 0.f)
            rootCells[h.detId] += h.energy;
        double rootFoot = 0.;
        for (size_t i = 0; i < ts.vertices().size(); ++i) {
          const auto lc = ts.vertices()[i];
          if (lc >= layerClusters.size())
            continue;
          const float share = 1.f / std::max<float>(1.f, ts.vertex_multiplicity()[i]);
          for (auto const& [id, frac] : layerClusters[lc].hitsAndFractions())
            if (auto it = rootCells.find(id.rawId()); it != rootCells.end())
              rootFoot += share * frac * it->second;
        }
        target_[v] = foot > 0. ? rootFoot / foot : -1.f;
      }
    }
    tsTree_->Fill();
    for (auto const& [j, s] : perJet) {
      jet_ = j;
      for (int x = 0; x < kNSpecies; ++x) {
        share_[x] = s[x];
        depTs[j][x] += s[x];
      }
      pairTree_->Fill();
    }
  }
  for (unsigned int j = 0; j < jets.size(); ++j) {
    jet_ = static_cast<int>(j);
    nConst_ = static_cast<int>(jets[j].constituents().size());
    jetPt_ = jets[j].pt();
    jetEta_ = jets[j].eta();
    jetPhi_ = jets[j].phi_std();
    jetE_ = jets[j].e();
    for (int x = 0; x < kNSpecies; ++x) {
      constE_[x] = constE[j][x];
      dep_[x] = dep[j][x];
      depTs_[x] = depTs[j][x];
      depRH_[x] = assocStudy_ ? depRH[j][x] : -1.f;
      depLC2_[x] = depLC2[j][x];
      depLC1_[x] = depLC1[j][x];
    }
    jetTree_->Fill();
  }
}

DEFINE_FWK_MODULE(JetTruthProbe);
