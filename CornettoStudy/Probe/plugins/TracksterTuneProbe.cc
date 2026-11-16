// Study probe, not for integration. The trackster-level metrics of one parameter set of the HGCAL chain, from the
// truth-graph trackster associators.
// Tree "legs": one row per signal member of the truth level with pT > minPt in minAbsEta < |eta| < maxAbsEta, and per
// trackster collection (stage): the trackster with the largest shared fraction S of the leg, its reco-to-truth score
// R, the sum of S over all tracksters, the raw energy of the best trackster, and the rechit energy of the leg (the
// shared-fraction denominator: per HGCAL cell, the simulated share of the leg times the rechit energy).
// Tree "tracksters": one row per trackster of the last collection: raw energy, |eta|, the lowest reco-to-truth score
// and whether that truth branch is a signal particle.
// Tree "jets": one row per anti-kT jet (radius jetR) of the signal reconstructableFinalState members with pT > minJetPt
// in minJetAbsEta < |eta| < maxJetAbsEta: the jet momentum, the rechit energy of the constituents (sum of the leg
// rechit energies), the energy that the tracksters of the last collection share with the constituents, the raw
// energy of the tracksters of the last collection with more than half of their energy from the constituents (reco
// purity, 1 - score of the Fixed reco-to-truth map), and the raw energy of the tracksters of the last collection with the barycenter within jetR of the jet axis
// (pileup included).

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

#include "TTree.h"

#include "fastjet/ClusterSequence.hh"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;

  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
}  // namespace

class TracksterTuneProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TracksterTuneProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  struct Stage {
    edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksters;
    edm::EDGetTokenT<MapType> truthToReco, recoToTruth;
  };
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<unsigned int>> targetsToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> recHitTokens_;
  std::vector<Stage> stages_;
  const double minAbsEta_, maxAbsEta_, minPt_;
  const double jetR_, minJetPt_, minJetAbsEta_, maxJetAbsEta_;

  TTree* legs_;
  TTree* tracksters_;
  TTree* jets_;
  unsigned long long event_;
  int pdg_, stage_, best_;
  float pt_, eta_, energy_, eRecHit_, s_, r_, sumS_, eBest_;
  float tsEnergy_, tsEta_, tsScore_;
  int tsSignal_;
  float jetPt_, jetEta_, jetEnergy_, jetRecHit_, jetShared_, jetOwned_, jetCone_;
  int jetN_;
};

TracksterTuneProbe::TracksterTuneProbe(const edm::ParameterSet& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      targetsToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("targets"))),
      minAbsEta_(cfg.getParameter<double>("minAbsEta")),
      maxAbsEta_(cfg.getParameter<double>("maxAbsEta")),
      minPt_(cfg.getParameter<double>("minPt")),
      jetR_(cfg.getParameter<double>("jetR")),
      minJetPt_(cfg.getParameter<double>("minJetPt")),
      minJetAbsEta_(cfg.getParameter<double>("minJetAbsEta")),
      maxJetAbsEta_(cfg.getParameter<double>("maxJetAbsEta")) {
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("hgcalRecHits"))
    recHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  const auto assoc = cfg.getParameter<std::string>("associator");
  for (auto const& label : cfg.getParameter<std::vector<std::string>>("tracksters"))
    stages_.push_back({consumes<std::vector<ticl::Trackster>>(edm::InputTag(label)),
                       consumes<MapType>(edm::InputTag(assoc, label + "TruthToReco")),
                       consumes<MapType>(edm::InputTag(assoc, label + "RecoToTruthFixed"))});
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  legs_ = fs->make<TTree>("legs", "legs");
  legs_->Branch("event", &event_);
  legs_->Branch("stage", &stage_);
  legs_->Branch("pdg", &pdg_);
  legs_->Branch("pt", &pt_);
  legs_->Branch("eta", &eta_);
  legs_->Branch("energy", &energy_);
  legs_->Branch("eRecHit", &eRecHit_);
  legs_->Branch("best", &best_);
  legs_->Branch("S", &s_);
  legs_->Branch("R", &r_);
  legs_->Branch("sumS", &sumS_);
  legs_->Branch("eBest", &eBest_);
  tracksters_ = fs->make<TTree>("tracksters", "tracksters");
  tracksters_->Branch("event", &event_);
  tracksters_->Branch("energy", &tsEnergy_);
  tracksters_->Branch("eta", &tsEta_);
  tracksters_->Branch("score", &tsScore_);
  tracksters_->Branch("signal", &tsSignal_);
  jets_ = fs->make<TTree>("jets", "jets");
  jets_->Branch("event", &event_);
  jets_->Branch("pt", &jetPt_);
  jets_->Branch("eta", &jetEta_);
  jets_->Branch("energy", &jetEnergy_);
  jets_->Branch("n", &jetN_);
  jets_->Branch("eRecHit", &jetRecHit_);
  jets_->Branch("eShared", &jetShared_);
  jets_->Branch("eOwned", &jetOwned_);
  jets_->Branch("eCone", &jetCone_);
}

void TracksterTuneProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& graph = event.get(graphToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& targets = event.get(targetsToken_);
  const uint32_t nP = graph.nParticles();
  event_ = event.id().event();

  std::unordered_map<uint32_t, float> recHitE;
  for (auto const& token : recHitTokens_)
    for (auto const& h : event.get(token))
      recHitE[h.id().rawId()] += h.energy();
  std::unordered_map<uint32_t, float> cellSim;
  for (uint32_t p = 0; p < nP; ++p)
    for (auto const& h : index.directHits(truth::HitChannel::Calo, p))
      if (isHGCal(h.detId))
        cellSim[h.detId] += h.energy;

  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  std::unordered_map<uint32_t, float> mine;
  // The shared-fraction denominator of a truth branch: per HGCAL cell, its simulated share times the rechit energy.
  auto recHitEnergy = [&](uint32_t b) {
    hits.clear();
    index.appendSubgraphHits(truth::HitChannel::Calo, b, hits);
    mine.clear();
    for (auto const& h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f)
        mine[h.detId] += h.energy;
    float e = 0.f;
    for (auto const& [id, en] : mine) {
      auto const r = recHitE.find(id);
      auto const s = cellSim.find(id);
      if (r != recHitE.end() && s != cellSim.end() && s->second > 0.f)
        e += std::min(1.f, en / s->second) * r->second;
    }
    return e;
  };
  for (const uint32_t b : targets) {
    if (b >= nP || truth::isInvisible(graph.particles()[b].pdgId))
      continue;
    auto const& pd = graph.particles()[b];
    const double aeta = std::abs(pd.momentum.eta());
    if (!pd.isSignal() || pd.momentum.pt() < minPt_ || aeta < minAbsEta_ || aeta > maxAbsEta_)
      continue;
    pdg_ = pd.pdgId;
    pt_ = pd.momentum.pt();
    eta_ = pd.momentum.eta();
    energy_ = pd.momentum.E();
    eRecHit_ = recHitEnergy(b);
    for (std::size_t k = 0; k < stages_.size(); ++k) {
      auto const& ts = event.get(stages_[k].tracksters);
      auto const& t2r = event.get(stages_[k].truthToReco).getMap();
      auto const& r2t = event.get(stages_[k].recoToTruth).getMap();
      stage_ = static_cast<int>(k);
      best_ = -1;
      s_ = sumS_ = 0.f;
      r_ = 1.f;
      eBest_ = 0.f;
      if (b < t2r.size())
        for (auto const& m : t2r[b]) {
          sumS_ += m.sharedEnergy();
          if (m.sharedEnergy() > s_) {
            s_ = m.sharedEnergy();
            best_ = static_cast<int>(m.index());
          }
        }
      if (best_ >= 0) {
        eBest_ = ts[best_].raw_energy();
        if (static_cast<std::size_t>(best_) < r2t.size())
          for (auto const& m : r2t[best_])
            if (m.index() == b)
              r_ = m.score();
      }
      legs_->Fill();
    }
  }

  if (stages_.empty())
    return;
  auto const& ts = event.get(stages_.back().tracksters);
  auto const& r2t = event.get(stages_.back().recoToTruth).getMap();
  for (std::size_t t = 0; t < ts.size(); ++t) {
    tsEnergy_ = ts[t].raw_energy();
    tsEta_ = std::abs(ts[t].barycenter().eta());
    tsScore_ = 1.f;
    tsSignal_ = 0;
    if (t < r2t.size())
      for (auto const& m : r2t[t])
        if (m.score() < tsScore_) {
          tsScore_ = m.score();
          tsSignal_ = m.index() < nP && graph.particles()[m.index()].isSignal() ? 1 : 0;
        }
    tracksters_->Fill();
  }

  std::vector<fastjet::PseudoJet> inputs;
  for (uint32_t p = 0; p < nP; ++p) {
    auto const& pd = graph.particles()[p];
    if (!pd.isSignal() || !pd.isAtLevel(truth::LevelFlag::ReconstructableFinalState))
      continue;
    fastjet::PseudoJet pj(pd.momentum.px(), pd.momentum.py(), pd.momentum.pz(), pd.momentum.e());
    pj.set_user_index(static_cast<int>(p));
    inputs.push_back(pj);
  }
  fastjet::ClusterSequence cs(inputs, fastjet::JetDefinition(fastjet::antikt_algorithm, jetR_));
  auto const& t2r = event.get(stages_.back().truthToReco).getMap();
  std::unordered_map<uint32_t, std::size_t> jetOf;
  std::vector<fastjet::PseudoJet> jets;
  for (auto const& j : cs.inclusive_jets(minJetPt_))
    if (std::abs(j.eta()) >= minJetAbsEta_ && std::abs(j.eta()) <= maxJetAbsEta_) {
      for (auto const& c : j.constituents())
        jetOf[static_cast<uint32_t>(c.user_index())] = jets.size();
      jets.push_back(j);
    }
  std::vector<float> owned(jets.size(), 0.f);
  std::vector<float> purity(jets.size());
  for (std::size_t t = 0; t < ts.size() && t < r2t.size(); ++t) {
    std::fill(purity.begin(), purity.end(), 0.f);
    for (auto const& m : r2t[t])
      if (auto const it = jetOf.find(m.index()); it != jetOf.end())
        purity[it->second] += 1.f - m.score();
    for (std::size_t k = 0; k < jets.size(); ++k)
      if (purity[k] > 0.5f)
        owned[k] += ts[t].raw_energy();
  }
  for (std::size_t k = 0; k < jets.size(); ++k) {
    auto const& j = jets[k];
    jetOwned_ = owned[k];
    jetPt_ = j.pt();
    jetEta_ = j.eta();
    jetEnergy_ = j.e();
    jetRecHit_ = jetShared_ = jetCone_ = 0.f;
    auto const constituents = j.constituents();
    jetN_ = static_cast<int>(constituents.size());
    for (auto const& c : constituents) {
      const auto b = static_cast<uint32_t>(c.user_index());
      const float e = recHitEnergy(b);
      jetRecHit_ += e;
      if (b < t2r.size())
        for (auto const& m : t2r[b])
          jetShared_ += m.sharedEnergy() * e;
    }
    for (auto const& t : ts) {
      const double dphi = std::remainder(t.barycenter().phi() - j.phi_std(), 2. * std::numbers::pi);
      const double deta = t.barycenter().eta() - j.eta();
      if (deta * deta + dphi * dphi < jetR_ * jetR_)
        jetCone_ += t.raw_energy();
    }
    jets_->Fill();
  }
}

DEFINE_FWK_MODULE(TracksterTuneProbe);
