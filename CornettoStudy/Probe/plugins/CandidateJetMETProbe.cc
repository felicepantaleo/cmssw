// Study probe, not for integration. Jets and missing transverse momentum from TICL candidates in the HGCAL endcap,
// against the stable visible signal generator particles in the same region. Reco is computed twice: all candidates,
// and charged candidates only from the signal vertex (|track vz - signal vertex z| < maxDz) with all neutrals.
// Tree jets: one row per generator jet with the matched reco jets and their pileup offset, the median pT in cones of the
// jet radius at the jet eta, rotated in phi by multiples of pi/4, away from every generator jet.
// Tree tjets: one row per truth jet, clustered from the calo-boundary particles of the truth graph (signal and in-time
// pileup, an antichain), with the matched reco jet of all candidates; both sides hold the pileup, so no offset is needed.
// Tree met: one row per event and endcap side, with the generator (signal) and the truth-graph (signal and in-time
// pileup) sums, the out-of-time truth pT sum, the controls (signal truth pT sum, interactions, Signal-level count),
// the highest-pT candidate with its track, its tracksters and the truth pT within 0.1 of it, and the neutral pT split by
// the truth of its tracksters: double counted (energy of particles whose track a charged candidate holds; for a residual
// neutral only its excess over the untracked share), split by residual / jet member / other track-only / with-trackster
// charged candidate,
// and from charged particles with no such track.
// With assignableRoots set, also the same sums with the deep track match (DeepTrackMatch.h) and the truth energy of a
// layer cluster divided by its vertex multiplicity: over-kept and under-kept neutral pT, max(0, E - Ets (1 - d)) pT / E
// and max(0, Ets (1 - d) - E) pT / E, for neutral candidates with trackster pT below 10 GeV; and the double-counted pT
// split at a trackster pT of 10 GeV, for both matches. For the closure of O and U, per side: the pT of all those
// neutral candidates, of those with no deep truth energy, the pT of the kept target Ets (1 - d) pT / E and the number
// of candidates with deep truth energy.
// With pfCandidates set, column 2 of the reco jets and of the reco missing transverse momentum uses the PF candidates
// in the same region with the PUPPI weights: puppiWeightsJets for the jets, puppiWeightsMet for the missing
// transverse momentum. Without it, column 2 holds -1.

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "CornettoStudy/Probe/plugins/DeepTrackMatch.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/InteractionId.h"

#include "fastjet/ClusterSequence.hh"
#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
  // Composition index of a particle: 0 charged, 1 photon or pi0 (EM), 2 neutral hadron, 3 other (lepton).
  int category(int pdg, int charge) {
    const int a = std::abs(pdg);
    if (a == 11 || a == 13)
      return 3;
    if (charge != 0)
      return 0;
    if (a == 22 || a == 111)
      return 1;
    return 2;
  }
  // Composition index from the pdg id alone (the truth graph stores no charge).
  int categoryOfPdg(int pdg) {
    const int a = std::abs(pdg);
    if (a == 22 || a == 111)
      return 1;
    if (a == 11 || a == 13)
      return 3;
    if (a == 130 || a == 2112 || a == 310 || a == 3122 || a == 3212 || a == 3322)
      return 2;
    return 0;
  }
  struct Jet {
    float pt, eta, phi, frac[3];
  };
  std::vector<Jet> cluster(const std::vector<fastjet::PseudoJet>& in, double r, double minPt) {
    std::vector<Jet> out;
    fastjet::ClusterSequence cs(in, fastjet::JetDefinition(fastjet::antikt_algorithm, r));
    for (auto const& j : fastjet::sorted_by_pt(cs.inclusive_jets(minPt))) {
      Jet o{static_cast<float>(j.pt()), static_cast<float>(j.eta()), static_cast<float>(j.phi_std()), {0.f, 0.f, 0.f}};
      for (auto const& c : j.constituents())
        if (c.user_index() < 3)
          o.frac[c.user_index()] += c.pt() / j.pt();
      out.push_back(o);
    }
    return out;
  }
}  // namespace

class CandidateJetMETProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit CandidateJetMETProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<std::vector<TICLCandidate>> candToken_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  const edm::EDGetTokenT<MapType> tkR2tToken_;
  // v6 only: the interpretation tracksters and the per-track assignment, to find the residual neutrals.
  edm::EDGetTokenT<std::vector<ticl::Trackster>> interpToken_;
  edm::EDGetTokenT<std::vector<int>> trackModeToken_, trackToTracksterToken_;
  // Deep match only: the assignable roots of the truth branch targets.
  edm::EDGetTokenT<std::vector<unsigned int>> assignableToken_;
  // PUPPI column only: the PF candidates and their PUPPI weights for the jets and for the missing transverse momentum.
  edm::EDGetTokenT<edm::View<reco::Candidate>> pfToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> puppiJetToken_, puppiMetToken_;
  const double minAbsEta_, maxAbsEta_, maxDz_, jetR_, minGenJetPt_, minJetAbsEta_, maxJetAbsEta_, maxMatchDR_;
  TTree* jets_;
  TTree* met_;
  TTree* tjets_;
  float tPt_, tEta_, tFrac_[3], tRecoPt_, tRecoFrac_[3];
  int event_, side_;
  float genPt_, genEta_, genPhi_, genFrac_[3], truthPt_, truthFrac_[3];
  float pt_[3], dR_[3], frac_[3][3], offset_[3];
  float truthMetX_, truthMetY_, truthSumPt_, truthSumPtOOT_, genMetX_, genMetY_, genSumPt_, metX_[3], metY_[3],
      sumPt_[3], maxPt_, maxPtRelErr_, badTrackSumPt_, truthSignalSumPt_, maxTrackP_, maxTracksterE_, maxTruthPt_;
  int nTruthInteractions_, nTruthOOT_, nTruthSignalLeaves_, maxPdg_, pvSource_;
  float pvZ_;
  float neutralPt_, doublePt_, untrackedChargedPt_, doubleResidualPt_, doubleTrackOnlyPt_, doubleWithTracksterPt_,
      doubleJetMemberPt_, doubleEMTypedPt_;
  float doublePtLo_, doublePtHi_, doubleDeepPtLo_, doubleDeepPtHi_, overKeptPt_, underKeptPt_;
  float neutralLoPt_, noTruthLoPt_, keptLoPt_;
  int nCountedLo_;
};

CandidateJetMETProbe::CandidateJetMETProbe(const edm::ParameterSet& cfg)
    : candToken_(consumes<std::vector<TICLCandidate>>(cfg.getParameter<edm::InputTag>("candidates"))),
      genToken_(consumes<std::vector<reco::GenParticle>>(cfg.getParameter<edm::InputTag>("genParticles"))),
      graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      lcToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      tkR2tToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      minAbsEta_(cfg.getParameter<double>("minAbsEta")),
      maxAbsEta_(cfg.getParameter<double>("maxAbsEta")),
      maxDz_(cfg.getParameter<double>("maxDz")),
      jetR_(cfg.getParameter<double>("jetR")),
      minGenJetPt_(cfg.getParameter<double>("minGenJetPt")),
      minJetAbsEta_(cfg.getParameter<double>("minJetAbsEta")),
      maxJetAbsEta_(cfg.getParameter<double>("maxJetAbsEta")),
      maxMatchDR_(cfg.getParameter<double>("maxMatchDR")) {
  usesResource(TFileService::kSharedResource);
  if (const auto interp = cfg.getParameter<std::string>("interpretations"); !interp.empty()) {
    interpToken_ = consumes<std::vector<ticl::Trackster>>(edm::InputTag(interp));
    trackModeToken_ = consumes<std::vector<int>>(edm::InputTag(interp, "trackMode"));
    trackToTracksterToken_ = consumes<std::vector<int>>(edm::InputTag(interp, "trackToTrackster"));
  }
  if (const auto roots = cfg.existsAs<std::string>("assignableRoots") ? cfg.getParameter<std::string>("assignableRoots")
                                                                        : std::string();
      !roots.empty())
    assignableToken_ = consumes<std::vector<unsigned int>>(edm::InputTag(roots));
  auto optional = [&cfg](const char* name) {
    return cfg.existsAs<std::string>(name) ? cfg.getParameter<std::string>(name) : std::string();
  };
  if (const auto pf = optional("pfCandidates"); !pf.empty()) {
    pfToken_ = consumes<edm::View<reco::Candidate>>(edm::InputTag(pf));
    puppiJetToken_ = consumes<edm::ValueMap<float>>(edm::InputTag(optional("puppiWeightsJets")));
    puppiMetToken_ = consumes<edm::ValueMap<float>>(edm::InputTag(optional("puppiWeightsMet")));
  }
  edm::Service<TFileService> fs;
  jets_ = fs->make<TTree>("jets", "jets");
  jets_->Branch("event", &event_);
  jets_->Branch("genPt", &genPt_);
  jets_->Branch("genEta", &genEta_);
  jets_->Branch("genPhi", &genPhi_);
  jets_->Branch("genFrac", genFrac_, "genFrac[3]/F");
  jets_->Branch("truthPt", &truthPt_);
  jets_->Branch("truthFrac", truthFrac_, "truthFrac[3]/F");
  jets_->Branch("pt", pt_, "pt[3]/F");
  jets_->Branch("dR", dR_, "dR[3]/F");
  jets_->Branch("frac", frac_, "frac[3][3]/F");
  jets_->Branch("offset", offset_, "offset[3]/F");
  met_ = fs->make<TTree>("met", "met");
  met_->Branch("event", &event_);
  met_->Branch("side", &side_);
  met_->Branch("genMetX", &genMetX_);
  met_->Branch("genMetY", &genMetY_);
  met_->Branch("genSumPt", &genSumPt_);
  met_->Branch("truthMetX", &truthMetX_);
  met_->Branch("truthMetY", &truthMetY_);
  met_->Branch("truthSumPt", &truthSumPt_);
  met_->Branch("truthSumPtOOT", &truthSumPtOOT_);
  met_->Branch("maxPt", &maxPt_);
  met_->Branch("maxPtRelErr", &maxPtRelErr_);
  met_->Branch("badTrackSumPt", &badTrackSumPt_);
  met_->Branch("truthSignalSumPt", &truthSignalSumPt_);
  met_->Branch("nTruthInteractions", &nTruthInteractions_);
  met_->Branch("nTruthOOT", &nTruthOOT_);
  met_->Branch("nTruthSignalLeaves", &nTruthSignalLeaves_);
  met_->Branch("maxPdg", &maxPdg_);
  met_->Branch("pvZ", &pvZ_);
  met_->Branch("pvSource", &pvSource_);
  met_->Branch("maxTrackP", &maxTrackP_);
  met_->Branch("maxTracksterE", &maxTracksterE_);
  met_->Branch("maxTruthPt", &maxTruthPt_);
  met_->Branch("neutralPt", &neutralPt_);
  met_->Branch("doublePt", &doublePt_);
  met_->Branch("untrackedChargedPt", &untrackedChargedPt_);
  met_->Branch("doubleResidualPt", &doubleResidualPt_);
  met_->Branch("doubleTrackOnlyPt", &doubleTrackOnlyPt_);
  met_->Branch("doubleWithTracksterPt", &doubleWithTracksterPt_);
  met_->Branch("doubleJetMemberPt", &doubleJetMemberPt_);
  met_->Branch("doubleEMTypedPt", &doubleEMTypedPt_);
  met_->Branch("doublePtLo", &doublePtLo_);
  met_->Branch("doublePtHi", &doublePtHi_);
  met_->Branch("doubleDeepPtLo", &doubleDeepPtLo_);
  met_->Branch("doubleDeepPtHi", &doubleDeepPtHi_);
  met_->Branch("overKeptPt", &overKeptPt_);
  met_->Branch("underKeptPt", &underKeptPt_);
  met_->Branch("neutralLoPt", &neutralLoPt_);
  met_->Branch("noTruthLoPt", &noTruthLoPt_);
  met_->Branch("keptLoPt", &keptLoPt_);
  met_->Branch("nCountedLo", &nCountedLo_);
  tjets_ = fs->make<TTree>("tjets", "tjets");
  tjets_->Branch("event", &event_);
  tjets_->Branch("pt", &tPt_);
  tjets_->Branch("eta", &tEta_);
  tjets_->Branch("frac", tFrac_, "frac[3]/F");
  tjets_->Branch("recoPt", &tRecoPt_);
  tjets_->Branch("recoFrac", tRecoFrac_, "recoFrac[3]/F");
  met_->Branch("metX", metX_, "metX[3]/F");
  met_->Branch("metY", metY_, "metY[3]/F");
  met_->Branch("sumPt", sumPt_, "sumPt[3]/F");
}

void CandidateJetMETProbe::analyze(const edm::Event& event, const edm::EventSetup&) {
  auto const& cands = event.get(candToken_);
  auto const& gen = event.get(genToken_);
  auto const& graph = event.get(graphToken_);
  event_ = event.id().event();
  if (gen.empty())
    return;
  // The signal vertex: the signal Interaction vertex of the truth graph (pvSource 0), else the production vertex of
  // the first signal particle (1), else the first final-state generator particle (2).
  pvSource_ = 2;
  const auto firstFinal = std::find_if(gen.begin(), gen.end(), [](auto const& p) { return p.status() == 1; });
  double pvZ = firstFinal != gen.end() ? firstFinal->vz() : gen.front().vz();
  for (auto const& v : graph.interactionVertices())
    if (v.isSignal()) {
      pvZ = v.position().z();
      pvSource_ = 0;
      break;
    }
  if (pvSource_ == 2)
    for (auto const& p : graph.signalParticles())
      if (auto const prod = p.productionVertices(); !prod.empty()) {
        pvZ = prod.front().position().z();
        pvSource_ = 1;
        break;
      }
  pvZ_ = pvZ;

  // Double counting from the truth graph. A truth particle is tracked when the track of a charged candidate matches it
  // (Fixed reco-to-truth association); it and its descendants then carry energy already counted by that track.
  auto const& layerClusters = event.get(lcToken_);
  auto const& index = event.get(hitIndexToken_);
  auto const& tkR2t = event.get(tkR2tToken_).getMap();
  const unsigned int nP = graph.particles().size();
  // Residual tracksters: held by a charged candidate, or the target of a track whose energy comes from the track (v6
  // track modes 2, 4, 5). A neutral candidate holding one is a residual and already subtracts the track momenta.
  std::vector<std::pair<edm::ProductID, size_t>> chargedTracksters;
  const std::vector<int> noModes;
  auto const& trackMode = interpToken_.isUninitialized() ? noModes : event.get(trackModeToken_);
  if (!interpToken_.isUninitialized()) {
    const auto interpH = event.getHandle(interpToken_);
    auto const& mode = trackMode;
    auto const& target = event.get(trackToTracksterToken_);
    for (size_t i = 0; i < mode.size() && i < target.size(); ++i)
      if ((mode[i] == 2 || mode[i] == 4 || mode[i] == 5) && target[i] >= 0)
        chargedTracksters.emplace_back(interpH.id(), static_cast<size_t>(target[i]));
  }
  // tracked: 1 when the charged candidate holds tracksters, 2 when it is built from the track alone.
  std::vector<char> tracked(nP, 0);
  for (auto const& c : cands) {
    if (c.charge() == 0 || c.trackPtrs().empty() || c.trackPtrs()[0].isNull())
      continue;
    for (auto const& t : c.tracksters())
      chargedTracksters.emplace_back(t.id(), t.key());
    const auto key = c.trackPtrs()[0].key();
    if (key >= tkR2t.size() || tkR2t[key].empty())
      continue;
    auto best = std::min_element(
        tkR2t[key].begin(), tkR2t[key].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
    const unsigned int u = best->index();
    if (u >= nP || tracked[u])
      continue;
    const char kind = !c.tracksters().empty() ? 1 : (key < trackMode.size() && trackMode[key] == 4 ? 2 : 3);
    tracked[u] = kind;
    for (auto const& d : truth::Particle(&graph, u).descendants())
      tracked[d.id()] = kind;
  }
  // trackedDeep: the deep match of the track of a charged candidate and its descendants.
  const bool deep = !assignableToken_.isUninitialized();
  std::vector<char> trackedDeep(nP, 0);
  if (deep) {
    const cornettostudy::DeepTrackMatch deepMatch(graph, tkR2t, event.get(assignableToken_));
    for (auto const& c : cands) {
      if (c.charge() == 0 || c.trackPtrs().empty() || c.trackPtrs()[0].isNull())
        continue;
      float purity;
      const int u = deepMatch(c.trackPtrs()[0].key(), purity);
      if (u < 0 || trackedDeep[u])
        continue;
      trackedDeep[u] = 1;
      for (auto const& d : truth::Particle(&graph, u).descendants())
        trackedDeep[d.id()] = 1;
    }
  }
  // Simulated HGCAL energy per cell and calo-boundary particle.
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
  // Truth energy of a candidate's tracksters: total, from tracked particles, from charged particles not tracked.
  auto candidateTruth = [&](const TICLCandidate& c,
                            double& total,
                            double& fromTracked,
                            double& fromTrackOnly,
                            double& fromJetMember,
                            double& fromUntrackedCharged) {
    total = fromTracked = fromTrackOnly = fromJetMember = fromUntrackedCharged = 0.;
    for (auto const& t : c.tracksters())
      for (auto lc : t->vertices()) {
        if (lc >= layerClusters.size())
          continue;
        for (auto const& [id, f] : layerClusters[lc].hitsAndFractions()) {
          auto it = cellTruth.find(id.rawId());
          if (it == cellTruth.end())
            continue;
          for (auto const& [p, e] : it->second) {
            total += f * e;
            if (tracked[p]) {
              fromTracked += f * e;
              if (tracked[p] == 2)
                fromJetMember += f * e;
              else if (tracked[p] == 3)
                fromTrackOnly += f * e;
            } else if (categoryOfPdg(graph.particles()[p].pdgId) == 0)
              fromUntrackedCharged += f * e;
          }
        }
      }
  };
  // Truth energy of a candidate's tracksters, each layer cluster divided by its vertex multiplicity: total and from
  // particles with trackedDeep.
  auto candidateTruthDeep = [&](const TICLCandidate& c, double& total, double& fromTracked) {
    total = fromTracked = 0.;
    for (auto const& t : c.tracksters())
      for (std::size_t i = 0; i < t->vertices().size(); ++i) {
        const auto lc = t->vertices()[i];
        if (lc >= layerClusters.size())
          continue;
        const double share = 1. / std::max<float>(1.f, t->vertex_multiplicity()[i]);
        for (auto const& [id, f] : layerClusters[lc].hitsAndFractions()) {
          auto it = cellTruth.find(id.rawId());
          if (it == cellTruth.end())
            continue;
          for (auto const& [p, e] : it->second) {
            total += share * f * e;
            if (trackedDeep[p])
              fromTracked += share * f * e;
          }
        }
      }
  };
  auto inRegion = [this](double eta) { return std::abs(eta) > minAbsEta_ && std::abs(eta) < maxAbsEta_; };
  const bool puppi = !pfToken_.isUninitialized();
  edm::Handle<edm::View<reco::Candidate>> pfH;
  edm::Handle<edm::ValueMap<float>> puppiJetH, puppiMetH;
  if (puppi) {
    pfH = event.getHandle(pfToken_);
    puppiJetH = event.getHandle(puppiJetToken_);
    puppiMetH = event.getHandle(puppiMetToken_);
  }

  for (side_ = 0; side_ < 2; ++side_) {
    const double sign = side_ == 0 ? -1. : 1.;
    std::vector<fastjet::PseudoJet> genIn, recoIn[3];
    genMetX_ = genMetY_ = genSumPt_ = 0.f;
    for (auto const& g : gen) {
      const int a = std::abs(g.pdgId());
      if (g.status() != 1 || a == 12 || a == 14 || a == 16 || !inRegion(g.eta()) || g.eta() * sign < 0.)
        continue;
      genIn.emplace_back(g.px(), g.py(), g.pz(), g.energy());
      genIn.back().set_user_index(category(g.pdgId(), g.charge()));
      genMetX_ -= g.px();
      genMetY_ -= g.py();
      genSumPt_ += g.pt();
    }
    std::vector<fastjet::PseudoJet> truthIn;
    truthMetX_ = truthMetY_ = truthSumPt_ = truthSumPtOOT_ = truthSignalSumPt_ = 0.f;
    nTruthOOT_ = nTruthSignalLeaves_ = 0;
    std::vector<uint64_t> interactions;
    for (auto const& pd : graph.particles())
      if (pd.isAtLevel(truth::LevelFlag::Signal) && side_ == 0)
        ++nTruthSignalLeaves_;
    for (auto const& pd : graph.particles()) {
      const int a = std::abs(pd.pdgId);
      if (!pd.isAtLevel(truth::LevelFlag::CaloBoundary) || a == 12 || a == 14 || a == 16)
        continue;
      auto const& p4 = pd.momentum;
      if (!(p4.pt() > 0.) || !inRegion(p4.eta()) || p4.eta() * sign < 0.)
        continue;
      interactions.push_back(pd.eventId);
      if (truth::bunchCrossingOf(pd.eventId) != 0) {
        ++nTruthOOT_;
        truthSumPtOOT_ += p4.pt();
        continue;
      }
      if (pd.eventId == 0)
        truthSignalSumPt_ += p4.pt();
      truthIn.emplace_back(p4.px(), p4.py(), p4.pz(), p4.E());
      truthIn.back().set_user_index(categoryOfPdg(pd.pdgId));
      truthMetX_ -= p4.px();
      truthMetY_ -= p4.py();
      truthSumPt_ += p4.pt();
    }
    std::sort(interactions.begin(), interactions.end());
    nTruthInteractions_ = std::unique(interactions.begin(), interactions.end()) - interactions.begin();
    for (int v = 0; v < 2; ++v)
      metX_[v] = metY_[v] = sumPt_[v] = 0.f;
    metX_[2] = metY_[2] = sumPt_[2] = puppi ? 0.f : -1.f;
    if (puppi)
      for (size_t i = 0; i < pfH->size(); ++i) {
        auto const& c = (*pfH)[i];
        if (!inRegion(c.eta()) || c.eta() * sign < 0.)
          continue;
        const auto ptr = pfH->ptrAt(i);
        const double wMet = (*puppiMetH)[ptr];
        metX_[2] -= wMet * c.px();
        metY_[2] -= wMet * c.py();
        sumPt_[2] += wMet * c.pt();
        if (const double wJet = (*puppiJetH)[ptr]; wJet > 0.) {
          recoIn[2].emplace_back(wJet * c.px(), wJet * c.py(), wJet * c.pz(), wJet * c.energy());
          recoIn[2].back().set_user_index(category(c.pdgId(), c.charge()));
        }
      }
    // Diagnostics, all candidates: the highest-pT candidate with the relative pT error of its track (-1: no track),
    // and the pT of charged candidates whose track has a relative pT error above 0.2.
    maxPt_ = badTrackSumPt_ = 0.f;
    neutralPt_ = doublePt_ = untrackedChargedPt_ = doubleResidualPt_ = doubleTrackOnlyPt_ = doubleWithTracksterPt_ =
        doubleJetMemberPt_ = doubleEMTypedPt_ = 0.f;
    doublePtLo_ = doublePtHi_ = 0.f;
    doubleDeepPtLo_ = doubleDeepPtHi_ = overKeptPt_ = underKeptPt_ = deep ? 0.f : -1.f;
    neutralLoPt_ = 0.f;
    noTruthLoPt_ = keptLoPt_ = deep ? 0.f : -1.f;
    nCountedLo_ = deep ? 0 : -1;
    maxPtRelErr_ = maxTrackP_ = -1.f;
    maxTracksterE_ = maxTruthPt_ = 0.f;
    maxPdg_ = 0;
    for (auto const& c : cands) {
      if (!inRegion(c.eta()) || c.eta() * sign < 0. || !(c.energy() > 0.))
        continue;
      const reco::Track* tk = !c.trackPtrs().empty() && c.trackPtrs()[0].isNonnull() ? c.trackPtrs()[0].get() : nullptr;
      const float relErr = tk && tk->pt() > 0. ? tk->ptError() / tk->pt() : -1.f;
      if (c.pt() > maxPt_) {
        maxPt_ = c.pt();
        maxPtRelErr_ = relErr;
        maxPdg_ = c.pdgId();
        maxTrackP_ = tk ? tk->p() : -1.f;
        maxTracksterE_ = 0.f;
        for (auto const& t : c.tracksters())
          maxTracksterE_ += t->raw_energy();
        maxTruthPt_ = 0.f;
        for (auto const& t : truthIn)
          if (reco::deltaR(c.eta(), c.phi(), t.eta(), t.phi_std()) < 0.1)
            maxTruthPt_ += t.pt();
      }
      if (relErr > 0.2f)
        badTrackSumPt_ += c.pt();
      if (c.charge() == 0) {
        double total, fromTracked, fromTrackOnly, fromJetMember, fromUntrackedCharged;
        candidateTruth(c, total, fromTracked, fromTrackOnly, fromJetMember, fromUntrackedCharged);
        neutralPt_ += c.pt();
        bool residual = false;
        double tracksterE = 0.;
        for (auto const& t : c.tracksters()) {
          tracksterE += t->regressed_energy();
          residual |= std::find(chargedTracksters.begin(), chargedTracksters.end(), std::make_pair(t.id(), t.key())) !=
                      chargedTracksters.end();
        }
        // The pT of the candidate before the neutral share.
        const bool lowPt = tracksterE / std::cosh(c.eta()) < 10.;
        if (lowPt)
          neutralLoPt_ += c.pt();
        if (total > 0.) {
          untrackedChargedPt_ += c.pt() * fromUntrackedCharged / total;
          double dPt;
          if (residual) {
            // A residual already subtracts the track momenta: only its excess over the untracked share is double.
            const double excess = c.energy() - tracksterE * (1. - fromTracked / total);
            dPt = c.energy() > 0. ? std::max(0., excess) * c.pt() / c.energy() : 0.;
            doubleResidualPt_ += dPt;
          } else {
            dPt = c.pt() * fromTracked / total;
            doubleTrackOnlyPt_ += c.pt() * fromTrackOnly / total;
            doubleJetMemberPt_ += c.pt() * fromJetMember / total;
            doubleWithTracksterPt_ += dPt - c.pt() * (fromTrackOnly + fromJetMember) / total;
            if (std::abs(c.pdgId()) == 22 || c.pdgId() == 111)
              doubleEMTypedPt_ += dPt;
          }
          doublePt_ += dPt;
          (lowPt ? doublePtLo_ : doublePtHi_) += dPt;
        }
        if (deep) {
          double totalDeep, fromTrackedDeep;
          candidateTruthDeep(c, totalDeep, fromTrackedDeep);
          if (totalDeep > 0.) {
            const double d = fromTrackedDeep / totalDeep;
            const double ptOverE = c.pt() / c.energy();
            const double kept = tracksterE * (1. - d);
            const double dPt = residual ? std::max(0., c.energy() - kept) * ptOverE : c.pt() * d;
            (lowPt ? doubleDeepPtLo_ : doubleDeepPtHi_) += dPt;
            if (lowPt) {
              overKeptPt_ += std::max(0., c.energy() - kept) * ptOverE;
              underKeptPt_ += std::max(0., kept - c.energy()) * ptOverE;
              keptLoPt_ += kept * ptOverE;
              ++nCountedLo_;
            }
          } else if (lowPt) {
            noTruthLoPt_ += c.pt();
          }
        }
      }
      const bool fromPU = c.charge() != 0 && !c.trackPtrs().empty() && c.trackPtrs()[0].isNonnull() &&
                          std::abs(c.trackPtrs()[0]->vz() - pvZ) > maxDz_;
      for (int v = 0; v < 2; ++v) {
        if (v == 1 && fromPU)
          continue;
        recoIn[v].emplace_back(c.px(), c.py(), c.pz(), c.energy());
        recoIn[v].back().set_user_index(category(c.pdgId(), c.charge()));
        metX_[v] -= c.px();
        metY_[v] -= c.py();
        sumPt_[v] += c.pt();
      }
    }
    met_->Fill();

    const auto genJets = cluster(genIn, jetR_, minGenJetPt_);
    const auto truthJets = cluster(truthIn, jetR_, minGenJetPt_);
    const std::vector<Jet> recoJets[3] = {
        cluster(recoIn[0], jetR_, 5.), cluster(recoIn[1], jetR_, 5.), cluster(recoIn[2], jetR_, 5.)};
    for (auto const& g : genJets) {
      if (std::abs(g.eta) < minJetAbsEta_ || std::abs(g.eta) > maxJetAbsEta_)
        continue;
      genPt_ = g.pt;
      genEta_ = g.eta;
      genPhi_ = g.phi;
      for (int k = 0; k < 3; ++k)
        genFrac_[k] = g.frac[k];
      truthPt_ = -1.f;
      for (int k = 0; k < 3; ++k)
        truthFrac_[k] = -1.f;
      float bestTruth = maxMatchDR_;
      for (auto const& t : truthJets)
        if (const float d = reco::deltaR(g.eta, g.phi, t.eta, t.phi); d < bestTruth) {
          bestTruth = d;
          truthPt_ = t.pt;
          for (int k = 0; k < 3; ++k)
            truthFrac_[k] = t.frac[k];
        }
      for (int v = 0; v < 3; ++v) {
        if (v == 2 && !puppi) {
          pt_[2] = dR_[2] = offset_[2] = -1.f;
          for (int k = 0; k < 3; ++k)
            frac_[2][k] = -1.f;
          continue;
        }
        std::vector<float> cones;
        for (int k = 1; k < 8; ++k) {
          const double phi = g.phi + k * M_PI / 4.;  // deltaR reduces the phi difference
          bool nearJet = false;
          for (auto const& o : genJets)
            nearJet |= reco::deltaR(g.eta, phi, o.eta, o.phi) < 2. * jetR_;
          if (nearJet)
            continue;
          float sum = 0.f;
          for (auto const& c : recoIn[v])
            if (reco::deltaR(g.eta, phi, c.eta(), c.phi_std()) < jetR_)
              sum += c.pt();
          cones.push_back(sum);
        }
        offset_[v] = -1.f;
        if (!cones.empty()) {
          std::nth_element(cones.begin(), cones.begin() + cones.size() / 2, cones.end());
          offset_[v] = cones[cones.size() / 2];
        }
        pt_[v] = -1.f;
        dR_[v] = maxMatchDR_;
        for (int k = 0; k < 3; ++k)
          frac_[v][k] = -1.f;
        for (auto const& r : recoJets[v]) {
          const float d = reco::deltaR(g.eta, g.phi, r.eta, r.phi);
          if (d < dR_[v]) {
            dR_[v] = d;
            pt_[v] = r.pt;
            for (int k = 0; k < 3; ++k)
              frac_[v][k] = r.frac[k];
          }
        }
      }
      jets_->Fill();
    }
    for (auto const& t : truthJets) {
      if (std::abs(t.eta) < minJetAbsEta_ || std::abs(t.eta) > maxJetAbsEta_)
        continue;
      tPt_ = t.pt;
      tEta_ = t.eta;
      tRecoPt_ = -1.f;
      float best = maxMatchDR_;
      for (int k = 0; k < 3; ++k) {
        tFrac_[k] = t.frac[k];
        tRecoFrac_[k] = -1.f;
      }
      for (auto const& r : recoJets[0]) {
        const float d = reco::deltaR(t.eta, t.phi, r.eta, r.phi);
        if (d < best) {
          best = d;
          tRecoPt_ = r.pt;
          for (int k = 0; k < 3; ++k)
            tRecoFrac_[k] = r.frac[k];
        }
      }
      tjets_->Fill();
    }
  }
}

DEFINE_FWK_MODULE(CandidateJetMETProbe);
