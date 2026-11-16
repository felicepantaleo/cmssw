// Study probe, not for integration. Training rows of two calibrations of TICLv6:
// - tree "groups": one row per final trackster that charged hadrons share in the arbitration (tracks in mode charged
//   hadron or recovery on a hadronic trackster, and jet members): calorimeter features of the trackster, the energies
//   and variances of the tracks (pion mass, trackMomentumErrorScale x sigma(p)), and the truth: the energy of the best
//   truth particle of each track (track associator) and the adaptive-associator truth particle of the trackster.
//   The truth group of a row is the set of distinct best truth particles of its tracks, without a particle that has an
//   ancestor in the set. grpTrueE is the sum of their energies. On the rechit scale (a cell gives a particle its
//   simulated share of the cell times the rechit energy, HGCAL only): grpInE is the energy of the truth group and its
//   descendants in the trackster, grpAllE is their energy in all cells, tsTruthEReco is the energy of the trackster
//   cells that hold simulated energy. Purity = grpInE / tsTruthEReco, completeness = grpInE / grpAllE.
// - tree "superclusters": one row per TICL e/gamma supercluster: the regression features (EGammaSuperClusterFeatures.h),
//   and the truth: the adaptive-associator truth particle of its final trackster, and the nearest generator electron or
//   photon (status 1) within dR 0.1.

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"
#include "RecoTICL/ParticleFlow/interface/EGammaSuperClusterFeatures.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

#include "TString.h"
#include "TTree.h"

namespace {
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;
  constexpr float kPionMass = 0.13957f;

  bool isHGCal(uint32_t rawId) {
    const auto det = DetId(rawId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }
}  // namespace

class TICLCalibrationProbe : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit TICLCalibrationProbe(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // Best truth particle of a reco object in a reco-to-truth map: index (-1: none), score and shared energy.
  struct Match {
    int particle = -1;
    float score = -1.f;
    float shared = 0.f;
  };
  static Match bestMatch(const MapType& map, unsigned int key, std::size_t nParticles);

  const edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken_;
  const edm::EDGetTokenT<std::vector<int>> trackToTracksterToken_;
  const edm::EDGetTokenT<std::vector<int>> trackModeToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> trackToken_;
  const edm::EDGetTokenT<MapType> tkR2tToken_;
  const edm::EDGetTokenT<MapType> tsR2tToken_;
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> lcToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> recHitTokens_;
  const edm::EDGetTokenT<reco::SuperClusterCollection> scToken_;
  const edm::EDGetTokenT<std::vector<int>> scIndexToken_;
  const edm::EDGetTokenT<std::vector<ticl::Trackster>> egToken_;
  const edm::EDGetTokenT<std::vector<int>> egIndexToken_;
  const edm::EDGetTokenT<std::vector<reco::GenParticle>> genToken_;
  const float trackMomentumErrorScale_;

  TTree *groups_, *superClusters_;
  int event_;
  // groups
  int gTs_, gHadronic_, gNLC_, gNTracks_, gNJetMembers_, gAdPdg_, gAdSignal_;
  float gRegE_, gRawE_, gRawEmE_, gAbsEta_, gSigPCA_[3], gAdE_, gAdScore_, gAdShared_;
  float gGrpTrueE_, gGrpInE_, gGrpAllE_, gTsTruthEReco_;
  std::vector<float> gTrkE_, gTrkVar_, gTrkTrueE_, gTrkPurity_;
  std::vector<int> gTrkTruePdg_, gTrkMode_, gTrkTruthIdx_;
  // superclusters
  int sNClusters_, sAdPdg_, sAdSignal_, sGenPdg_;
  float sFeatures_[ticl::egammaSuperCluster::kNFeatures], sRawE_, sE_, sEta_, sPhi_, sAdE_, sAdScore_, sAdShared_,
      sGenE_, sGenDR_;
};

TICLCalibrationProbe::TICLCalibrationProbe(const edm::ParameterSet& cfg)
    : tracksterToken_(consumes(cfg.getParameter<edm::InputTag>("tracksters"))),
      trackToTracksterToken_(consumes(cfg.getParameter<edm::InputTag>("trackToTrackster"))),
      trackModeToken_(consumes(cfg.getParameter<edm::InputTag>("trackMode"))),
      trackToken_(consumes(cfg.getParameter<edm::InputTag>("tracks"))),
      tkR2tToken_(consumes(cfg.getParameter<edm::InputTag>("trackRecoToTruth"))),
      tsR2tToken_(consumes(cfg.getParameter<edm::InputTag>("tracksterRecoToTruth"))),
      graphToken_(consumes(cfg.getParameter<edm::InputTag>("graph"))),
      hitIndexToken_(consumes(cfg.getParameter<edm::InputTag>("hitIndex"))),
      lcToken_(consumes(cfg.getParameter<edm::InputTag>("layerClusters"))),
      scToken_(consumes(cfg.getParameter<edm::InputTag>("superClusters"))),
      scIndexToken_(
          consumes(edm::InputTag(cfg.getParameter<edm::InputTag>("superClusters").label(), "tracksterIndex"))),
      egToken_(consumes(cfg.getParameter<edm::InputTag>("eGammaTracksters"))),
      egIndexToken_(
          consumes(edm::InputTag(cfg.getParameter<edm::InputTag>("eGammaTracksters").label(), "interpretationIndex"))),
      genToken_(consumes(cfg.getParameter<edm::InputTag>("genParticles"))),
      trackMomentumErrorScale_(cfg.getParameter<double>("trackMomentumErrorScale")) {
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("hgcalRecHits"))
    recHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  usesResource("TFileService");
  edm::Service<TFileService> fs;
  groups_ = fs->make<TTree>("groups", "charged-hadron groups on a final trackster");
  groups_->Branch("event", &event_);
  groups_->Branch("ts", &gTs_);
  groups_->Branch("hadronic", &gHadronic_);
  groups_->Branch("nLC", &gNLC_);
  groups_->Branch("nTracks", &gNTracks_);
  groups_->Branch("nJetMembers", &gNJetMembers_);
  groups_->Branch("regE", &gRegE_);
  groups_->Branch("rawE", &gRawE_);
  groups_->Branch("rawEmE", &gRawEmE_);
  groups_->Branch("absEta", &gAbsEta_);
  groups_->Branch("sigPCA", gSigPCA_, "sigPCA[3]/F");
  groups_->Branch("adPdg", &gAdPdg_);
  groups_->Branch("adSignal", &gAdSignal_);
  groups_->Branch("adE", &gAdE_);
  groups_->Branch("adScore", &gAdScore_);
  groups_->Branch("adShared", &gAdShared_);
  groups_->Branch("trkE", &gTrkE_);
  groups_->Branch("trkVar", &gTrkVar_);
  groups_->Branch("trkTrueE", &gTrkTrueE_);
  groups_->Branch("trkPurity", &gTrkPurity_);
  groups_->Branch("trkTruePdg", &gTrkTruePdg_);
  groups_->Branch("trkMode", &gTrkMode_);
  groups_->Branch("trkTruthIdx", &gTrkTruthIdx_);
  groups_->Branch("grpTrueE", &gGrpTrueE_);
  groups_->Branch("grpInE", &gGrpInE_);
  groups_->Branch("grpAllE", &gGrpAllE_);
  groups_->Branch("tsTruthEReco", &gTsTruthEReco_);
  superClusters_ = fs->make<TTree>("superclusters", "TICL e/gamma superclusters");
  superClusters_->Branch("event", &event_);
  superClusters_->Branch("features", sFeatures_, Form("features[%zu]/F", ticl::egammaSuperCluster::kNFeatures));
  superClusters_->Branch("rawE", &sRawE_);
  superClusters_->Branch("energy", &sE_);
  superClusters_->Branch("eta", &sEta_);
  superClusters_->Branch("phi", &sPhi_);
  superClusters_->Branch("nClusters", &sNClusters_);
  superClusters_->Branch("adPdg", &sAdPdg_);
  superClusters_->Branch("adSignal", &sAdSignal_);
  superClusters_->Branch("adE", &sAdE_);
  superClusters_->Branch("adScore", &sAdScore_);
  superClusters_->Branch("adShared", &sAdShared_);
  superClusters_->Branch("genPdg", &sGenPdg_);
  superClusters_->Branch("genE", &sGenE_);
  superClusters_->Branch("genDR", &sGenDR_);
}

TICLCalibrationProbe::Match TICLCalibrationProbe::bestMatch(const MapType& map,
                                                            unsigned int key,
                                                            std::size_t nParticles) {
  Match m;
  if (key >= map.size() || map[key].empty())
    return m;
  auto best = std::min_element(
      map[key].begin(), map[key].end(), [](auto const& a, auto const& b) { return a.score() < b.score(); });
  if (best->index() >= nParticles)
    return m;
  m.particle = static_cast<int>(best->index());
  m.score = best->score();
  m.shared = best->sharedEnergy();
  return m;
}

void TICLCalibrationProbe::analyze(const edm::Event& evt, const edm::EventSetup&) {
  event_ = static_cast<int>(evt.id().event());
  const auto& tracksters = evt.get(tracksterToken_);
  const auto& trackToTrackster = evt.get(trackToTracksterToken_);
  const auto& trackMode = evt.get(trackModeToken_);
  const auto& tracks = evt.get(trackToken_);
  const auto& tkR2t = evt.get(tkR2tToken_);
  const auto& tsR2t = evt.get(tsR2tToken_);
  const auto& graph = evt.get(graphToken_);
  const std::size_t nP = graph.particles().size();
  const auto& hitIndex = evt.get(hitIndexToken_);
  const auto& layerClusters = evt.get(lcToken_);

  // Rechit energy and simulated energy of all particles, per HGCAL cell.
  std::unordered_map<uint32_t, float> recHitE, cellSim;
  for (auto const& token : recHitTokens_)
    for (auto const& h : evt.get(token))
      recHitE[h.id().rawId()] += h.energy();
  for (uint32_t p = 0; p < nP; ++p)
    for (auto const& h : hitIndex.directHits(truth::HitChannel::Calo, p))
      if (isHGCal(h.detId) && h.energy > 0.f)
        cellSim[h.detId] += h.energy;
  // Rechit energy of a cell times the share of the simulated energy e in it; 0 without a rechit or simulated energy.
  auto recoScale = [&](uint32_t id, float e) {
    const auto r = recHitE.find(id);
    const auto s = cellSim.find(id);
    return r == recHitE.end() || s == cellSim.end() ? 0.f : r->second * std::min(1.f, e / s->second);
  };
  std::vector<uint32_t> truthGroup, kept;
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  std::unordered_map<uint32_t, float> groupSim;

  // Charged-hadron groups, as in TICLCandidateArbitrationProducer.
  std::vector<std::vector<unsigned int>> groupTracks(tracksters.size());
  for (std::size_t iTrack = 0; iTrack < trackMode.size() && iTrack < trackToTrackster.size(); ++iTrack) {
    const int ts = trackToTrackster[iTrack];
    if (ts < 0 || ts >= static_cast<int>(tracksters.size()))
      continue;
    const auto mode = static_cast<ticl::TrackMode>(trackMode[iTrack]);
    const bool chargedHadron = mode == ticl::TrackMode::kChargedHadron || mode == ticl::TrackMode::kRecovery;
    if (mode == ticl::TrackMode::kJetMember || (chargedHadron && tracksters[ts].isHadronic()))
      groupTracks[ts].push_back(static_cast<unsigned int>(iTrack));
  }
  for (std::size_t ts = 0; ts < tracksters.size(); ++ts) {
    if (groupTracks[ts].empty())
      continue;
    const auto& t = tracksters[ts];
    gTs_ = static_cast<int>(ts);
    gHadronic_ = t.isHadronic() ? 1 : 0;
    gNLC_ = static_cast<int>(t.vertices().size());
    gRegE_ = t.regressed_energy();
    gRawE_ = t.raw_energy();
    gRawEmE_ = t.raw_em_energy();
    gAbsEta_ = std::abs(t.barycenter().eta());
    for (int i = 0; i < 3; ++i)
      gSigPCA_[i] = t.sigmasPCA()[i];
    const auto ad = bestMatch(tsR2t, ts, nP);
    gAdPdg_ = ad.particle >= 0 ? graph.particles()[ad.particle].pdgId : 0;
    gAdSignal_ = ad.particle >= 0 && graph.particles()[ad.particle].isSignal() ? 1 : 0;
    gAdE_ = ad.particle >= 0 ? graph.particles()[ad.particle].momentum.energy() : -1.f;
    gAdScore_ = ad.score;
    gAdShared_ = ad.shared;
    gTrkE_.clear();
    gTrkVar_.clear();
    gTrkTrueE_.clear();
    gTrkPurity_.clear();
    gTrkTruePdg_.clear();
    gTrkMode_.clear();
    gTrkTruthIdx_.clear();
    truthGroup.clear();
    gNJetMembers_ = 0;
    for (auto iTrack : groupTracks[ts]) {
      const auto& tk = tracks[iTrack];
      const float p = tk.p();
      const float sigmaP = trackMomentumErrorScale_ * tk.qoverpError() * p * p;
      gTrkE_.push_back(std::sqrt(p * p + kPionMass * kPionMass));
      gTrkVar_.push_back(sigmaP * sigmaP);
      gTrkMode_.push_back(trackMode[iTrack]);
      gNJetMembers_ += trackMode[iTrack] == static_cast<int>(ticl::TrackMode::kJetMember) ? 1 : 0;
      const auto m = bestMatch(tkR2t, iTrack, nP);
      gTrkTrueE_.push_back(m.particle >= 0 ? graph.particles()[m.particle].momentum.energy() : -1.f);
      gTrkPurity_.push_back(m.particle >= 0 ? 1.f - m.score : -1.f);
      gTrkTruePdg_.push_back(m.particle >= 0 ? graph.particles()[m.particle].pdgId : 0);
      gTrkTruthIdx_.push_back(m.particle);
      const auto particle = static_cast<uint32_t>(m.particle);
      if (m.particle >= 0 && std::find(truthGroup.begin(), truthGroup.end(), particle) == truthGroup.end())
        truthGroup.push_back(particle);
    }
    gNTracks_ = static_cast<int>(groupTracks[ts].size());
    if (truthGroup.size() > 1) {
      kept.clear();
      for (const auto p : truthGroup) {
        const auto ancestors = graph.particle(p).ancestors();
        if (std::none_of(ancestors.begin(), ancestors.end(), [&](auto const& a) {
              return std::find(truthGroup.begin(), truthGroup.end(), a.id()) != truthGroup.end();
            }))
          kept.push_back(p);
      }
      truthGroup.swap(kept);
    }
    double trueE = 0., inE = 0., allE = 0., truthEReco = 0.;
    hits.clear();
    for (const auto p : truthGroup) {
      trueE += graph.particles()[p].momentum.energy();
      hitIndex.appendSubgraphHits(truth::HitChannel::Calo, p, hits);
    }
    groupSim.clear();
    for (auto const& h : hits)
      if (isHGCal(h.detId) && h.energy > 0.f)
        groupSim[h.detId] += h.energy;
    for (auto const& [id, e] : groupSim)
      allE += recoScale(id, e);
    for (std::size_t i = 0; i < t.vertices().size(); ++i) {
      const auto lc = t.vertices()[i];
      if (lc >= layerClusters.size())
        continue;
      const float share =
          i < t.vertex_multiplicity().size() ? 1.f / std::max<float>(1.f, t.vertex_multiplicity()[i]) : 1.f;
      for (auto const& [id, fraction] : layerClusters[lc].hitsAndFractions()) {
        const auto s = cellSim.find(id.rawId());
        if (s == cellSim.end())
          continue;
        truthEReco += share * fraction * recoScale(id.rawId(), s->second);
        if (const auto g = groupSim.find(id.rawId()); g != groupSim.end())
          inE += share * fraction * recoScale(id.rawId(), g->second);
      }
    }
    gGrpTrueE_ = static_cast<float>(trueE);
    gGrpInE_ = static_cast<float>(inE);
    gGrpAllE_ = static_cast<float>(allE);
    gTsTruthEReco_ = static_cast<float>(truthEReco);
    groups_->Fill();
  }

  // TICL e/gamma superclusters.
  const auto& superClusters = evt.get(scToken_);
  const auto& scIndex = evt.get(scIndexToken_);
  const auto& eGammaTracksters = evt.get(egToken_);
  const auto& egIndex = evt.get(egIndexToken_);
  const auto genH = evt.getHandle(genToken_);
  for (std::size_t i = 0; i < superClusters.size() && i < scIndex.size(); ++i) {
    const auto& sc = superClusters[i];
    const int k = scIndex[i];
    if (k < 0 || k >= static_cast<int>(eGammaTracksters.size()) || k >= static_cast<int>(egIndex.size()))
      continue;
    // Features 0-10 (regression v2); the cell-level features 11-18 stay 0.
    const auto features = ticl::egammaSuperCluster::regressionFeatures(
        sc, eGammaTracksters[k], ticl::egammaSuperCluster::CellFeatures{}, 0);
    std::copy(features.begin(), features.end(), sFeatures_);
    sRawE_ = sc.rawEnergy();
    sE_ = sc.energy();
    sEta_ = sc.eta();
    sPhi_ = sc.phi();
    sNClusters_ = static_cast<int>(sc.clustersSize());
    const auto ad = bestMatch(tsR2t, static_cast<unsigned int>(egIndex[k]), nP);
    sAdPdg_ = ad.particle >= 0 ? graph.particles()[ad.particle].pdgId : 0;
    sAdSignal_ = ad.particle >= 0 && graph.particles()[ad.particle].isSignal() ? 1 : 0;
    sAdE_ = ad.particle >= 0 ? graph.particles()[ad.particle].momentum.energy() : -1.f;
    sAdScore_ = ad.score;
    sAdShared_ = ad.shared;
    sGenPdg_ = 0;
    sGenE_ = -1.f;
    sGenDR_ = -1.f;
    if (genH.isValid())
      for (auto const& g : *genH) {
        if (g.status() != 1 || (std::abs(g.pdgId()) != 11 && g.pdgId() != 22))
          continue;
        const float dr = reco::deltaR(sc.eta(), sc.phi(), g.eta(), g.phi());
        if (dr < 0.1f && (sGenDR_ < 0.f || dr < sGenDR_)) {
          sGenDR_ = dr;
          sGenPdg_ = g.pdgId();
          sGenE_ = g.energy();
        }
      }
    superClusters_->Fill();
  }
}

DEFINE_FWK_MODULE(TICLCalibrationProbe);
