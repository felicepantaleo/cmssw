// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

// DQM plots that compare truth::Branch to the legacy HGCAL truth objects (CaloParticle, SimCluster).
// For each object, the Branch seeded by the object's first SimTrack is compared to the object's
// hits_and_fractions. The BranchHitAssociator checks that this Branch is also the best hit match.
// The harvester (DQMGenericClient) computes the reproduction efficiency from the numerator and
// denominator histograms.

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

#include "PhysicsTools/TruthInfo/interface/Branch.h"
#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/SubgraphHitView.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/InteractionId.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

// One set of monitor elements per legacy collection (CaloParticle, SimCluster).
struct BranchHGCalPlots {
  // Numerator/denominator for the harvester-computed reproduction efficiency.
  dqm::reco::MonitorElement* denomEta = nullptr;
  dqm::reco::MonitorElement* denomPt = nullptr;
  dqm::reco::MonitorElement* denomEnergy = nullptr;
  dqm::reco::MonitorElement* effNumEta = nullptr;
  dqm::reco::MonitorElement* effNumPt = nullptr;
  dqm::reco::MonitorElement* effNumEnergy = nullptr;
  // Quality distributions.
  dqm::reco::MonitorElement* purity = nullptr;
  dqm::reco::MonitorElement* completenessHits = nullptr;
  dqm::reco::MonitorElement* completenessEnergy = nullptr;
  dqm::reco::MonitorElement* energyResponse = nullptr;
  // Raw energy response: Branch energy over the object's hit energy, on the deposited (sim) and
  // reconstructed (rec) scales. It is near 1 when the Branch reproduces the object's calorimeter energy.
  dqm::reco::MonitorElement* rawEnergyResponseSim = nullptr;
  dqm::reco::MonitorElement* rawEnergyResponseReco = nullptr;
  // Profiles vs kinematics.
  dqm::reco::MonitorElement* purityVsEta = nullptr;
  dqm::reco::MonitorElement* completenessVsEta = nullptr;
  dqm::reco::MonitorElement* responseVsEta = nullptr;
  dqm::reco::MonitorElement* responseVsEnergy = nullptr;
  dqm::reco::MonitorElement* rawResponseSimVsEnergy = nullptr;
  dqm::reco::MonitorElement* rawResponseRecoVsEnergy = nullptr;

  // Performance of the best hit-matched Branch of each object. This Branch can differ from the
  // natural (trackId-seeded) Branch.
  dqm::reco::MonitorElement* bestPurity = nullptr;
  dqm::reco::MonitorElement* bestCompletenessHits = nullptr;
  dqm::reco::MonitorElement* bestCompletenessEnergy = nullptr;
  dqm::reco::MonitorElement* bestResponse = nullptr;
  // Self-match numerator: best hit-matched Branch == natural Branch (denom reused).
  dqm::reco::MonitorElement* selfMatchEta = nullptr;
  dqm::reco::MonitorElement* selfMatchPt = nullptr;
  // Merge/split: distinct Branches sharing >=10% of the object's hits.
  dqm::reco::MonitorElement* nSharingBranches = nullptr;
};

struct BranchHGCalHistograms {
  BranchHGCalPlots caloParticle;
  BranchHGCalPlots simCluster;
};

class BranchHGCalValidator : public DQMGlobalEDAnalyzer<BranchHGCalHistograms> {
public:
  explicit BranchHGCalValidator(edm::ParameterSet const&);
  void bookHistograms(dqm::reco::DQMStore::IBooker&,
                      edm::Run const&,
                      edm::EventSetup const&,
                      BranchHGCalHistograms&) const override;
  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, BranchHGCalHistograms const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void book(dqm::reco::DQMStore::IBooker&, BranchHGCalPlots&, std::string const& sub) const;

  template <class Collection>
  void validate(Collection const& objects,
                truth::Graph const& graph,
                TruthGraph const& raw,
                truth::SubgraphHitView& hitIndex,
                truth::BranchHitAssociator const& assoc,
                std::unordered_map<uint64_t, uint32_t> const& tidToParticle,
                std::unordered_map<uint32_t, float> const& cellSimEnergy,
                std::unordered_map<uint32_t, float> const& recHitEnergyByDetId,
                BranchHGCalPlots const& plots) const;

  // Whole-cell RecHit energy keyed by DetId, from the HGCal then the PF RecHit collections
  // (the DetIdToRecHitMapProducer order). The first entry wins for a duplicate DetId.
  std::unordered_map<uint32_t, float> collectRecHitEnergyByDetId(edm::Event const&) const;

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<TruthGraph> rawToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;
  const edm::EDGetTokenT<std::vector<SimCluster>> simClusterToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcalRecHitTokens_;
  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> pfRecHitTokens_;
  std::vector<edm::InputTag> hgcalRecHitTags_;
  std::vector<edm::InputTag> pfRecHitTags_;

  const std::string folder_;
  const double minPt_;
  const double maxEta_;
};

BranchHGCalValidator::BranchHGCalValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      rawToken_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      caloParticleToken_(consumes<std::vector<CaloParticle>>(cfg.getParameter<edm::InputTag>("caloParticles"))),
      simClusterToken_(consumes<std::vector<SimCluster>>(cfg.getParameter<edm::InputTag>("simClusters"))),
      folder_(cfg.getParameter<std::string>("folder")),
      minPt_(cfg.getParameter<double>("minPt")),
      maxEta_(cfg.getParameter<double>("maxEta")) {
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("hgcalRecHits")) {
    hgcalRecHitTags_.push_back(tag);
    hgcalRecHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  }
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("pfRecHits")) {
    pfRecHitTags_.push_back(tag);
    pfRecHitTokens_.push_back(consumes<reco::PFRecHitCollection>(tag));
  }
}

void BranchHGCalValidator::book(dqm::reco::DQMStore::IBooker& ib, BranchHGCalPlots& p, std::string const& sub) const {
  ib.setCurrentFolder(folder_ + "/" + sub);

  constexpr int kEtaBins = 40;
  constexpr double kEtaMax = 3.2;
  constexpr int kPtBins = 50;
  constexpr double kPtMax = 200.;
  constexpr int kEBins = 50;
  constexpr double kEMax = 500.;

  p.denomEta = ib.book1D("denom_eta", "Selected truth objects vs #eta;#eta;objects", kEtaBins, -kEtaMax, kEtaMax);
  p.denomPt = ib.book1D("denom_pt", "Selected truth objects vs p_{T};p_{T} [GeV];objects", kPtBins, 0., kPtMax);
  p.denomEnergy = ib.book1D("denom_energy", "Selected truth objects vs E;E [GeV];objects", kEBins, 0., kEMax);
  p.effNumEta =
      ib.book1D("effnum_eta", "Branch-reproduced truth objects vs #eta;#eta;objects", kEtaBins, -kEtaMax, kEtaMax);
  p.effNumPt =
      ib.book1D("effnum_pt", "Branch-reproduced truth objects vs p_{T};p_{T} [GeV];objects", kPtBins, 0., kPtMax);
  p.effNumEnergy =
      ib.book1D("effnum_energy", "Branch-reproduced truth objects vs E;E [GeV];objects", kEBins, 0., kEMax);

  p.purity = ib.book1D("purity", "Branch hit purity;purity;objects", 52, -0.01, 1.03);
  p.completenessHits = ib.book1D("completeness_hits", "Branch hit completeness;completeness;objects", 52, -0.01, 1.03);
  p.completenessEnergy =
      ib.book1D("completeness_energy", "Branch energy completeness;completeness;objects", 52, -0.01, 1.03);
  p.energyResponse =
      ib.book1D("energy_response", "Branch sim-energy containment;E^{sim}_{Branch}/E_{gen};objects", 60, 0., 1.5);
  // The deposited-scale response is a closure test: it is 1 when the Branch holds the object's SimTracks.
  // A deviation flags a fraction or deposit bug. The reconstructed-scale response is the informative one.
  p.rawEnergyResponseSim = ib.book1D("raw_energy_response_sim",
                                     "Branch raw energy response (deposited);E^{sim}_{Branch}/E^{sim}_{hits};objects",
                                     80,
                                     0.,
                                     2.);
  p.rawEnergyResponseReco =
      ib.book1D("raw_energy_response_reco",
                "Branch raw energy response (reconstructed);E^{rec}_{Branch}/E^{rec}_{hits};objects",
                80,
                0.,
                4.);

  p.purityVsEta =
      ib.bookProfile("purity_vs_eta", "Branch hit purity vs #eta;#eta;purity", kEtaBins, -kEtaMax, kEtaMax, 0., 1.05);
  p.completenessVsEta = ib.bookProfile("completeness_vs_eta",
                                       "Branch energy completeness vs #eta;#eta;completeness",
                                       kEtaBins,
                                       -kEtaMax,
                                       kEtaMax,
                                       0.,
                                       1.05);
  p.responseVsEta = ib.bookProfile("response_vs_eta",
                                   "Branch sim-energy containment vs #eta;#eta;E^{sim}_{Branch}/E_{gen}",
                                   kEtaBins,
                                   -kEtaMax,
                                   kEtaMax,
                                   0.,
                                   1.5);
  p.responseVsEnergy = ib.bookProfile("response_vs_energy",
                                      "Branch sim-energy containment vs E;E [GeV];E^{sim}_{Branch}/E_{gen}",
                                      kEBins,
                                      0.,
                                      kEMax,
                                      0.,
                                      1.5);
  p.rawResponseSimVsEnergy =
      ib.bookProfile("raw_response_sim_vs_energy",
                     "Branch raw energy response (deposited) vs E;E [GeV];E^{sim}_{Branch}/E^{sim}_{hits}",
                     kEBins,
                     0.,
                     kEMax,
                     0.,
                     2.);
  p.rawResponseRecoVsEnergy =
      ib.bookProfile("raw_response_reco_vs_energy",
                     "Branch raw energy response (reconstructed) vs E;E [GeV];E^{rec}_{Branch}/E^{rec}_{hits}",
                     kEBins,
                     0.,
                     kEMax,
                     0.,
                     4.);

  // Best hit-matched Branch per object.
  p.bestPurity = ib.book1D("bestmatch_purity", "Best-match Branch hit purity;purity;objects", 52, -0.01, 1.03);
  p.bestCompletenessHits = ib.book1D(
      "bestmatch_completeness_hits", "Best-match Branch hit completeness;completeness;objects", 52, -0.01, 1.03);
  p.bestCompletenessEnergy = ib.book1D(
      "bestmatch_completeness_energy", "Best-match Branch energy completeness;completeness;objects", 52, -0.01, 1.03);
  p.bestResponse = ib.book1D(
      "bestmatch_response", "Best-match Branch sim-energy containment;E^{sim}_{Branch}/E_{gen};objects", 60, 0., 1.5);
  p.selfMatchEta = ib.book1D(
      "selfmatch_eta", "Objects whose best Branch is the natural one vs #eta;#eta;objects", kEtaBins, -kEtaMax, kEtaMax);
  p.selfMatchPt = ib.book1D(
      "selfmatch_pt", "Objects whose best Branch is the natural one vs p_{T};p_{T} [GeV];objects", kPtBins, 0., kPtMax);
  p.nSharingBranches = ib.book1D(
      "n_sharing_branches", "Distinct Branches sharing >=10% of the object hits;#Branches;objects", 51, -0.5, 50.5);
}

void BranchHGCalValidator::bookHistograms(dqm::reco::DQMStore::IBooker& ib,
                                          edm::Run const&,
                                          edm::EventSetup const&,
                                          BranchHGCalHistograms& histograms) const {
  book(ib, histograms.caloParticle, "CaloParticle");
  book(ib, histograms.simCluster, "SimCluster");
}

namespace {

  // (EncodedEventId, SimTrack trackId) -> logical particle.
  std::unordered_map<uint64_t, uint32_t> buildTrackIdToParticle(truth::Graph const& graph, TruthGraph const& raw) {
    std::unordered_map<uint64_t, uint32_t> out;
    out.reserve(graph.nParticles());
    for (uint32_t i = 0; i < graph.nParticles(); ++i) {
      const int32_t simNode = graph.particles()[i].simNode;
      if (simNode < 0 || static_cast<uint32_t>(simNode) >= raw.nNodes())
        continue;
      auto const& nr = raw.nodeRef(static_cast<uint32_t>(simNode));
      if (nr.kind == TruthGraph::NodeKind::SimTrack)
        out[truth::simObjectKey(raw.nodeEventId(static_cast<uint32_t>(simNode)), static_cast<uint32_t>(nr.key))] = i;
    }
    return out;
  }
}  // namespace

template <class Collection>
void BranchHGCalValidator::validate(Collection const& objects,
                                    truth::Graph const& graph,
                                    TruthGraph const& raw,
                                    truth::SubgraphHitView& hitIndex,
                                    truth::BranchHitAssociator const& assoc,
                                    std::unordered_map<uint64_t, uint32_t> const& tidToParticle,
                                    std::unordered_map<uint32_t, float> const& cellSimEnergy,
                                    std::unordered_map<uint32_t, float> const& recHitEnergyByDetId,
                                    BranchHGCalPlots const& plots) const {
  auto recoEnergyOf = [&recHitEnergyByDetId](uint32_t detId) -> double {
    auto it = recHitEnergyByDetId.find(detId);
    return it != recHitEnergyByDetId.end() ? static_cast<double>(it->second) : 0.;
  };
  for (auto const& obj : objects) {
    if (obj.g4Tracks().empty())
      continue;

    const double eta = obj.eta();
    const double pt = obj.pt();
    const double energy = obj.energy();
    if (pt < minPt_ || std::abs(eta) > maxEta_)
      continue;

    auto const& hitsAndFractions = obj.hits_and_fractions();
    if (hitsAndFractions.empty())
      continue;

    // Selected object: fills the efficiency denominator.
    plots.denomEta->Fill(eta);
    plots.denomPt->Fill(pt);
    plots.denomEnergy->Fill(energy);

    auto it = tidToParticle.find(
        truth::simObjectKey(obj.g4Tracks().front().eventId().rawId(), obj.g4Tracks().front().trackId()));
    if (it == tidToParticle.end())
      continue;  // unmapped -> counts as inefficiency
    const uint32_t particleId = it->second;

    // branchEnergy is the total deposited (sim) energy of the Branch subgraph calo hits.
    // branchCellEnergy is its per-cell breakdown. The raw response uses only the object's own cells,
    // so a small object whose trackId maps to a large shower does not inflate the sim ratio.
    std::unordered_map<uint32_t, double> branchCellEnergy;
    double branchEnergy = 0.;
    for (auto const& hit : hitIndex.subgraphHits(truth::HitChannel::Calo, particleId)) {
      branchCellEnergy[hit.detId] += hit.energy;
      branchEnergy += hit.energy;
    }

    std::vector<truth::RecoHit> recoHits;
    recoHits.reserve(hitsAndFractions.size());
    uint32_t shared = 0;
    double totalFraction = 0.;
    double sharedFraction = 0.;
    // Raw energy response inputs, all on the object's cells. The object energy is fraction-weighted
    // (the CaloParticle/SimCluster convention). The Branch energy is its per-cell sim deposit and the
    // whole-cell RecHit energy. The sim and reco responses differ only by the energy scale.
    double objectSimEnergy = 0.;
    double objectRecoEnergy = 0.;
    double branchSimOnObject = 0.;
    double branchRecoOnObject = 0.;
    for (auto const& [detId, fraction] : hitsAndFractions) {
      recoHits.push_back(truth::RecoHit{detId, 1.f, fraction});
      totalFraction += fraction;
      if (auto cs = cellSimEnergy.find(detId); cs != cellSimEnergy.end())
        objectSimEnergy += static_cast<double>(fraction) * static_cast<double>(cs->second);
      objectRecoEnergy += static_cast<double>(fraction) * recoEnergyOf(detId);
      if (auto be = branchCellEnergy.find(detId); be != branchCellEnergy.end()) {
        ++shared;
        sharedFraction += fraction;
        branchSimOnObject += be->second;
        branchRecoOnObject += recoEnergyOf(detId);
      }
    }

    const double completenessHits = static_cast<double>(shared) / hitsAndFractions.size();
    const double purity = branchCellEnergy.empty() ? 0. : static_cast<double>(shared) / branchCellEnergy.size();
    const double completenessEnergy = totalFraction > 0. ? sharedFraction / totalFraction : 0.;
    // Energy containment: Branch subgraph sim-hit energy over the object generator energy.
    // CaloParticle::simEnergy() is not filled, so the ratio includes the sampling fraction
    // and changes with the detector region.
    const double response = energy > 0. ? branchEnergy / energy : 0.;

    plots.purity->Fill(purity);
    plots.completenessHits->Fill(completenessHits);
    plots.completenessEnergy->Fill(completenessEnergy);
    plots.purityVsEta->Fill(eta, purity);
    plots.completenessVsEta->Fill(eta, completenessEnergy);
    if (energy > 0.) {
      plots.energyResponse->Fill(response);
      plots.responseVsEta->Fill(eta, response);
      plots.responseVsEnergy->Fill(energy, response);
    }

    // Raw energy response: the Branch energy on the object's cells over the object's hit energy.
    if (objectSimEnergy > 0.) {
      const double rawSim = branchSimOnObject / objectSimEnergy;
      plots.rawEnergyResponseSim->Fill(rawSim);
      plots.rawResponseSimVsEnergy->Fill(energy, rawSim);
    }
    if (objectRecoEnergy > 0.) {
      const double rawReco = branchRecoOnObject / objectRecoEnergy;
      plots.rawEnergyResponseReco->Fill(rawReco);
      plots.rawResponseRecoVsEnergy->Fill(energy, rawReco);
    }

    // Reproduction efficiency numerator: the best hit match is this particle's Branch.
    // Among equal best scores, the Branch with the fewest subgraph hits wins.
    auto matches = assoc.bestBranches(std::span<const truth::RecoHit>(recoHits));
    if (!matches.empty()) {
      const float bestScore = matches.front().score;
      uint32_t tightest = matches.front().rootParticleId;
      std::size_t tightestSize = hitIndex.subgraphHits(truth::HitChannel::Calo, tightest).size();
      for (auto const& m : matches) {
        if (m.score > bestScore)
          break;
        const std::size_t size = hitIndex.subgraphHits(truth::HitChannel::Calo, m.rootParticleId).size();
        if (size < tightestSize) {
          tightestSize = size;
          tightest = m.rootParticleId;
        }
      }
      if (tightest == particleId) {
        plots.effNumEta->Fill(eta);
        plots.effNumPt->Fill(pt);
        plots.effNumEnergy->Fill(energy);
      }

      // Self-match: the best Branch is the natural (trackId-seeded) one.
      if (tightest == particleId) {
        plots.selfMatchEta->Fill(eta);
        plots.selfMatchPt->Fill(pt);
      }

      // Merge/split: number of distinct Branches that share >=10% of the object's hits.
      // With the SharedHits metric, BranchMatch::sharedEnergy is the shared-cell count.
      const double shareThreshold = 0.1 * static_cast<double>(hitsAndFractions.size());
      uint32_t nSharing = 0;
      for (auto const& m : matches)
        if (static_cast<double>(m.sharedEnergy) >= shareThreshold)
          ++nSharing;
      plots.nSharingBranches->Fill(std::min<uint32_t>(nSharing, 50));

      // Best Branch's purity / completeness / response vs the object.
      std::unordered_set<uint32_t> bestDetIds;
      double bestBranchEnergy = 0.;
      for (auto const& hit : hitIndex.subgraphHits(truth::HitChannel::Calo, tightest)) {
        bestDetIds.insert(hit.detId);
        bestBranchEnergy += hit.energy;
      }
      uint32_t bestShared = 0;
      double bestSharedFraction = 0.;
      for (auto const& [detId, fraction] : hitsAndFractions) {
        if (bestDetIds.count(detId)) {
          ++bestShared;
          bestSharedFraction += fraction;
        }
      }
      plots.bestPurity->Fill(bestDetIds.empty() ? 0. : static_cast<double>(bestShared) / bestDetIds.size());
      plots.bestCompletenessHits->Fill(static_cast<double>(bestShared) / hitsAndFractions.size());
      plots.bestCompletenessEnergy->Fill(totalFraction > 0. ? bestSharedFraction / totalFraction : 0.);
      if (energy > 0.)
        plots.bestResponse->Fill(bestBranchEnergy / energy);
    }
  }
}

std::unordered_map<uint32_t, float> BranchHGCalValidator::collectRecHitEnergyByDetId(edm::Event const& event) const {
  std::unordered_map<uint32_t, float> energies;
  for (uint32_t i = 0; i < hgcalRecHitTokens_.size(); ++i) {
    edm::Handle<HGCRecHitCollection> handle;
    event.getByToken(hgcalRecHitTokens_[i], handle);
    if (!handle.isValid()) {
      edm::LogWarning("BranchHGCalValidator")
          << "Missing HGCRecHit collection " << hgcalRecHitTags_[i].encode() << "; skipping it.";
      continue;
    }
    energies.reserve(energies.size() + handle->size());
    for (auto const& hit : *handle)
      energies.emplace(hit.detid().rawId(), hit.energy());  // keep first for duplicate DetIds
  }
  for (uint32_t i = 0; i < pfRecHitTokens_.size(); ++i) {
    edm::Handle<reco::PFRecHitCollection> handle;
    event.getByToken(pfRecHitTokens_[i], handle);
    if (!handle.isValid()) {
      edm::LogWarning("BranchHGCalValidator")
          << "Missing reco::PFRecHitCollection " << pfRecHitTags_[i].encode() << "; skipping it.";
      continue;
    }
    energies.reserve(energies.size() + handle->size());
    for (auto const& hit : *handle)
      energies.emplace(hit.detId(), hit.energy());
  }
  return energies;
}

void BranchHGCalValidator::dqmAnalyze(edm::Event const& event,
                                      edm::EventSetup const&,
                                      BranchHGCalHistograms const& histograms) const {
  auto const& graph = event.get(graphToken_);
  auto const& raw = event.get(rawToken_);
  auto const& hitIndexProduct = event.get(hitIndexToken_);
  truth::SubgraphHitView hitIndex(hitIndexProduct);

  const auto tidToParticle = buildTrackIdToParticle(graph, raw);
  const auto generations = truth::particleGenerations(graph);
  truth::BranchHitAssociator assoc(hitIndexProduct,
                                   {},
                                   truth::BranchHitAssociator::Metric::SharedHits,
                                   truth::HitChannel::Calo,
                                   /*emptyRootsMeansAll=*/true,
                                   truth::BranchHitAssociator::kAllDetectors,
                                   /*recHitEnergies=*/nullptr,
                                   generations);

  // Per-cell deposited (sim) energy: the sum of the direct Calo hits of all particles in the cell.
  // Each PCaloHit belongs to exactly one SimTrack.
  std::unordered_map<uint32_t, float> cellSimEnergy;
  for (uint32_t p = 0; p < hitIndex.nParticles(); ++p)
    for (auto const& hit : hitIndex.directHits(truth::HitChannel::Calo, p))
      cellSimEnergy[hit.detId] += hit.energy;
  const auto recHitEnergyByDetId = collectRecHitEnergyByDetId(event);

  validate(event.get(caloParticleToken_),
           graph,
           raw,
           hitIndex,
           assoc,
           tidToParticle,
           cellSimEnergy,
           recHitEnergyByDetId,
           histograms.caloParticle);
  validate(event.get(simClusterToken_),
           graph,
           raw,
           hitIndex,
           assoc,
           tidToParticle,
           cellSimEnergy,
           recHitEnergyByDetId,
           histograms.simCluster);
}

void BranchHGCalValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("rawSrc", edm::InputTag("mix"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("simClusters", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<std::string>("folder", "HGCAL/BranchValidator");
  desc.add<double>("minPt", 1.0);
  desc.add<double>("maxEta", 3.0);
  // RecHit collections for the raw (reconstructed) energy response, in the DetIdToRecHitMapProducer order.
  desc.add<std::vector<edm::InputTag>>("hgcalRecHits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  desc.add<std::vector<edm::InputTag>>("pfRecHits",
                                       {edm::InputTag("particleFlowRecHitECAL"),
                                        edm::InputTag("particleFlowRecHitHBHE"),
                                        edm::InputTag("particleFlowRecHitHF"),
                                        edm::InputTag("particleFlowRecHitHO")});
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(BranchHGCalValidator);
