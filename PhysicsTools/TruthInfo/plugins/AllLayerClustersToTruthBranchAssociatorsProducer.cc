// Layer-cluster to truth-branch associations, built by COMPOSING existing ticl
// association maps instead of a per-LC hit merge-join (no combinatorics). The truth
// side is the LogicalGraphHitIndex (branch subtree -> rechits, each with its global
// rechit index); the reco side is hitToLayerClusterMap (rechit -> layer cluster +
// fraction). Walking each candidate branch's subtree hits once and following the
// rechit into its layer cluster yields the branch<->LC shared energy directly in
// O(truth hits), with no bestBranches call per layer cluster. The rechit index is
// shared: recHitMapProducer (which keys hitToLayerClusterMap) and detIdToRecHitMapProducer
// (which sets the hit index recHitIndex) both concatenate HGCEE,HGCHEF,HGCHEB in the
// same order, so the HGCAL indices coincide; the rechit energy vector is filled in the
// same order.
//
// Output mirrors AllTracksterToTruthBranchAssociatorsProducer: a pair of maps per
// direction, fixed (leaf/antichain) and adaptive (best graph level), instance labels
// "<lcLabel>ToTruthBranch" / "TruthBranchTo<lcLabel>" and the ...Adaptive variants.
// Shared energy is the layer cluster's reconstructed rechit energy on the branch's
// cells; score is the reco-normalized coverage failure (1 - shared/lcEnergy) and
// reverseScore the branch-normalized one (1 - shared/branchDeposit).

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace {
  using BranchAssociationMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;
  using HitToLCMap = ticl::AssociationMap<ticl::mapWithFraction>;

  // Hadronization ceiling for the adaptive climb (same as the trackster associator):
  // stop only on real particles, never on partons/gluon/diquarks/strings/EWK bosons.
  bool isLabelableTruthType(int pdgId) {
    const int a = std::abs(pdgId);
    if (a <= 8)
      return false;  // quarks
    if (a == 9 || a == 21)
      return false;  // gluon
    if (a == 91 || a == 92)
      return false;  // cluster / string
    if (a == 23 || a == 24 || a == 25 || a == 32 || a == 33 || a == 34 || a == 37)
      return false;  // Z/W/H and extended EWK bosons
    if (a >= 1000 && (a / 100) % 10 == 0)
      return false;  // diquarks
    return true;
  }
}  // namespace

class AllLayerClustersToTruthBranchAssociatorsProducer : public edm::global::EDProducer<> {
public:
  explicit AllLayerClustersToTruthBranchAssociatorsProducer(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<HitToLCMap> hitToLCToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> recHitTokens_;
  const std::string lcLabel_;
  const std::vector<int> branchPdgIds_;
  edm::EDGetTokenT<std::vector<unsigned int>> rootsToken_;
  const bool useExternalRoots_;
  const double adaptiveReverseWeight_;
  const double adaptiveMaxReverseScore_;
};

AllLayerClustersToTruthBranchAssociatorsProducer::AllLayerClustersToTruthBranchAssociatorsProducer(
    edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      hitToLCToken_(consumes<HitToLCMap>(cfg.getParameter<edm::InputTag>("hitToLayerClusterMap"))),
      layerClustersToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      lcLabel_([&] {
        auto const& tag = cfg.getParameter<edm::InputTag>("layerClusters");
        return tag.instance().empty() ? tag.label() : tag.label() + tag.instance();
      }()),
      branchPdgIds_(cfg.getParameter<std::vector<int>>("branchPdgIds")),
      useExternalRoots_(!cfg.getParameter<edm::InputTag>("rootsSrc").label().empty()),
      adaptiveReverseWeight_(cfg.getParameter<double>("adaptiveReverseWeight")),
      adaptiveMaxReverseScore_(cfg.getParameter<double>("adaptiveMaxReverseScore")) {
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("recHits"))
    recHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  if (useExternalRoots_)
    rootsToken_ = consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("rootsSrc"));
  produces<BranchAssociationMap>(lcLabel_ + "ToTruthBranch");
  produces<BranchAssociationMap>("TruthBranchTo" + lcLabel_);
  produces<BranchAssociationMap>(lcLabel_ + "ToTruthBranchAdaptive");
  produces<BranchAssociationMap>("TruthBranchTo" + lcLabel_ + "Adaptive");
}

void AllLayerClustersToTruthBranchAssociatorsProducer::produce(edm::StreamID,
                                                               edm::Event& event,
                                                               edm::EventSetup const&) const {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  auto const& hitToLC = event.get(hitToLCToken_);
  auto const& layerClusters = event.get(layerClustersToken_);
  const unsigned nLC = layerClusters.size();

  // Rechit energy per global rechit index (HGCEE,HGCHEF,HGCHEB concatenation): the same
  // index the branch hit index and hitToLayerClusterMap use.
  std::vector<float> rechitEnergy;
  for (auto const& token : recHitTokens_) {
    auto const& hits = event.get(token);
    rechitEnergy.reserve(rechitEnergy.size() + hits.size());
    for (auto const& rh : hits)
      rechitEnergy.push_back(rh.energy());
  }

  // Candidate branch roots (default: calo-boundary-crossing particles, an antichain).
  std::vector<uint32_t> roots;
  if (useExternalRoots_) {
    for (unsigned int r : event.get(rootsToken_))
      roots.push_back(r);
  } else {
    for (uint32_t i = 0; i < graph.nParticles(); ++i) {
      auto const& p = graph.particles()[i];
      if (p.backscattered)
        continue;
      bool crossed = false;
      for (auto const& cp : p.checkpoints)
        if (cp.checkpointId == 0) {
          crossed = true;
          break;
        }
      if (!crossed)
        continue;
      if (!branchPdgIds_.empty() &&
          std::find(branchPdgIds_.begin(), branchPdgIds_.end(), std::abs(p.pdgId)) == branchPdgIds_.end())
        continue;
      roots.push_back(i);
    }
  }
  std::vector<uint32_t> antichain = roots;
  std::sort(antichain.begin(), antichain.end());
  antichain.erase(std::unique(antichain.begin(), antichain.end()), antichain.end());

  std::vector<uint32_t> closure = antichain;
  for (const uint32_t r : antichain) {
    if (r >= graph.nParticles())
      continue;
    for (auto const& a : graph.particle(r).ancestors())
      if (isLabelableTruthType(a.pdgId()))
        closure.push_back(a.id());
  }
  std::sort(closure.begin(), closure.end());
  closure.erase(std::unique(closure.begin(), closure.end()), closure.end());
  auto isAntichain = [&antichain](uint32_t id) {
    return std::binary_search(antichain.begin(), antichain.end(), id);
  };

  auto lcToBranch = std::make_unique<BranchAssociationMap>(nLC);
  auto branchToLc = std::make_unique<BranchAssociationMap>(graph.nParticles());
  auto lcToBranchAdaptive = std::make_unique<BranchAssociationMap>(nLC);
  auto branchToLcAdaptive = std::make_unique<BranchAssociationMap>(graph.nParticles());

  // Per-LC best adaptive level, accumulated incrementally across the closure so no
  // per-LC list of all levels is stored.
  struct Best {
    double obj = std::numeric_limits<double>::infinity();
    uint32_t root = 0;
    float shared = 0.f, score = 0.f, rscore = 0.f;
    bool set = false;
  };
  std::vector<Best> best(nLC);
  std::vector<double> sharedScratch(nLC, 0.0);
  std::vector<uint32_t> touched;

  for (const uint32_t r : closure) {
    if (r >= hitIndex.nParticles())
      continue;
    double branchDep = 0.0;
    touched.clear();
    for (auto const& h : hitIndex.subgraphHits(truth::HitChannel::HGCalCalo, r)) {
      if (!h.hasRecHit() || h.recHitIndex >= rechitEnergy.size())
        continue;  // branch sim hit with no associated rechit
      const double e = rechitEnergy[h.recHitIndex];
      branchDep += e;
      if (h.recHitIndex >= hitToLC.size())
        continue;
      for (auto const& el : hitToLC[h.recHitIndex]) {
        const unsigned lc = el.index();
        if (lc >= nLC)
          continue;  // rechit not in any layer cluster (or out of range)
        if (sharedScratch[lc] == 0.0)
          touched.push_back(lc);
        sharedScratch[lc] += e * el.fraction();
      }
    }
    const bool anti = isAntichain(r);
    for (const uint32_t lc : touched) {
      const double shared = sharedScratch[lc];
      sharedScratch[lc] = 0.0;
      if (branchDep <= 0.0 || shared <= 0.0)
        continue;
      const double lcE = layerClusters[lc].energy();
      const float score = lcE > 0.0 ? static_cast<float>(std::max(0.0, 1.0 - shared / lcE)) : 1.f;
      const float rscore = static_cast<float>(std::max(0.0, 1.0 - shared / branchDep));
      if (anti) {
        lcToBranch->insert(lc, r, static_cast<float>(shared), score);
        branchToLc->insert(r, lc, static_cast<float>(shared), rscore);
      }
      if (rscore <= adaptiveMaxReverseScore_) {
        const double obj = static_cast<double>(score) + adaptiveReverseWeight_ * rscore;
        if (obj < best[lc].obj)
          best[lc] = Best{obj, r, static_cast<float>(shared), score, rscore, true};
      }
    }
  }
  for (unsigned lc = 0; lc < nLC; ++lc) {
    if (!best[lc].set)
      continue;
    lcToBranchAdaptive->insert(lc, best[lc].root, best[lc].shared, best[lc].score);
    branchToLcAdaptive->insert(best[lc].root, lc, best[lc].shared, best[lc].rscore);
  }

  auto byAscendingScore = [](auto const& a, auto const& b) {
    if (a.score() != b.score())
      return a.score() < b.score();
    return a.index() < b.index();
  };
  lcToBranch->sort(byAscendingScore);
  branchToLc->sort(byAscendingScore);
  lcToBranchAdaptive->sort(byAscendingScore);
  branchToLcAdaptive->sort(byAscendingScore);
  event.put(std::move(lcToBranch), lcLabel_ + "ToTruthBranch");
  event.put(std::move(branchToLc), "TruthBranchTo" + lcLabel_);
  event.put(std::move(lcToBranchAdaptive), lcLabel_ + "ToTruthBranchAdaptive");
  event.put(std::move(branchToLcAdaptive), "TruthBranchTo" + lcLabel_ + "Adaptive");
}

void AllLayerClustersToTruthBranchAssociatorsProducer::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("hitToLayerClusterMap",
                          edm::InputTag("hitToLayerClusterAssociator", "hitToLayerClusterMap"));
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<std::vector<edm::InputTag>>("recHits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  desc.add<edm::InputTag>("rootsSrc", edm::InputTag(""))
      ->setComment("Optional external root list (vector<unsigned int>); overrides the boundary selection.");
  desc.add<std::vector<int>>("branchPdgIds", {})
      ->setComment("Optional |pdgId| restriction on the branch roots; empty keeps every calo-boundary crosser.");
  desc.add<double>("adaptiveReverseWeight", 1.0)
      ->setComment("Weight of the branch-spread (reverse) score in the adaptive objective score + w*reverseScore.");
  desc.add<double>("adaptiveMaxReverseScore", 1.0)
      ->setComment("Reject adaptive levels whose reverse score exceeds this ceiling; 1.0 keeps every level.");
  descriptions.add("allLayerClustersToTruthBranchAssociations", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AllLayerClustersToTruthBranchAssociatorsProducer);
