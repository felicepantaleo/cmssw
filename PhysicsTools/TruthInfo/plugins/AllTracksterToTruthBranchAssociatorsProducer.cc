// One producer for all trackster-to-truth-branch associations, following the
// AllTracksterToSimTracksterAssociators pattern: a vector of trackster collections
// in, one pair of association maps (both directions) out per collection, with
// instance labels "<label>ToTruthBranch" / "TruthBranchTo<label>". The branch key
// is the root particle index in the truth::Graph; shared energy and the normalized
// association scores come from truth::BranchHitAssociator over the HGCAL rechit
// channel. Intended as the label source for PID/regression training datasets
// (dumped to NanoAOD by TracksterTruthBranchTableProducer) and for branch-based
// validation.

#include <algorithm>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/RecoHitAdapters.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace {
  using BranchAssociationMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;

  // Hadronization ceiling for the adaptive climb: a truth type is a valid stopping
  // level only if it is a real particle (lepton, photon, meson, or baryon). Bare
  // partons (quarks, gluon), diquarks, the string/cluster hadronization nodes, and
  // the electroweak bosons are NOT physical calorimeter objects, so the climb must
  // never select or cross into them; excluding them from the candidate closure caps
  // the climb at the last hadron/lepton/photon (e.g. a pi0 or a rho, not a quark).
  bool isLabelableTruthType(int pdgId) {
    const int a = std::abs(pdgId);
    if (a <= 8)
      return false;  // quarks (and b'/t')
    if (a == 9 || a == 21)
      return false;  // gluon
    if (a == 91 || a == 92)
      return false;  // cluster / string
    if (a == 23 || a == 24 || a == 25 || a == 32 || a == 33 || a == 34 || a == 37)
      return false;  // Z/W/H and extended EWK bosons
    if (a >= 1000 && (a / 100) % 10 == 0)
      return false;  // diquarks (nq3 digit == 0)
    return true;
  }
}  // namespace

class AllTracksterToTruthBranchAssociatorsProducer : public edm::global::EDProducer<> {
public:
  explicit AllTracksterToTruthBranchAssociatorsProducer(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
  std::vector<std::pair<std::string, edm::EDGetTokenT<std::vector<ticl::Trackster>>>> tracksterCollectionTokens_;
  const std::vector<int> branchPdgIds_;
  edm::EDGetTokenT<std::vector<unsigned int>> rootsToken_;
  const bool useExternalRoots_;
  // Adaptive-level match: how strongly the branch-spread (reverse) score counts
  // against climbing up, and the spread ceiling above which a level is rejected.
  const float adaptiveReverseWeight_;
  const float adaptiveMaxReverseScore_;
};

AllTracksterToTruthBranchAssociatorsProducer::AllTracksterToTruthBranchAssociatorsProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      layerClustersToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      branchPdgIds_(cfg.getParameter<std::vector<int>>("branchPdgIds")),
      useExternalRoots_(!cfg.getParameter<edm::InputTag>("rootsSrc").label().empty()),
      adaptiveReverseWeight_(cfg.getParameter<float>("adaptiveReverseWeight")),
      adaptiveMaxReverseScore_(cfg.getParameter<float>("adaptiveMaxReverseScore")) {
  if (useExternalRoots_)
    rootsToken_ = consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("rootsSrc"));
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("tracksterCollections")) {
    std::string label = tag.label();
    if (!tag.instance().empty())
      label += tag.instance();
    tracksterCollectionTokens_.emplace_back(label, consumes<std::vector<ticl::Trackster>>(tag));
    produces<BranchAssociationMap>(label + "ToTruthBranch");
    produces<BranchAssociationMap>("TruthBranchTo" + label);
    // Adaptive-level maps: one best branch per trackster, chosen at the graph
    // level that balances completeness against branch spread.
    produces<BranchAssociationMap>(label + "ToTruthBranchAdaptive");
    produces<BranchAssociationMap>("TruthBranchTo" + label + "Adaptive");
  }
}

void AllTracksterToTruthBranchAssociatorsProducer::produce(edm::StreamID,
                                                           edm::Event& event,
                                                           edm::EventSetup const&) const {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  auto const& layerClusters = event.get(layerClustersToken_);

  // Candidate branch roots. When rootsSrc is set (e.g. the node list of
  // BranchSimTracksterProducer, i.e. every graph level), it wins; otherwise the
  // particles that PHYSICALLY ENTERED the calorimeter (SimTrack tracker-calo
  // boundary checkpoint, id 0), excluding back-scattered re-entries: the
  // CaloParticle boundary semantics read off the truth graph, an antichain almost
  // by construction. An optional |pdgId| restriction narrows the species.
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
      for (auto const& cp : p.checkpoints) {
        if (cp.checkpointId == 0) {
          crossed = true;
          break;
        }
      }
      if (!crossed)
        continue;
      if (!branchPdgIds_.empty()) {
        const int apdg = std::abs(p.pdgId);
        if (std::find(branchPdgIds_.begin(), branchPdgIds_.end(), apdg) == branchPdgIds_.end())
          continue;
      }
      roots.push_back(i);
    }
  }

  // Ancestor closure of the antichain: the leaves plus every ancestor reachable
  // by climbing the graph. This is the candidate set for the adaptive match; the
  // leaves alone reproduce the original fixed-level association. antichain stays
  // sorted so leaf-vs-ancestor membership is a binary search.
  std::vector<uint32_t> antichain = roots;
  std::sort(antichain.begin(), antichain.end());
  antichain.erase(std::unique(antichain.begin(), antichain.end()), antichain.end());

  std::vector<uint32_t> closure = antichain;
  for (const uint32_t r : antichain) {
    if (r >= graph.nParticles())
      continue;
    // Climb only through labelable particle levels; the hadronization ceiling drops
    // partons/strings/bosons so the adaptive match stays on a real calo object.
    for (auto const& a : graph.particle(r).ancestors())
      if (isLabelableTruthType(a.pdgId()))
        closure.push_back(a.id());
  }
  std::sort(closure.begin(), closure.end());
  closure.erase(std::unique(closure.begin(), closure.end()), closure.end());

  auto isAntichain = [&antichain](uint32_t id) { return std::binary_search(antichain.begin(), antichain.end(), id); };

  truth::BranchHitAssociator assoc(hitIndex,
                                   closure,
                                   truth::BranchHitAssociator::Metric::SharedEnergy,
                                   truth::HitChannel::Calo,
                                   /*emptyRootsMeansAll=*/false);

  auto byAscendingScore = [](auto const& a, auto const& b) {
    if (a.score() != b.score())
      return a.score() < b.score();
    return a.index() < b.index();
  };

  for (auto const& [label, token] : tracksterCollectionTokens_) {
    auto const& tracksters = event.get(token);
    auto tracksterToBranch = std::make_unique<BranchAssociationMap>(static_cast<unsigned int>(tracksters.size()));
    auto branchToTrackster = std::make_unique<BranchAssociationMap>(graph.nParticles());
    auto tracksterToBranchAdaptive =
        std::make_unique<BranchAssociationMap>(static_cast<unsigned int>(tracksters.size()));
    auto branchToTracksterAdaptive = std::make_unique<BranchAssociationMap>(graph.nParticles());

    for (unsigned int t = 0; t < tracksters.size(); ++t) {
      const auto hits = truth::recoHits(tracksters[t], layerClusters);
      if (hits.empty())
        continue;
      const std::span<const truth::RecoHit> hitSpan(hits);

      // Fixed-level (leaf) association: all matches restricted to the antichain,
      // identical to the original single-associator behavior.
      double bestObj = std::numeric_limits<double>::infinity();
      truth::BranchMatch adaptive;
      adaptive.rootParticleId = truth::BranchMatch::kInvalidRoot;
      for (auto const& m : assoc.bestBranches(hitSpan)) {
        if (isAntichain(m.rootParticleId)) {
          tracksterToBranch->insert(t, m.rootParticleId, m.sharedEnergy, m.score);
          branchToTrackster->insert(m.rootParticleId, t, m.sharedEnergy, m.reverseScore);
        }
        // Adaptive level: argmin over ALL levels (leaves and ancestors) of the
        // balanced objective, under the branch-spread ceiling.
        if (m.reverseScore <= adaptiveMaxReverseScore_) {
          const double obj = static_cast<double>(m.score) + adaptiveReverseWeight_ * m.reverseScore;
          if (obj < bestObj) {
            bestObj = obj;
            adaptive = m;
          }
        }
      }
      if (adaptive.rootParticleId != truth::BranchMatch::kInvalidRoot) {
        tracksterToBranchAdaptive->insert(t, adaptive.rootParticleId, adaptive.sharedEnergy, adaptive.score);
        branchToTracksterAdaptive->insert(adaptive.rootParticleId, t, adaptive.sharedEnergy, adaptive.reverseScore);
      }
    }
    tracksterToBranch->sort(byAscendingScore);
    branchToTrackster->sort(byAscendingScore);
    tracksterToBranchAdaptive->sort(byAscendingScore);
    branchToTracksterAdaptive->sort(byAscendingScore);
    event.put(std::move(tracksterToBranch), label + "ToTruthBranch");
    event.put(std::move(branchToTrackster), "TruthBranchTo" + label);
    event.put(std::move(tracksterToBranchAdaptive), label + "ToTruthBranchAdaptive");
    event.put(std::move(branchToTracksterAdaptive), "TruthBranchTo" + label + "Adaptive");
  }
}

void AllTracksterToTruthBranchAssociatorsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<std::vector<edm::InputTag>>(
      "tracksterCollections",
      {edm::InputTag("ticlTrackstersCLUE3DHigh"), edm::InputTag("ticlTracksterInterpretations")});
  desc.add<edm::InputTag>("rootsSrc", edm::InputTag(""))
      ->setComment("Optional external root list (vector<unsigned int>); overrides the boundary selection.");
  desc.add<std::vector<int>>("branchPdgIds", {})
      ->setComment(
          "Optional |pdgId| restriction on the branch roots; empty keeps every "
          "calo-boundary-crossing particle (the training default).");
  desc.add<float>("adaptiveReverseWeight", 1.f)
      ->setComment(
          "Weight of the branch-spread (reverse) score in the adaptive-level "
          "objective score + w*reverseScore; higher penalizes climbing up.");
  desc.add<float>("adaptiveMaxReverseScore", 1.f)
      ->setComment(
          "Reject adaptive levels whose branch-spread (reverse) score exceeds this "
          "ceiling. The reverse score is not bounded by 1: a branch can spread into "
          "more energy than the object carries, which is what a ceiling above 1 admits.");
  descriptions.add("allTrackstersToTruthBranchAssociations", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AllTracksterToTruthBranchAssociatorsProducer);
