// One producer for all trackster-to-truth-branch associations, following the
// AllTracksterToSimTracksterAssociators pattern: a vector of trackster collections
// in, one pair of association maps (both directions) out per collection, with
// instance labels "<label>ToTruthBranch" / "TruthBranchTo<label>". The branch key
// is the root particle index in the truth::Graph; shared energy and the normalized
// association scores come from truth::BranchHitAssociator over the HGCAL rechit
// channel. Intended as the label source for PID/regression training datasets
// (dumped to NanoAOD by TracksterTruthBranchTableProducer) and for branch-based
// validation.

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
}

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
};

AllTracksterToTruthBranchAssociatorsProducer::AllTracksterToTruthBranchAssociatorsProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      layerClustersToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))),
      branchPdgIds_(cfg.getParameter<std::vector<int>>("branchPdgIds")) {
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("tracksterCollections")) {
    std::string label = tag.label();
    if (!tag.instance().empty())
      label += tag.instance();
    tracksterCollectionTokens_.emplace_back(label, consumes<std::vector<ticl::Trackster>>(tag));
    produces<BranchAssociationMap>(label + "ToTruthBranch");
    produces<BranchAssociationMap>("TruthBranchTo" + label);
  }
}

void AllTracksterToTruthBranchAssociatorsProducer::produce(edm::StreamID,
                                                           edm::Event& event,
                                                           edm::EventSetup const&) const {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  auto const& layerClusters = event.get(layerClustersToken_);

  // Candidate branch roots: the physical shower-initiating species. Intermediate
  // GEN particles (quarks, gluons, decayed resonances) must not be roots, or every
  // trackster also matches all the ancestors of its particle. An empty list keeps
  // every particle (not what training wants; see fillDescriptions).
  // Candidate branch roots: the particles that PHYSICALLY ENTERED the calorimeter
  // (SimTrack tracker-calo boundary checkpoint, id 0), excluding back-scattered
  // re-entries. This is the CaloParticle boundary semantics read off the truth
  // graph, and it is an antichain almost by construction: beam particles never
  // cross, in-calo shower secondaries are born inside and never cross, and a
  // particle interacting or converting BEFORE the calorimeter promotes its
  // crossing products instead. An optional |pdgId| restriction narrows the species.
  std::vector<uint32_t> roots;
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

  truth::BranchHitAssociator assoc(hitIndex,
                                   roots,
                                   truth::BranchHitAssociator::Metric::SharedEnergy,
                                   truth::HitChannel::HGCalCalo,
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

    for (unsigned int t = 0; t < tracksters.size(); ++t) {
      const auto hits = truth::recoHits(tracksters[t], layerClusters);
      if (hits.empty())
        continue;
      for (auto const& m : assoc.bestBranches(std::span<const truth::RecoHit>(hits))) {
        tracksterToBranch->insert(t, m.rootParticleId, m.sharedEnergy, m.score);
        branchToTrackster->insert(m.rootParticleId, t, m.sharedEnergy, m.reverseScore);
      }
    }
    tracksterToBranch->sort(byAscendingScore);
    branchToTrackster->sort(byAscendingScore);
    event.put(std::move(tracksterToBranch), label + "ToTruthBranch");
    event.put(std::move(branchToTrackster), "TruthBranchTo" + label);
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
  desc.add<std::vector<int>>("branchPdgIds", {})
      ->setComment(
          "Optional |pdgId| restriction on the branch roots; empty keeps every "
          "calo-boundary-crossing particle (the training default).");
  descriptions.add("allTrackstersToTruthBranchAssociations", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AllTracksterToTruthBranchAssociatorsProducer);
