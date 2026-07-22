// Parallel "by hits" trackster-to-truth-branch associations (distinct instance labels
// <label>ToTruthBranchByHits...), kept alongside the BranchHitAssociator default for
// comparison at training time. Built by COMPOSING
// existing ticl association maps instead of a per-trackster hit merge-join (no
// combinatorics). The truth side is the LogicalGraphHitIndex (branch subtree ->
// rechits, each with its global rechit index); the reco side is the per-collection
// hitToTrackster map (rechit -> trackster + fraction) from AllHitToTracksterAssociators.
// Walking each candidate branch's subtree hits once and following the rechit into its
// tracksters yields the branch<->trackster shared energy directly in O(truth hits), with
// no bestBranches call per trackster. The rechit index is shared: recHitMapProducer
// (which keys the hitToTrackster map) and detIdToRecHitMapProducer (which sets the hit
// index recHitIndex) both concatenate HGCEE,HGCHEF,HGCHEB in the same order.
//
// One pair of maps per collection and direction, fixed (leaf/antichain) and adaptive
// (best graph level), instance labels "<label>ToTruthBranch" / "TruthBranchTo<label>"
// and the ...Adaptive variants. Shared energy is the trackster's reconstructed rechit
// energy on the branch's cells; score is the reco-normalized coverage failure
// (1 - shared/tracksterEnergy) and reverseScore the branch-normalized one
// (1 - shared/branchDeposit). The adaptive level minimizes score + w*reverseScore.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace {
  using BranchAssociationMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;
  using HitToTracksterMap = ticl::AssociationMap<ticl::mapWithFraction>;

  // Hadronization ceiling for the adaptive climb: stop only on real particles, never on
  // partons/gluon/diquarks/strings/EWK bosons, so the adaptive match stays on a real
  // calorimeter object (e.g. a pi0 or a rho, not a quark).
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

class AllTracksterToTruthBranchByHitsAssociatorsProducer : public edm::global::EDProducer<> {
public:
  explicit AllTracksterToTruthBranchByHitsAssociatorsProducer(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  struct Collection {
    std::string label;
    edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksterToken;
    edm::EDGetTokenT<HitToTracksterMap> hitToTracksterToken;
  };

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  std::vector<Collection> collections_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> recHitTokens_;
  const std::vector<int> branchPdgIds_;
  edm::EDGetTokenT<std::vector<unsigned int>> rootsToken_;
  const bool useExternalRoots_;
  const double adaptiveReverseWeight_;
  const double adaptiveMaxReverseScore_;
};

AllTracksterToTruthBranchByHitsAssociatorsProducer::AllTracksterToTruthBranchByHitsAssociatorsProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      branchPdgIds_(cfg.getParameter<std::vector<int>>("branchPdgIds")),
      useExternalRoots_(!cfg.getParameter<edm::InputTag>("rootsSrc").label().empty()),
      adaptiveReverseWeight_(cfg.getParameter<double>("adaptiveReverseWeight")),
      adaptiveMaxReverseScore_(cfg.getParameter<double>("adaptiveMaxReverseScore")) {
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("recHits"))
    recHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  if (useExternalRoots_)
    rootsToken_ = consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("rootsSrc"));
  const std::string hitToTracksterProducer = cfg.getParameter<std::string>("hitToTracksterProducer");
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("tracksterCollections")) {
    std::string label = tag.label();
    if (!tag.instance().empty())
      label += tag.instance();
    collections_.push_back(Collection{label,
                                      consumes<std::vector<ticl::Trackster>>(tag),
                                      consumes<HitToTracksterMap>(edm::InputTag(hitToTracksterProducer, "hitTo" + label))});
    produces<BranchAssociationMap>(label + "ToTruthBranchByHits");
    produces<BranchAssociationMap>("TruthBranchTo" + label + "ByHits");
    produces<BranchAssociationMap>(label + "ToTruthBranchByHitsAdaptive");
    produces<BranchAssociationMap>("TruthBranchTo" + label + "ByHitsAdaptive");
  }
}

void AllTracksterToTruthBranchByHitsAssociatorsProducer::produce(edm::StreamID,
                                                           edm::Event& event,
                                                           edm::EventSetup const&) const {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);

  std::vector<float> rechitEnergy;  // indexed by global rechit index (HGCAL concatenation)
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

  auto byAscendingScore = [](auto const& a, auto const& b) {
    if (a.score() != b.score())
      return a.score() < b.score();
    return a.index() < b.index();
  };

  struct Best {
    double obj = std::numeric_limits<double>::infinity();
    uint32_t root = 0;
    float shared = 0.f, score = 0.f, rscore = 0.f;
    bool set = false;
  };

  for (auto const& coll : collections_) {
    auto const& tracksters = event.get(coll.tracksterToken);
    auto const& hitToTrackster = event.get(coll.hitToTracksterToken);
    const unsigned nT = tracksters.size();
    auto tToBranch = std::make_unique<BranchAssociationMap>(nT);
    auto branchToT = std::make_unique<BranchAssociationMap>(graph.nParticles());
    auto tToBranchAdaptive = std::make_unique<BranchAssociationMap>(nT);
    auto branchToTAdaptive = std::make_unique<BranchAssociationMap>(graph.nParticles());

    std::vector<Best> best(nT);
    std::vector<double> sharedScratch(nT, 0.0);
    std::vector<uint32_t> touched;

    for (const uint32_t r : closure) {
      if (r >= hitIndex.nParticles())
        continue;
      double branchDep = 0.0;
      touched.clear();
      for (auto const& h : hitIndex.subgraphHits(truth::HitChannel::HGCalCalo, r)) {
        if (!h.hasRecHit() || h.recHitIndex >= rechitEnergy.size())
          continue;
        const double e = rechitEnergy[h.recHitIndex];
        branchDep += e;
        if (h.recHitIndex >= hitToTrackster.size())
          continue;
        for (auto const& el : hitToTrackster[h.recHitIndex]) {
          const unsigned trk = el.index();
          if (trk >= nT)
            continue;
          if (sharedScratch[trk] == 0.0)
            touched.push_back(trk);
          sharedScratch[trk] += e * el.fraction();
        }
      }
      const bool anti = isAntichain(r);
      for (const uint32_t trk : touched) {
        const double shared = sharedScratch[trk];
        sharedScratch[trk] = 0.0;
        if (branchDep <= 0.0 || shared <= 0.0)
          continue;
        const double trkE = tracksters[trk].raw_energy();
        const float score = trkE > 0.0 ? static_cast<float>(std::max(0.0, 1.0 - shared / trkE)) : 1.f;
        const float rscore = static_cast<float>(std::max(0.0, 1.0 - shared / branchDep));
        if (anti) {
          tToBranch->insert(trk, r, static_cast<float>(shared), score);
          branchToT->insert(r, trk, static_cast<float>(shared), rscore);
        }
        if (rscore <= adaptiveMaxReverseScore_) {
          const double obj = static_cast<double>(score) + adaptiveReverseWeight_ * rscore;
          if (obj < best[trk].obj)
            best[trk] = Best{obj, r, static_cast<float>(shared), score, rscore, true};
        }
      }
    }
    for (unsigned trk = 0; trk < nT; ++trk) {
      if (!best[trk].set)
        continue;
      tToBranchAdaptive->insert(trk, best[trk].root, best[trk].shared, best[trk].score);
      branchToTAdaptive->insert(best[trk].root, trk, best[trk].shared, best[trk].rscore);
    }
    tToBranch->sort(byAscendingScore);
    branchToT->sort(byAscendingScore);
    tToBranchAdaptive->sort(byAscendingScore);
    branchToTAdaptive->sort(byAscendingScore);
    event.put(std::move(tToBranch), coll.label + "ToTruthBranchByHits");
    event.put(std::move(branchToT), "TruthBranchTo" + coll.label + "ByHits");
    event.put(std::move(tToBranchAdaptive), coll.label + "ToTruthBranchByHitsAdaptive");
    event.put(std::move(branchToTAdaptive), "TruthBranchTo" + coll.label + "ByHitsAdaptive");
  }
}

void AllTracksterToTruthBranchByHitsAssociatorsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<std::vector<edm::InputTag>>(
      "tracksterCollections",
      {edm::InputTag("ticlTrackstersCLUE3DHigh"), edm::InputTag("ticlTracksterInterpretations")});
  desc.add<std::string>("hitToTracksterProducer", "allHitToTracksterAssociations")
      ->setComment("Module producing the per-collection hitTo<label> maps (AllHitToTracksterAssociatorsProducer).");
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
  descriptions.add("allTrackstersToTruthBranchByHitsAssociations", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(AllTracksterToTruthBranchByHitsAssociatorsProducer);
