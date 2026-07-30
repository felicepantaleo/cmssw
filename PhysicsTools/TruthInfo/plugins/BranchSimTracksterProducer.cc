// Materializes truth branches as SimTrackster collections at every level of the
// graph: level 0 is the calo-entering antichain (one trackster per particle that
// crossed the tracker-calo boundary, its whole shower), and each higher level is
// an ancestor that MERGES at least two selected nodes (a pi0 over its photons, a
// K0S over its pions, a D0 or tau over its products, up to the hard-interaction
// legs). Each trackster's vertices are the layer clusters touched by the node's
// subgraph hits, so the whole SimTrackster toolchain (associators, dumper,
// tables) works against any level. Parallel products carry the node metadata
// (level, rootId, pdgId) and the full root list, which
// AllTracksterToTruthBranchAssociatorsProducer can consume (rootsSrc) to produce
// reco associations against every level at once.

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "DataFormats/HepMCCandidate/interface/GenStatusFlags.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace {
  int chargeFromPdgId(int pdgId) {
    const int a = std::abs(pdgId);
    if (a == 11 || a == 13 || a == 15 || a == 211 || a == 321 || a == 2212 || a == 3112 || a == 3222)
      return pdgId > 0 ? 1 : -1;
    return 0;
  }
}  // namespace

class BranchSimTracksterProducer : public edm::global::EDProducer<> {
public:
  explicit BranchSimTracksterProducer(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<reco::CaloCluster>> layerClustersToken_;
};

BranchSimTracksterProducer::BranchSimTracksterProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      layerClustersToken_(consumes<std::vector<reco::CaloCluster>>(cfg.getParameter<edm::InputTag>("layerClusters"))) {
  produces<std::vector<ticl::Trackster>>();
  produces<std::vector<int>>("level");
  produces<std::vector<int>>("rootId");
  produces<std::vector<int>>("pdgId");
  produces<std::vector<unsigned int>>("roots");
}

void BranchSimTracksterProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);
  auto const& layerClusters = event.get(layerClustersToken_);

  // Level 0: the calo-entering antichain (boundary checkpoint, no back-scatter)
  // with a calorimeter footprint.
  std::vector<uint32_t> leaves;
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
    if (hitIndex.subgraphHits(truth::HitChannel::Calo, i).empty())
      continue;
    leaves.push_back(i);
  }

  // Merge nodes: ancestors reached by at least two distinct leaves. Count leaf
  // visits, guarding against multiple paths from the same leaf.
  std::unordered_map<uint32_t, int> leafCount;
  for (uint32_t leaf : leaves) {
    std::vector<bool> seen(graph.nParticles(), false);
    std::vector<uint32_t> stack{leaf};
    seen[leaf] = true;
    while (!stack.empty()) {
      const uint32_t n = stack.back();
      stack.pop_back();
      for (auto const& par : truth::Particle(&graph, n).parents()) {
        if (seen[par.id()])
          continue;
        seen[par.id()] = true;
        ++leafCount[par.id()];
        stack.push_back(par.id());
      }
    }
  }
  // A merge node must be a physical particle (a decay or interaction ancestor:
  // pi0, K0S, D, tau, ...). The parton SHOWER is excluded, or every gluon
  // splitting would add a nested, nearly identical jet-level closure; only the
  // hard-process legs themselves (the signal partons) are kept as the top level.
  auto keepMergeNode = [&graph](uint32_t node) {
    auto const& p = graph.particles()[node];
    const int a = std::abs(p.pdgId);
    const bool partonic = (a == 0 || a <= 8 || a == 21);
    if (!partonic)
      return true;
    return ((p.statusFlags >> reco::GenStatusFlags::kIsHardProcess) & 1) != 0;
  };
  std::vector<uint32_t> mergeNodes;
  for (auto const& [node, count] : leafCount) {
    if (count >= 2 && !truth::Particle(&graph, node).parents().empty() && keepMergeNode(node))
      mergeNodes.push_back(node);
  }
  std::sort(mergeNodes.begin(), mergeNodes.end());

  // Levels: leaves are 0; a merge node is one above the highest selected node
  // among its descendants. Process merge nodes by increasing selected-descendant
  // count so dependencies are resolved before use.
  std::unordered_map<uint32_t, int> level;
  for (uint32_t leaf : leaves)
    level[leaf] = 0;
  std::vector<std::pair<uint32_t, std::vector<uint32_t>>> mergeDesc;
  for (uint32_t m : mergeNodes) {
    std::vector<uint32_t> sel;
    for (auto const& d : truth::Particle(&graph, m).descendants()) {
      if (level.count(d.id()) || std::binary_search(mergeNodes.begin(), mergeNodes.end(), d.id()))
        sel.push_back(d.id());
    }
    mergeDesc.emplace_back(m, std::move(sel));
  }
  std::sort(mergeDesc.begin(), mergeDesc.end(), [](auto const& a, auto const& b) {
    return a.second.size() < b.second.size();
  });
  for (auto const& [m, sel] : mergeDesc) {
    int maxBelow = -1;
    for (uint32_t d : sel) {
      auto it = level.find(d);
      if (it != level.end())
        maxBelow = std::max(maxBelow, it->second);
    }
    level[m] = maxBelow + 1;
  }

  // detId -> (layer cluster, fraction), for footprint-to-LC conversion.
  std::unordered_map<uint32_t, std::vector<std::pair<unsigned int, float>>> detIdToLc;
  for (unsigned int lc = 0; lc < layerClusters.size(); ++lc)
    for (auto const& [detId, frac] : layerClusters[lc].hitsAndFractions())
      detIdToLc[detId.rawId()].emplace_back(lc, frac);

  auto tracksters = std::make_unique<std::vector<ticl::Trackster>>();
  auto levels = std::make_unique<std::vector<int>>();
  auto rootIds = std::make_unique<std::vector<int>>();
  auto pdgIds = std::make_unique<std::vector<int>>();
  auto roots = std::make_unique<std::vector<unsigned int>>();

  std::vector<uint32_t> nodes = leaves;
  nodes.insert(nodes.end(), mergeNodes.begin(), mergeNodes.end());
  for (uint32_t node : nodes) {
    std::unordered_map<unsigned int, float> lcShared;
    for (auto const& hit : hitIndex.subgraphHits(truth::HitChannel::Calo, node)) {
      auto it = detIdToLc.find(hit.detId);
      if (it == detIdToLc.end())
        continue;
      for (auto const& [lc, frac] : it->second)
        lcShared[lc] += hit.energy * frac;
    }
    if (lcShared.empty())
      continue;

    ticl::Trackster ts;
    std::vector<std::pair<unsigned int, float>> ordered(lcShared.begin(), lcShared.end());
    std::sort(ordered.begin(), ordered.end());
    float raw = 0.f;
    math::XYZVector bary(0, 0, 0);
    for (auto const& [lc, shared] : ordered) {
      ts.vertices().push_back(lc);
      const float lcE = static_cast<float>(layerClusters[lc].energy());
      ts.vertex_multiplicity().push_back(shared > 0.f ? std::max(1.f, lcE / shared) : 1.f);
      raw += shared;
      bary += math::XYZVector(layerClusters[lc].position()) * shared;
    }
    ts.setRawEnergy(raw);
    if (raw > 0.f)
      ts.setBarycenter(ticl::Trackster::Vector(bary / raw));

    auto const& p = graph.particles()[node];
    ts.setRegressedEnergy(static_cast<float>(p.momentum.energy()));
    const auto type = ticl::tracksterParticleTypeFromPdgId(p.pdgId, chargeFromPdgId(p.pdgId));
    ts.setIdProbability(type, 1.f);
    if (type == ticl::Trackster::ParticleType::photon || type == ticl::Trackster::ParticleType::electron)
      ts.setRawEmEnergy(raw);
    ts.setSeed(edm::ProductID(), static_cast<int>(node));
    ts.setIteration(ticl::Trackster::IterationIndex::SIM);

    tracksters->push_back(std::move(ts));
    levels->push_back(level[node]);
    rootIds->push_back(static_cast<int>(node));
    pdgIds->push_back(p.pdgId);
    roots->push_back(node);
  }

  event.put(std::move(tracksters));
  event.put(std::move(levels), "level");
  event.put(std::move(rootIds), "rootId");
  event.put(std::move(pdgIds), "pdgId");
  event.put(std::move(roots), "roots");
}

void BranchSimTracksterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  descriptions.add("branchSimTracksters", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BranchSimTracksterProducer);
