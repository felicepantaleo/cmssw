// Root-cause diagnostic: for the merged HGCal sim-hits, how many hit-producing
// SimTracks (and how many hits) attach to a logical-graph particle, split by
// signal vs pileup. Replicates the hit-index key logic exactly. Not production.
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"
#include <unordered_map>
#include <unordered_set>
#include <iostream>

class MergedVsIndexAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit MergedVsIndexAnalyzer(edm::ParameterSet const& p)
      : hitsTok_(consumes<std::vector<PCaloHit>>(p.getParameter<edm::InputTag>("mergedHits"))),
        gTok_(consumes<truth::Graph>(p.getParameter<edm::InputTag>("graph"))),
        rgTok_(consumes<TruthGraph>(p.getParameter<edm::InputTag>("rawGraph"))) {}
  static uint64_t key(uint64_t eventId, uint32_t trackId) { return (eventId << 32) | uint64_t(trackId); }
  void analyze(edm::Event const& e, edm::EventSetup const&) override {
    auto const& hits = e.get(hitsTok_);
    auto const& g = e.get(gTok_);
    auto const& rg = e.get(rgTok_);
    const uint32_t nN = rg.nNodes();
    // particle primary-track node -> particle (the OLD registration)
    std::unordered_map<uint32_t, uint32_t> simNodeToParticle;
    for (uint32_t i = 0; i < g.nParticles(); ++i) {
      auto const& pd = g.particles()[i];
      if (!pd.hasSim() || pd.simNode < 0 || uint32_t(pd.simNode) >= nN)
        continue;
      if (rg.nodeRef(uint32_t(pd.simNode)).kind != TruthGraph::NodeKind::SimTrack)
        continue;
      simNodeToParticle.emplace(uint32_t(pd.simNode), i);
    }
    // (eventId,trackId) -> raw node, and vertex -> parent track (invert decay edges)
    std::unordered_map<uint64_t, uint32_t> keyToNode;
    std::vector<int32_t> vertexParent(nN, -1);
    for (uint32_t n = 0; n < nN; ++n) {
      auto const& ref = rg.nodeRef(n);
      if (ref.kind != TruthGraph::NodeKind::SimTrack)
        continue;
      if (ref.key > 0)
        keyToNode[key(rg.nodeEventId(n), uint32_t(ref.key))] = n;
      for (uint32_t c : rg.children(n))
        if (c < nN && rg.nodeRef(c).kind == TruthGraph::NodeKind::SimVertex)
          vertexParent[c] = int32_t(n);
    }
    // nearest-ancestor particle (the NEW roll-up), memoized
    std::vector<int32_t> owner(nN, -2);
    auto resolve = [&](uint32_t start) -> int32_t {
      std::vector<uint32_t> chain;
      int32_t o = -1;
      uint32_t cur = start;
      for (uint32_t guard = 0; guard <= nN; ++guard) {
        if (owner[cur] != -2) {
          o = owner[cur];
          break;
        }
        auto it = simNodeToParticle.find(cur);
        if (it != simNodeToParticle.end()) {
          o = int32_t(it->second);
          break;
        }
        chain.push_back(cur);
        int32_t v = rg.nodeSimTrackToVtx(cur);
        if (v < 0 || uint32_t(v) >= nN || vertexParent[uint32_t(v)] < 0) {
          o = -1;
          break;
        }
        cur = uint32_t(vertexParent[uint32_t(v)]);
      }
      for (uint32_t c : chain)
        owner[c] = o;
      return o;
    };
    // per hit: raw-hit and energy attachment, old (primary) vs new (roll-up)
    double sigE = 0, puE = 0, sigEold = 0, puEold = 0, sigEnew = 0, puEnew = 0;
    unsigned long sigH = 0, puH = 0, sigHold = 0, puHold = 0, sigHnew = 0, puHnew = 0;
    for (auto const& h : hits) {
      int gt = h.geantTrackId();
      if (gt <= 0)
        continue;
      uint64_t eid = h.eventId().rawId();
      uint64_t k = key(eid, uint32_t(gt));
      double en = h.energy();
      auto nit = keyToNode.find(k);
      bool found = nit != keyToNode.end();
      bool old = found && simNodeToParticle.count(nit->second);
      bool neu = found && resolve(nit->second) >= 0;
      if (eid == 0) {
        ++sigH;
        sigE += en;
        if (old) {
          ++sigHold;
          sigEold += en;
        }
        if (neu) {
          ++sigHnew;
          sigEnew += en;
        }
      } else {
        ++puH;
        puE += en;
        if (old) {
          ++puHold;
          puEold += en;
        }
        if (neu) {
          ++puHnew;
          puEnew += en;
        }
      }
    }
    // Diagnose the raw-graph SIM lineage for PU: how many PU SimTrack nodes have a
    // production vertex, a resolvable parent track, and reach a particle at all.
    unsigned long puNodes = 0, puWithVtx = 0, puWithParent = 0, puReachPart = 0, puGenLinked = 0;
    for (uint32_t n = 0; n < nN; ++n) {
      auto const& ref = rg.nodeRef(n);
      if (ref.kind != TruthGraph::NodeKind::SimTrack)
        continue;
      if (rg.nodeEventId(n) == 0ull)
        continue;  // PU only
      ++puNodes;
      int32_t v = rg.nodeSimTrackToVtx(n);
      if (v >= 0 && uint32_t(v) < nN) {
        ++puWithVtx;
        if (vertexParent[uint32_t(v)] >= 0)
          ++puWithParent;
      }
      if (resolve(n) >= 0)
        ++puReachPart;
      if (rg.nodeSimTrackToGen(n) >= 0)
        ++puGenLinked;
    }
    auto pct = [](double a, double b) { return b > 0 ? 100.0 * a / b : 0.0; };
    std::cout << "MVI evt: PU energy oldAtt=" << pct(puEold, puE) << "% newAtt=" << pct(puEnew, puE) << "%"
              << " | PU SimTracks=" << puNodes << " withProdVtx=" << pct(puWithVtx, puNodes)
              << "% withParent=" << pct(puWithParent, puNodes) << "% reachParticle=" << pct(puReachPart, puNodes)
              << "% genLinked=" << pct(puGenLinked, puNodes) << "%" << std::endl;
  }

private:
  edm::EDGetTokenT<std::vector<PCaloHit>> hitsTok_;
  edm::EDGetTokenT<truth::Graph> gTok_;
  edm::EDGetTokenT<TruthGraph> rgTok_;
};
DEFINE_FWK_MODULE(MergedVsIndexAnalyzer);
