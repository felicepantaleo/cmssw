// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Levels of the truth graph: the a-priori definition of WHAT a truth object is, for the
// truth-driven direction of the association.
//
// A level must be an ANTICHAIN: no particle in it may be an ancestor of another. That is
// not a stylistic requirement, it is what makes the efficiency denominator answerable.
// Selecting truth objects by kinematics alone does not give an antichain: a tau, its
// decay products and their calorimeter-crossing descendants all pass a pt cut, so all
// three end up in the denominator, each expected to be reconstructed as its own object
// out of the same hits. Measured on ttbar, 24% of a pt-selected set had another selected
// particle as an ancestor, and the efficiency that came out of it was meaningless.
//
// The levels below are the three a tau makes obvious, and they generalise:
//   HardProcess           the tau itself, the object the analysis names
//   StableDecayProducts   each stable particle its decay produced
//   CaloBoundary          each particle that actually reaches the calorimeter
// Each is a different question about the same event, so each gets its own collection and
// its own efficiency, and none of them is more correct than the others.

#ifndef PhysicsTools_TruthInfo_interface_TruthLevels_h
#define PhysicsTools_TruthInfo_interface_TruthLevels_h

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/Vertex.h"

namespace truth {

  enum class Level { StableLegsFromUpstream, HardProcess, StableDecayProducts, CaloBoundary };

  [[nodiscard]] inline Level levelFromName(std::string const& name) {
    if (name == "stableLegsFromUpstream")
      return Level::StableLegsFromUpstream;
    if (name == "hardProcess")
      return Level::HardProcess;
    if (name == "stableDecayProducts")
      return Level::StableDecayProducts;
    if (name == "caloBoundary")
      return Level::CaloBoundary;
    throw std::runtime_error("unknown truth level '" + name +
                             "', expected hardProcess, stableDecayProducts or caloBoundary");
  }

  // Inverse of levelFromName, so a log line and a configuration string use one spelling.
  [[nodiscard]] inline std::string levelName(Level level) {
    switch (level) {
      case Level::StableLegsFromUpstream:
        return "stableLegsFromUpstream";
      case Level::HardProcess:
        return "hardProcess";
      case Level::StableDecayProducts:
        return "stableDecayProducts";
      case Level::CaloBoundary:
        return "caloBoundary";
    }
    return "unknown";
  }

  namespace detail {
    // reco::GenStatusFlags bit positions, as packed into ParticleData::statusFlags.
    constexpr uint16_t kIsHardProcess = 1u << 7;
    constexpr uint16_t kIsLastCopy = 1u << 13;
  }  // namespace detail

  // Whether one particle belongs to a level, before the antichain check.
  [[nodiscard]] inline bool atLevel(Graph const& graph, uint32_t id, Level level) {
    auto const& data = graph.particles()[id];
    switch (level) {
      case Level::StableLegsFromUpstream:
        // Not a per-particle predicate: it is reachability from the Upstream node, so
        // it is answered by stableLegsFromUpstream and never reaches here.
        return false;
      case Level::HardProcess:
        // isHardProcess alone. Requiring isLastCopy as well made this level EMPTY for
        // every sample: the two flags are never set on the same copy, measured as 0.00
        // per event on the generator record of ttbar, DYToLL and VBFHZZ4Nu alike. The
        // repeated copies are removed by taking the LAST hard-process copy in the
        // antichain below, which is what the isLastCopy requirement was reaching for.
        return (data.statusFlags & detail::kIsHardProcess) != 0;
      case Level::StableDecayProducts:
        // Final-state generator particles. Stable at GEN means no GEN descendant, so
        // these cannot contain one another.
        return data.hasGen() && data.status == 1;
      case Level::CaloBoundary:
        // Recorded crossing the tracker-calorimeter boundary outward. Back-scattered
        // tracks crossed it inward and are the same particle coming back.
        return !data.backscattered && Particle(&graph, id).checkpoint(0).has_value();
    }
    return false;
  }

  // The default level, and the one that needs no per-process rule: every stable leg
  // hanging off the Upstream node.
  //
  // The graph marks the interesting activity of any event with an artificial Upstream
  // vertex (VertexRole::Upstream, reached from the per-interaction Interaction node).
  // Following each of its outgoing particles down to where the chain stops gives one
  // branch per stable leg, whatever the process: for a tau event the legs are the tau's
  // decay products, for a dijet event they are the hadrons, and nothing in the rule
  // mentions taus or jets. It is an antichain by construction because a leg is a
  // particle that produced nothing further.
  [[nodiscard]] inline std::vector<uint32_t> stableLegsFromUpstream(Graph const& graph) {
    std::vector<uint32_t> legs;
    std::vector<bool> seen(graph.nParticles(), false);

    const uint32_t nVertices = graph.nVertices();
    for (uint32_t v = 0; v < nVertices; ++v) {
      auto const& vertexData = graph.vertices()[v];
      if (vertexData.vertexRole() != VertexRole::Upstream) {
        continue;
      }
      // Depth-first from each outgoing particle; a particle with no children is a leg.
      std::vector<uint32_t> stack;
      for (auto const& outgoing : Vertex(&graph, v).outgoingParticles()) {
        stack.push_back(outgoing.id());
      }
      while (!stack.empty()) {
        const uint32_t id = stack.back();
        stack.pop_back();
        if (id >= seen.size() || seen[id]) {
          continue;
        }
        seen[id] = true;
        const auto children = Particle(&graph, id).children();
        if (children.empty()) {
          legs.push_back(id);
          continue;
        }
        for (auto const& child : children) {
          stack.push_back(child.id());
        }
      }
    }
    std::sort(legs.begin(), legs.end());
    return legs;
  }

  // The level as an antichain. Candidates that have another candidate as an ancestor are
  // dropped, so what remains is one entry per physical object at that level. The
  // membership rules above are already antichains in a well-formed graph; the check is
  // kept because a denominator that silently contains a particle and its own parent is
  // the failure this class exists to prevent.
  [[nodiscard]] inline std::vector<uint32_t> levelAntichain(Graph const& graph, Level level) {
    if (level == Level::StableLegsFromUpstream) {
      return stableLegsFromUpstream(graph);
    }
    std::vector<uint32_t> candidates;
    const uint32_t nParticles = graph.nParticles();
    for (uint32_t id = 0; id < nParticles; ++id) {
      if (atLevel(graph, id, level)) {
        candidates.push_back(id);
      }
    }
    const std::vector<bool> isCandidate = [&] {
      std::vector<bool> flags(nParticles, false);
      for (uint32_t id : candidates) {
        flags[id] = true;
      }
      return flags;
    }();

    // Which end of a chain of candidates to keep. For every level but HardProcess the
    // members are final states and a candidate with a candidate ANCESTOR is a duplicate
    // of it, so the earliest is kept. HardProcess is the opposite: the incoming partons
    // and the outgoing particles both carry the flag, the incoming ones are ancestors of
    // the outgoing ones, and it is the outgoing ones that the level is about. Keeping
    // the earliest there would return the beam partons, which sit at pt 0 and enormous
    // eta and are then dropped by any kinematic selector, leaving the level empty.
    const bool keepDeepest = (level == Level::HardProcess);

    std::vector<uint32_t> antichain;
    antichain.reserve(candidates.size());
    for (uint32_t id : candidates) {
      bool covered = false;
      if (keepDeepest) {
        for (auto const& descendant : Particle(&graph, id).descendants()) {
          if (descendant.id() < nParticles && isCandidate[descendant.id()]) {
            covered = true;
            break;
          }
        }
      } else {
        for (auto const& ancestor : Particle(&graph, id).ancestors()) {
          if (ancestor.id() < nParticles && isCandidate[ancestor.id()]) {
            covered = true;
            break;
          }
        }
      }
      if (!covered) {
        antichain.push_back(id);
      }
    }
    return antichain;
  }

  // The persisted bit for a level. Kept next to the Level enum so adding a level forces
  // the author past this switch, which has no default for that reason.
  [[nodiscard]] inline LevelFlag levelFlagOf(Level level) {
    switch (level) {
      case Level::StableLegsFromUpstream:
        return LevelFlag::StableLegsFromUpstream;
      case Level::HardProcess:
        return LevelFlag::HardProcess;
      case Level::StableDecayProducts:
        return LevelFlag::StableDecayProducts;
      case Level::CaloBoundary:
        return LevelFlag::CaloBoundary;
    }
    return LevelFlag::CaloBoundary;
  }

  inline constexpr std::array<Level, 4> kAllLevels = {
      Level::StableLegsFromUpstream, Level::HardProcess, Level::StableDecayProducts, Level::CaloBoundary};

  // Stamp every particle with the levels it belongs to. Call once, on the COMPLETE graph:
  // levelAntichain walks ancestors and descendants, so a graph still being assembled
  // gives an antichain of whatever existed at the time.
  //
  // Clears first, so calling it twice is the same as calling it once. That matters
  // because a stale flag is indistinguishable from a fresh one by inspection, and the
  // only defence is that the operation is reproducible and idempotent.
  inline void fillLevelFlags(Graph& graph) {
    for (auto& particle : graph.particles()) {
      particle.levelFlags = 0;
    }
    for (const Level level : kAllLevels) {
      const LevelFlag flag = levelFlagOf(level);
      for (const uint32_t id : levelAntichain(graph, level)) {
        if (id < graph.nParticles()) {
          graph.particles()[id].setLevel(flag);
        }
      }
    }
  }

}  // namespace truth

#endif
