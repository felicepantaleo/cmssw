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
//   HardProcess           the OUTGOING LEGS of the hard scatter
//   StableDecayProducts   each stable particle its decay produced
//   CaloBoundary          each particle that actually reaches the calorimeter
//
// HardProcess is NOT the resonance, and the name invites that mistake. isHardProcess is
// set on the hard-scatter participants, and the deepest-element antichain below keeps the
// outgoing ones, so on ttbar the level holds b, b~ and the W decay products rather than
// the two tops; on H to gamma gamma it holds the two photons, not the Higgs; on VBF it
// holds the tagging quarks and the four neutrinos. Measured on one event of each of the
// eleven generator templates.
// For the resonance itself use the SIGNAL selection, whose seeds are the resonance PDG
// ids for that generator fragment: it gives the two tops, the one Z, the one Higgs.
// Each is a different question about the same event, so each gets its own collection and
// its own efficiency, and none of them is more correct than the others.

#ifndef PhysicsTools_TruthInfo_interface_TruthLevels_h
#define PhysicsTools_TruthInfo_interface_TruthLevels_h

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/Vertex.h"

namespace truth {

  enum class Level {
    StableLegsFromUpstream,
    HardProcess,
    StableDecayProducts,
    CaloBoundary,
    ReconstructableFromSignal,
    UnderlyingEvent,
    PartonJets,
    BHadrons,
    CHadrons
  };

  [[nodiscard]] inline Level levelFromName(std::string const& name) {
    if (name == "stableLegsFromUpstream")
      return Level::StableLegsFromUpstream;
    if (name == "hardProcess")
      return Level::HardProcess;
    if (name == "stableDecayProducts")
      return Level::StableDecayProducts;
    if (name == "caloBoundary")
      return Level::CaloBoundary;
    if (name == "reconstructableFromSignal")
      return Level::ReconstructableFromSignal;
    if (name == "underlyingEvent")
      return Level::UnderlyingEvent;
    if (name == "partonJets")
      return Level::PartonJets;
    if (name == "bHadrons")
      return Level::BHadrons;
    if (name == "cHadrons")
      return Level::CHadrons;
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
      case Level::ReconstructableFromSignal:
        return "reconstructableFromSignal";
      case Level::UnderlyingEvent:
        return "underlyingEvent";
      case Level::PartonJets:
        return "partonJets";
      case Level::BHadrons:
        return "bHadrons";
      case Level::CHadrons:
        return "cHadrons";
    }
    return "unknown";
  }

  namespace detail {
    // reco::GenStatusFlags bit positions, as packed into ParticleData::statusFlags.
    constexpr uint16_t kIsHardProcess = 1u << 7;
    constexpr uint16_t kIsLastCopy = 1u << 13;
  }  // namespace detail

  // Quarks and gluons. Strings, clusters and diquarks are collapsed away by
  // truth::collapseGenShower before the graph is built, so they cannot appear here.
  [[nodiscard]] inline bool isParton(int32_t pdgId) {
    const int32_t a = std::abs(pdgId);
    return (a >= 1 && a <= 6) || a == 21;
  }

  // Ordinary hadron whose quark content includes `flavor` (5 = b, 4 = c), read off the
  // PDG hadron-numbering digits. Nuclei and generator-internal codes are not hadrons here.
  [[nodiscard]] inline bool hadronHasQuark(int32_t pdgId, int32_t flavor) {
    const int32_t id = std::abs(pdgId);
    if (id < 100 || id >= 1000000000)
      return false;
    const int32_t nq1 = (id / 1000) % 10;
    const int32_t nq2 = (id / 100) % 10;
    const int32_t nq3 = (id / 10) % 10;
    return nq1 == flavor || nq2 == flavor || nq3 == flavor;
  }

  // Whether one particle belongs to a level, before the antichain check.
  [[nodiscard]] inline bool atLevel(Graph const& graph, uint32_t id, Level level) {
    auto const& data = graph.particles()[id];
    switch (level) {
      case Level::StableLegsFromUpstream:
        // Not a per-particle predicate: it is reachability from the Upstream node, so
        // it is answered by stableLegsFromUpstream and never reaches here.
        return false;
      case Level::HardProcess:
        // The hard-scatter legs, not the resonance: see the header note.
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
      case Level::UnderlyingEvent:
        // Reachability from the artificial UnderlyingEvent vertex, answered by
        // stableLegsFromUnderlyingEvent.
        return false;
      case Level::ReconstructableFromSignal:
        // Not a per-particle predicate either: it is a walk down from the signal roots,
        // so it is answered by reconstructableFromSignal and never reaches here.
        return false;
      case Level::PartonJets:
        // Derived from the HardProcess antichain, so it needs that level's result rather
        // than a per-particle rule, and is answered by partonJets().
        return false;
      case Level::BHadrons:
        // The earliest-element antichain then keeps the B* and drops the B below it.
        return hadronHasQuark(data.pdgId, 5);
      case Level::CHadrons:
        // A c hadron from a B decay is a legitimate member: the nesting that matters is
        // within one flavour, and beauty and charm are deliberately different levels.
        return hadronHasQuark(data.pdgId, 4);
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
  // Stable legs hanging off every artificial vertex of one role. Upstream collects the
  // ISR and upstream side of the interaction, UnderlyingEvent the spectators; the walk is
  // identical, so it is written once. A leg is a particle that produced nothing further,
  // which makes the result an antichain by construction.
  [[nodiscard]] inline std::vector<uint32_t> stableLegsFromRole(Graph const& graph, VertexRole role) {
    std::vector<uint32_t> legs;
    std::vector<bool> seen(graph.nParticles(), false);

    const uint32_t nVertices = graph.nVertices();
    for (uint32_t v = 0; v < nVertices; ++v) {
      auto const& vertexData = graph.vertices()[v];
      if (vertexData.vertexRole() != role) {
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

  [[nodiscard]] inline std::vector<uint32_t> stableLegsFromUpstream(Graph const& graph) {
    return stableLegsFromRole(graph, VertexRole::Upstream);
  }

  [[nodiscard]] inline std::vector<uint32_t> stableLegsFromUnderlyingEvent(Graph const& graph) {
    return stableLegsFromRole(graph, VertexRole::UnderlyingEvent);
  }

  // Species a detector cannot reconstruct at all, so they are not part of the visible
  // final state. Only the neutrinos today; anything else invisible would belong here.
  [[nodiscard]] inline bool isInvisible(int32_t pdgId) {
    const int32_t a = std::abs(pdgId);
    return a == 12 || a == 14 || a == 16;
  }

  // The first stable, reconstructable particles the signal produced.
  //
  // Walk down from every Signal root and stop at the first generator-stable descendant,
  // which is where the decay chain ends and the detector's job begins. GEN-stable
  // terminates the walk on purpose: a stable pion still has a SIM continuation as it
  // showers, and descending into that would return shower fragments instead of the
  // particle the resonance actually produced.
  //
  // Neutrinos are dropped rather than walked through, so the result is the VISIBLE final
  // state of the resonance. A signal root that is itself stable, a gun electron say, is
  // its own leg.
  //
  // An antichain by construction: the walk stops at each leg, so no leg can be an
  // ancestor of another. Empty when nothing carries the Signal flag.
  [[nodiscard]] inline std::vector<uint32_t> reconstructableFromSignal(Graph const& graph) {
    const uint32_t nParticles = graph.nParticles();
    std::vector<uint32_t> legs;
    std::vector<bool> seen(nParticles, false);
    std::vector<uint32_t> stack;

    for (uint32_t p = 0; p < nParticles; ++p) {
      if (graph.particles()[p].isAtLevel(LevelFlag::Signal)) {
        seen[p] = true;
        stack.push_back(p);
      }
    }

    while (!stack.empty()) {
      const uint32_t p = stack.back();
      stack.pop_back();
      auto const& data = graph.particles()[p];

      // Terminal three ways: the detector reconstructs this species as an object even
      // though it decays (pi0), the generator called it stable, or the graph has nothing
      // below it. Anything else is an intermediate the detector never sees as an object,
      // an a1 or a rho, and the walk goes through it without labelling it.
      // The seen mask makes this terminate on a graph with a cycle.
      auto const& terminating = graph.reconstructablePdgIds();
      const bool reconstructableSpecies =
          std::find(terminating.begin(), terminating.end(), data.pdgId) != terminating.end();
      const bool genStable = data.hasGen() && data.status == 1;
      if (reconstructableSpecies || genStable || graph.decayVertices(p).empty()) {
        if (!isInvisible(data.pdgId)) {
          legs.push_back(p);
        }
        continue;
      }

      for (const uint32_t vertexId : graph.decayVertices(p)) {
        if (vertexId >= graph.nVertices()) {
          continue;
        }
        for (const uint32_t child : graph.outgoingParticles(vertexId)) {
          if (child < nParticles && !seen[child]) {
            seen[child] = true;
            stack.push_back(child);
          }
        }
      }
    }

    std::sort(legs.begin(), legs.end());
    legs.erase(std::unique(legs.begin(), legs.end()), legs.end());
    return legs;
  }

  // PartonJets is defined in terms of the HardProcess antichain and levelAntichain
  // dispatches back to it, so one of the two has to be declared ahead of the other.
  [[nodiscard]] inline std::vector<uint32_t> levelAntichain(Graph const& graph, Level level);

  // One root per parton-initiated jet. The jet is the descendant subgraph of the parton;
  // there is no clustering and no cone, and the flavour is the parton's own PDG id.
  //
  // The members are the hard-scatter legs that are partons. That is the COMPLETE set of
  // quarks and gluons the graph holds: collapseGenShower removes every shower parton, so
  // a parton survives the build only by carrying isHardProcess, and "the early quark" and
  // "the quark that is present" are the same particle here.
  //
  // Inherits the deepest-element rule from HardProcess, which is what keeps a top out in
  // favour of its b, and what keeps the INCOMING beam partons out: they sit at pt 0 and
  // beam rapidity, and the outgoing legs are their descendants. A hard gluon radiated at
  // the production vertex has no hard-process descendant, so it stays, and it is a real
  // gluon jet rather than an artefact.
  //
  // EMPTY, not wrong, when statusFlags are unavailable: the HepMC3 path never fills them,
  // and pile-up sub-events contribute only stable GEN particles, so a jet here is always
  // a signal jet.
  //
  // The ROOTS are an antichain but the SUBGRAPHS are not disjoint, and that is inherent to
  // defining a jet without a clustering algorithm. Two quarks colour-connected to each
  // other, the u and dbar of a hadronic W, fragment through one string, so its hadrons
  // descend from both and their hits are counted under both jets. Measured on no-PU ttbar:
  // 1221 of 8096 hits shared, 0.15 of the union, ALL of it between that one pair, while
  // the b and bbar share nothing; a dileptonic event with no hadronic W shares 0.00.
  // Assigning each hadron to exactly one jet is what a clustering algorithm is for.
  [[nodiscard]] inline std::vector<uint32_t> partonJets(Graph const& graph) {
    std::vector<uint32_t> roots = levelAntichain(graph, Level::HardProcess);
    roots.erase(
        std::remove_if(
            roots.begin(), roots.end(), [&graph](uint32_t id) { return !isParton(graph.particles()[id].pdgId); }),
        roots.end());
    return roots;
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
    if (level == Level::ReconstructableFromSignal) {
      return reconstructableFromSignal(graph);
    }
    if (level == Level::UnderlyingEvent) {
      return stableLegsFromUnderlyingEvent(graph);
    }
    if (level == Level::PartonJets) {
      return partonJets(graph);
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
      case Level::ReconstructableFromSignal:
        return LevelFlag::ReconstructableFromSignal;
      case Level::UnderlyingEvent:
        return LevelFlag::UnderlyingEvent;
      case Level::PartonJets:
        return LevelFlag::PartonJets;
      case Level::BHadrons:
        return LevelFlag::BHadrons;
      case Level::CHadrons:
        return LevelFlag::CHadrons;
    }
    return LevelFlag::CaloBoundary;
  }

  inline constexpr std::array<Level, 9> kAllLevels = {Level::StableLegsFromUpstream,
                                                      Level::HardProcess,
                                                      Level::StableDecayProducts,
                                                      Level::CaloBoundary,
                                                      Level::ReconstructableFromSignal,
                                                      Level::UnderlyingEvent,
                                                      Level::PartonJets,
                                                      Level::BHadrons,
                                                      Level::CHadrons};

  // Stamp every particle with the levels it belongs to. Call once, on the COMPLETE graph:
  // levelAntichain walks ancestors and descendants, so a graph still being assembled
  // gives an antichain of whatever existed at the time.
  //
  // Clears first, so calling it twice is the same as calling it once. That matters
  // because a stale flag is indistinguishable from a fresh one by inspection, and the
  // only defence is that the operation is reproducible and idempotent.
  inline void fillLevelFlags(Graph& graph) {
    // Clear only the bits this function owns. LevelFlag::Signal is set upstream, by the
    // selection post-processing that knows the seed species, and clearing it here would
    // silently erase the resonance.
    constexpr uint32_t kOwned =
        static_cast<uint32_t>(LevelFlag::StableLegsFromUpstream) | static_cast<uint32_t>(LevelFlag::HardProcess) |
        static_cast<uint32_t>(LevelFlag::StableDecayProducts) | static_cast<uint32_t>(LevelFlag::CaloBoundary) |
        static_cast<uint32_t>(LevelFlag::PartonJets) | static_cast<uint32_t>(LevelFlag::BHadrons) |
        static_cast<uint32_t>(LevelFlag::CHadrons);
    for (auto& particle : graph.particles()) {
      particle.levelFlags &= ~kOwned;
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
