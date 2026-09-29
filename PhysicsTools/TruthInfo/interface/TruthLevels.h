// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Levels of the truth graph: the a-priori definition of a truth object, for the
// truth-driven direction of the association.
//
// A level must be an antichain: no member is an ancestor of another. A nested pair asks
// for a tau and its decay products as separate objects made of the same hits. A
// kinematic cut alone does not give an antichain.
//
// HardProcess is the outgoing legs of the hard scatter, not the resonance: on ttbar it
// keeps b, b~ and the W decay products, not the tops. The resonance is the signal
// selection, seeded on its PDG ids.

#ifndef PhysicsTools_TruthInfo_interface_TruthLevels_h
#define PhysicsTools_TruthInfo_interface_TruthLevels_h

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"
#include "SimDataFormats/TruthInfo/interface/Vertex.h"

namespace truth {

  enum class Level {
    StableLegsFromInitialState,
    HardProcess,
    StableDecayProducts,
    CaloBoundary,
    ReconstructableFromSignal,
    UnderlyingEvent,
    PartonJets,
    BHadrons,
    CHadrons,
    ReconstructableFinalState,
    TauVisibleHadronic,
    TauVisibleLeptonic
  };

  // One row per level: the enum value, the bit it stamps on the graph, and its
  // configuration name. The name and flag lookups, kAllLevels and kOwnedLevelFlags derive
  // from this table. LevelFlag::Signal is not a row: the selection post-processing owns it.
  struct LevelRow {
    Level level;
    LevelFlag flag;
    char const* name;
  };

  inline constexpr std::array<LevelRow, 12> kLevelTable = {
      {{Level::StableLegsFromInitialState, LevelFlag::StableLegsFromInitialState, "stableLegsFromInitialState"},
       {Level::HardProcess, LevelFlag::HardProcess, "hardProcess"},
       {Level::StableDecayProducts, LevelFlag::StableDecayProducts, "stableDecayProducts"},
       {Level::CaloBoundary, LevelFlag::CaloBoundary, "caloBoundary"},
       {Level::ReconstructableFromSignal, LevelFlag::ReconstructableFromSignal, "reconstructableFromSignal"},
       {Level::UnderlyingEvent, LevelFlag::UnderlyingEvent, "underlyingEvent"},
       {Level::PartonJets, LevelFlag::PartonJets, "partonJets"},
       {Level::BHadrons, LevelFlag::BHadrons, "bHadrons"},
       {Level::CHadrons, LevelFlag::CHadrons, "cHadrons"},
       {Level::ReconstructableFinalState, LevelFlag::ReconstructableFinalState, "reconstructableFinalState"},
       {Level::TauVisibleHadronic, LevelFlag::TauVisibleHadronic, "tauVisibleHadronic"},
       {Level::TauVisibleLeptonic, LevelFlag::TauVisibleLeptonic, "tauVisibleLeptonic"}}};

  inline constexpr std::array<Level, kLevelTable.size()> kAllLevels = [] {
    std::array<Level, kLevelTable.size()> levels{};
    for (std::size_t i = 0; i < kLevelTable.size(); ++i) {
      levels[i] = kLevelTable[i].level;
    }
    return levels;
  }();

  // Every bit that the level machinery stamps. fillLevelFlags clears exactly these bits.
  inline constexpr uint32_t kOwnedLevelFlags = [] {
    uint32_t mask = 0;
    for (auto const& row : kLevelTable) {
      mask |= static_cast<uint32_t>(row.flag);
    }
    return mask;
  }();

  [[nodiscard]] inline Level levelFromName(std::string const& name) {
    for (auto const& row : kLevelTable) {
      if (name == row.name) {
        return row.level;
      }
    }
    cms::Exception ex("TruthLevels");
    ex << "unknown truth level '" << name << "', expected one of:";
    for (auto const& row : kLevelTable) {
      ex << " " << row.name;
    }
    throw ex;
  }

  // Inverse of levelFromName, so a log line and a configuration string use one spelling.
  [[nodiscard]] inline const char* levelName(Level level) {
    for (auto const& row : kLevelTable) {
      if (row.level == level) {
        return row.name;
      }
    }
    return "unknown";
  }

  // The name of LevelFlag::Signal, which has no row in kLevelTable.
  inline constexpr char const* kSignalLevelName = "signal";

  // The names of the levels a particle belongs to, in kLevelTable order, signal last.
  [[nodiscard]] inline std::vector<char const*> levelNamesOf(ParticleData const& data) {
    std::vector<char const*> names;
    for (auto const& row : kLevelTable) {
      if (data.isAtLevel(row.flag)) {
        names.push_back(row.name);
      }
    }
    if (data.isAtLevel(LevelFlag::Signal)) {
      names.push_back(kSignalLevelName);
    }
    return names;
  }

  namespace detail {
    // reco::GenStatusFlags bit positions, as packed into ParticleData::statusFlags.
    constexpr uint16_t kIsHardProcess = 1u << 7;
    constexpr uint16_t kIsLastCopy = 1u << 13;
  }  // namespace detail

  // Quarks and gluons.
  [[nodiscard]] inline bool isParton(int32_t pdgId) {
    const int64_t a = std::abs(static_cast<int64_t>(pdgId));
    return (a >= 1 && a <= 6) || a == 21;
  }

  // Charged leptons. A neutrino is not one of these; ask isInvisible for that.
  [[nodiscard]] inline bool isLepton(int32_t pdgId) {
    const int64_t a = std::abs(static_cast<int64_t>(pdgId));
    return a == 11 || a == 13 || a == 15;
  }

  // The W and the Z. The Higgs is not a weak boson and is not one of these.
  [[nodiscard]] inline bool isWeakBoson(int32_t pdgId) {
    const int64_t a = std::abs(static_cast<int64_t>(pdgId));
    return a == 23 || a == 24;
  }

  // Shower bookkeeping rather than a particle a detector could be asked about: a parton,
  // a diquark, a Pythia string or cluster, a beam or generator-internal pseudoparticle.
  // The main event keeps its shower, so these are in the graph.
  [[nodiscard]] inline bool isShowerObject(int32_t pdgId) {
    const int64_t a = std::abs(static_cast<int64_t>(pdgId));
    if (isParton(pdgId)) {
      return true;
    }
    if (a >= 91 && a <= 94) {  // cluster, string and the other hadronization placeholders
      return true;
    }
    if (a == 990) {  // pomeron
      return true;
    }
    if (a >= 1000 && a <= 9999 && (a / 10) % 10 == 0 && (a / 100) % 10 != 0) {  // diquarks, e.g. 2101, 2203
      return true;
    }
    return a >= 9900000 && a < 1000000000;  // generator-internal states, below the nuclei codes
  }

  // The id form of Particle::lastCopy.
  [[nodiscard]] inline uint32_t lastCopyOf(truth::Graph const& graph, uint32_t rootId) {
    return Particle(&graph, rootId).lastCopy().id();
  }

  // A shower object that turns into hadrons rather than decaying. The top is excluded
  // although its pdgId makes it a parton: it decays before it can hadronize.
  [[nodiscard]] inline bool hadronizes(int32_t pdgId) { return isShowerObject(pdgId) && std::abs(pdgId) != 6; }

  // The physical reason a GEN-only vertex exists, from the species and the generator
  // status codes of the particles that meet there. Returns Unknown for a vertex with no
  // GEN side, for an artificial vertex, and for anything the rules do not cover. The
  // caller decides whether a vertex with a SIM side keeps its Geant4 reason instead. The
  // graph must have its adjacency built.
  [[nodiscard]] inline VertexReason genVertexReason(Graph const& graph, uint32_t vertexId) {
    const std::size_t next = static_cast<std::size_t>(vertexId) + 1;
    if (next >= graph.vertexToIncomingParticleOffsets().size() ||
        next >= graph.vertexToOutgoingParticleOffsets().size())
      return VertexReason::Unknown;

    auto const& vertex = graph.vertices()[vertexId];
    if (vertex.isArtificial() || !vertex.hasGen())
      return VertexReason::Unknown;

    const auto incoming = graph.incomingParticles(vertexId);
    const auto outgoing = graph.outgoingParticles(vertexId);
    if (incoming.empty() || outgoing.empty())
      return VertexReason::Unknown;

    auto pdgIdOf = [&graph](uint32_t id) { return graph.particles()[id].pdgId; };
    auto anyOutgoingStatus = [&](int16_t low, int16_t high) {
      return std::any_of(outgoing.begin(), outgoing.end(), [&](uint32_t id) {
        const int16_t status = graph.particles()[id].status;
        return status >= low && status <= high;
      });
    };
    // Every incoming leg hadronizes, so the vertex belongs to the shower and not to the
    // decay of a particle a detector could be asked about.
    const bool fromShower =
        std::all_of(incoming.begin(), incoming.end(), [&](uint32_t id) { return hadronizes(pdgIdOf(id)); });

    // Two or more incoming hard-process legs. The count is required: a hard-process
    // resonance enters its own decay vertex and would otherwise match here. The flag is
    // the generator-independent form, but buildFromHepMC3 leaves it empty, so the Pythia
    // incoming-hard-parton code is the fallback.
    if (incoming.size() >= 2) {
      auto allIncoming = [&](auto predicate) { return std::all_of(incoming.begin(), incoming.end(), predicate); };
      const bool flagged =
          allIncoming([&](uint32_t id) { return (graph.particles()[id].statusFlags & detail::kIsHardProcess) != 0; });
      const bool coded = allIncoming([&](uint32_t id) { return graph.particles()[id].status == 21; });
      if (flagged || coded)
        return VertexReason::HardScatter;
    }

    // A branching inside the shower, initial or final state. The species test carries the
    // rule: a status code travels with a particle's own history, so a decay whose product
    // is a shower copy would match the code alone.
    if (fromShower && anyOutgoingStatus(41, 59))
      return VertexReason::ShowerBranching;

    const bool fromString = std::any_of(incoming.begin(), incoming.end(), [&](uint32_t id) {
      const int64_t pdgId = std::abs(static_cast<int64_t>(pdgIdOf(id)));
      return pdgId >= 91 && pdgId <= 94;
    });
    const bool toHadron =
        std::any_of(outgoing.begin(), outgoing.end(), [&](uint32_t id) { return !isShowerObject(pdgIdOf(id)); });
    if (fromString || (fromShower && (toHadron || anyOutgoingStatus(71, 79))))
      return VertexReason::Hadronization;

    // A particle that comes out of its own vertex radiated, it did not decay: QED
    // radiation off a lepton is a shower branching with the lepton on both sides.
    if (incoming.size() == 1 && anyOutgoingStatus(41, 59) &&
        std::any_of(outgoing.begin(), outgoing.end(), [&](uint32_t id) { return pdgIdOf(id) == pdgIdOf(incoming[0]); }))
      return VertexReason::ShowerBranching;

    if (incoming.size() == 1 && !hadronizes(pdgIdOf(incoming[0])))
      return VertexReason::Decay;

    return VertexReason::Unknown;
  }

  // Ordinary hadron whose quark content includes `flavor` (5 = b, 4 = c), read off the
  // PDG hadron-numbering digits. Nuclei and generator-internal codes are not hadrons here.
  [[nodiscard]] inline bool hadronHasQuark(int32_t pdgId, int32_t flavor) {
    const int64_t id = std::abs(static_cast<int64_t>(pdgId));
    if (id < 100 || id >= 1000000000)
      return false;
    // A diquark is nq1 nq2 0 nJ, so its third quark digit is zero. It carries the
    // flavour digits of the hadron it fragments into and would otherwise be flagged
    // AND, being that hadron's ancestor, cover it in the earliest-element antichain.
    if (id >= 1000 && id <= 9999 && (id / 10) % 10 == 0 && (id / 100) % 10 != 0)
      return false;
    const int64_t nq1 = (id / 1000) % 10;
    const int64_t nq2 = (id / 100) % 10;
    const int64_t nq3 = (id / 10) % 10;
    return nq1 == flavor || nq2 == flavor || nq3 == flavor;
  }

  // Whether a seed pdgId list names a resonance. An empty list (no preset) and {0} (the
  // full-graph selection) both mean "no selection": the signal level is not offered.
  // Templated for std::vector<int> (module parameters) and std::vector<int32_t> (Graph).
  template <typename Seeds>
  [[nodiscard]] inline bool seedsNameAResonance(Seeds const& seeds) {
    return !seeds.empty() && std::find(seeds.begin(), seeds.end(), 0) == seeds.end();
  }

  // A selection also names a signal when it seeds on heavy-flavour hadron content: the
  // heavyflavor preset has an empty pdgId list and a flavour list.
  template <typename Seeds, typename Flavors>
  [[nodiscard]] inline bool seedsNameAResonance(Seeds const& seeds, Flavors const& flavors) {
    return seedsNameAResonance(seeds) || !flavors.empty();
  }

  // How the generator recorded the decay of one tau.
  enum class TauDecay : uint8_t { None, Hadronic, Leptonic };

  // The decay mode of one physical tau. None when the particle is not a tau, is
  // synthetic, has no GEN decay record, or has a tau child (a radiative copy). Leptonic
  // when an electron or a muon is among the children, else Hadronic, as in
  // TauGenJetProducer. Only the last tau of a radiative chain has a mode, so each tau
  // level is an antichain.
  [[nodiscard]] inline TauDecay tauDecay(Graph const& graph, uint32_t id) {
    auto const& data = graph.particles()[id];
    if (std::abs(static_cast<int64_t>(data.pdgId)) != 15 || data.isSynthetic()) {
      return TauDecay::None;
    }
    bool hasGenDecay = false;
    bool leptonic = false;
    for (const uint32_t vertexId : graph.decayVertices(id)) {
      if (vertexId >= graph.nVertices() || !graph.vertices()[vertexId].hasGen()) {
        continue;
      }
      hasGenDecay = true;
      for (const uint32_t child : graph.outgoingParticles(vertexId)) {
        if (child >= graph.nParticles() || child == id) {
          continue;
        }
        const int64_t a = std::abs(static_cast<int64_t>(graph.particles()[child].pdgId));
        if (a == 15) {
          return TauDecay::None;
        }
        if (a == 11 || a == 13) {
          leptonic = true;
        }
      }
    }
    if (!hasGenDecay) {
      return TauDecay::None;
    }
    return leptonic ? TauDecay::Leptonic : TauDecay::Hadronic;
  }

  // One entry per physical tau that decays to hadrons. The member is the tau itself, so
  // its visible part is the branch of its decay products with the neutrino dropped. This
  // is what tau identification measures efficiency against.
  [[nodiscard]] inline bool isTauVisibleHadronic(Graph const& graph, uint32_t id) {
    return tauDecay(graph, id) == TauDecay::Hadronic;
  }

  // One entry per physical tau that decays to an electron or a muon. The member is the
  // tau itself, as in the hadronic level, so its visible part is the charged lepton.
  [[nodiscard]] inline bool isTauVisibleLeptonic(Graph const& graph, uint32_t id) {
    return tauDecay(graph, id) == TauDecay::Leptonic;
  }

  // Whether one particle belongs to a level, before the antichain check.
  [[nodiscard]] inline bool atLevel(Graph const& graph, uint32_t id, Level level) {
    auto const& data = graph.particles()[id];
    switch (level) {
      case Level::StableLegsFromInitialState:
        // Not a per-particle predicate: it is reachability from the InitialState node, so
        // it is answered by stableLegsFromInitialState and never reaches here.
        return false;
      case Level::HardProcess:
        // The hard-scatter legs, not the resonance: see the header note.
        // isHardProcess alone. The copy collapse ORs the flags of a chain onto one
        // particle, and the deepest-element antichain below removes repeated copies.
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
        // The deepest-element antichain then keeps the weakly decaying hadron and drops
        // the B* above it.
        return hadronHasQuark(data.pdgId, 5);
      case Level::ReconstructableFinalState:
        // A walk from the GEN roots, answered by reconstructableFinalState, so it never
        // reaches here.
        return false;
      case Level::TauVisibleHadronic:
        return isTauVisibleHadronic(graph, id);
      case Level::TauVisibleLeptonic:
        return isTauVisibleLeptonic(graph, id);
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

  // The stable GEN descendants of every artificial vertex of one role. InitialState
  // gives the stable descendants of the selected roots. UnderlyingEvent gives the other
  // stable particles, initial-state radiation included. A leg has no GEN children.
  [[nodiscard]] inline std::vector<uint32_t> stableLegsFromRole(Graph const& graph, VertexRole role) {
    std::vector<uint32_t> legs;
    std::vector<bool> seen(graph.nParticles(), false);
    std::vector<uint32_t> stack;

    const uint32_t nVertices = graph.nVertices();
    for (uint32_t v = 0; v < nVertices; ++v) {
      auto const& vertexData = graph.vertices()[v];
      if (vertexData.vertexRole() != role) {
        continue;
      }
      // Depth-first from each outgoing particle. Only GEN decay vertices are descended:
      // a SIM continuation is transport, so a stable ISR photon that converts in the
      // tracker stays the leg.
      for (const uint32_t outgoing : graph.outgoingParticles(v)) {
        stack.push_back(outgoing);
      }
      while (!stack.empty()) {
        const uint32_t id = stack.back();
        stack.pop_back();
        if (id >= seen.size() || seen[id]) {
          continue;
        }
        seen[id] = true;
        bool isLeg = true;
        for (const uint32_t vertexId : graph.decayVertices(id)) {
          if (vertexId >= nVertices || !graph.vertices()[vertexId].hasGen()) {
            continue;
          }
          for (const uint32_t child : graph.outgoingParticles(vertexId)) {
            if (child == id) {
              continue;
            }
            isLeg = false;
            if (child < seen.size() && !seen[child]) {
              stack.push_back(child);
            }
          }
        }
        if (isLeg) {
          legs.push_back(id);
        }
      }
    }
    std::sort(legs.begin(), legs.end());
    return legs;
  }

  [[nodiscard]] inline std::vector<uint32_t> stableLegsFromInitialState(Graph const& graph) {
    return stableLegsFromRole(graph, VertexRole::InitialState);
  }

  [[nodiscard]] inline std::vector<uint32_t> stableLegsFromUnderlyingEvent(Graph const& graph) {
    return stableLegsFromRole(graph, VertexRole::UnderlyingEvent);
  }

  // Species that a detector cannot reconstruct: the neutrinos.
  [[nodiscard]] inline bool isInvisible(int32_t pdgId) {
    const int64_t a = std::abs(static_cast<int64_t>(pdgId));
    return a == 12 || a == 14 || a == 16;
  }

  namespace detail {
    // The reconstructable-final-state walk from the seeds that isSeed selects.
    template <typename SeedPredicate>
    [[nodiscard]] inline std::vector<uint32_t> reconstructableLegsFrom(Graph const& graph, SeedPredicate isSeed) {
      const uint32_t nParticles = graph.nParticles();
      std::vector<uint32_t> legs;
      std::vector<bool> seen(nParticles, false);
      std::vector<uint32_t> stack;
      auto const& terminating = graph.reconstructablePdgIds();

      for (uint32_t p = 0; p < nParticles; ++p) {
        if (isSeed(p)) {
          seen[p] = true;
          stack.push_back(p);
        }
      }

      while (!stack.empty()) {
        const uint32_t p = stack.back();
        stack.pop_back();
        auto const& data = graph.particles()[p];

        // A particle is terminal when its species is reconstructable (pi0), it is GEN
        // stable, or it has no GEN decay. The walk goes through other particles (a1, rho).
        // It descends through GEN decay vertices only: a K0S that the generator decays
        // gives its GEN pions, not the nuclear secondaries of its SIM vertex.
        // The seen mask makes the walk terminate on a graph with a cycle.
        const bool reconstructableSpecies =
            std::find(terminating.begin(), terminating.end(), data.pdgId) != terminating.end();
        const bool genStable = data.hasGen() && data.status == 1;
        bool hasGenDecay = false;
        for (const uint32_t vertexId : graph.decayVertices(p)) {
          if (vertexId < graph.nVertices() && graph.vertices()[vertexId].hasGen()) {
            hasGenDecay = true;
            break;
          }
        }
        if (reconstructableSpecies || genStable || !hasGenDecay) {
          // A synthetic particle has no hits and is not a leg. The signal stand-in is
          // synthetic and has no vertex.
          if (!isInvisible(data.pdgId) && !data.isSynthetic()) {
            legs.push_back(p);
          }
          continue;
        }

        for (const uint32_t vertexId : graph.decayVertices(p)) {
          if (vertexId >= graph.nVertices() || !graph.vertices()[vertexId].hasGen()) {
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
      return legs;
    }
  }  // namespace detail

  // The first reconstructable particles that the signal produces: the walk from every
  // Signal root stops at the first terminal descendant, not at shower fragments.
  // Neutrinos are dropped, so the result is the visible final state. A stable signal
  // root (a gun electron) is its own leg. Empty when no particle has the Signal flag.
  [[nodiscard]] inline std::vector<uint32_t> reconstructableFromSignal(Graph const& graph) {
    return detail::reconstructableLegsFrom(
        graph, [&graph](uint32_t p) { return graph.particles()[p].isAtLevel(LevelFlag::Signal); });
  }

  // The same walk seeded from every GEN root (a GEN particle with no GEN parent). The
  // level exists on every sample, also where no resonance is selected.
  [[nodiscard]] inline std::vector<uint32_t> reconstructableFinalState(Graph const& graph) {
    auto const isGenRoot = [&graph](uint32_t p) {
      if (!graph.particles()[p].hasGen()) {
        return false;
      }
      for (const uint32_t vertexId : graph.productionVertices(p)) {
        if (vertexId >= graph.nVertices()) {
          continue;
        }
        for (const uint32_t parent : graph.incomingParticles(vertexId)) {
          if (parent != p && graph.particles()[parent].hasGen()) {
            return false;
          }
        }
      }
      return true;
    };
    return detail::reconstructableLegsFrom(graph, isGenRoot);
  }

  // Forward declaration: partonJets and levelAntichain call each other.
  [[nodiscard]] inline std::vector<uint32_t> levelAntichain(Graph const& graph, Level level);

  // One root per parton-initiated jet: the HardProcess members that are partons. No
  // clustering; the flavour is the PDG id of the parton. Empty when statusFlags are not
  // available (the HepMC3 path). Not restricted to the signal interaction, but on 10 PU200
  // ttbar events no overlaid interaction has the isHardProcess flag. The roots are an
  // antichain, but the subgraphs can overlap: two colour-connected quarks fragment
  // through one string.
  [[nodiscard]] inline std::vector<uint32_t> partonJets(Graph const& graph) {
    std::vector<uint32_t> roots = levelAntichain(graph, Level::HardProcess);
    roots.erase(
        std::remove_if(
            roots.begin(), roots.end(), [&graph](uint32_t id) { return !isParton(graph.particles()[id].pdgId); }),
        roots.end());
    return roots;
  }

  // Drop every member that another member covers. With keepDeepest false, a member with
  // a member ancestor is dropped, which keeps the earliest of each chain. keepDeepest
  // reverses the direction. Every level runs it: on a re-convergent history, a walk that
  // stops at a pi0 on one path reaches the photon of that pi0 on another path.
  inline void dropCoveredMembers(Graph const& graph, std::vector<uint32_t>& members, bool keepDeepest) {
    const uint32_t nParticles = graph.nParticles();
    std::vector<uint8_t> covered(nParticles, 0);
    std::vector<uint32_t> stack;
    stack.reserve(members.size());
    // Seed with the direct neighbours of the members, so a member is marked only when
    // another member reaches it.
    auto pushNeighbours = [&](uint32_t id) {
      if (keepDeepest) {
        for (const uint32_t vertexId : graph.productionVertices(id)) {
          if (vertexId >= graph.nVertices()) {
            continue;
          }
          for (const uint32_t parent : graph.incomingParticles(vertexId)) {
            // Skip self-loops, so a particle does not cover itself.
            if (parent != id && parent < nParticles && covered[parent] == 0) {
              covered[parent] = 1;
              stack.push_back(parent);
            }
          }
        }
      } else {
        for (const uint32_t vertexId : graph.decayVertices(id)) {
          if (vertexId >= graph.nVertices()) {
            continue;
          }
          for (const uint32_t child : graph.outgoingParticles(vertexId)) {
            if (child != id && child < nParticles && covered[child] == 0) {
              covered[child] = 1;
              stack.push_back(child);
            }
          }
        }
      }
    };
    for (uint32_t id : members) {
      pushNeighbours(id);
    }
    while (!stack.empty()) {
      const uint32_t id = stack.back();
      stack.pop_back();
      pushNeighbours(id);
    }

    std::erase_if(members, [&covered](uint32_t id) { return covered[id] != 0; });
  }

  // The members of a level, reduced to an antichain.
  [[nodiscard]] inline std::vector<uint32_t> levelAntichain(Graph const& graph, Level level) {
    if (level == Level::StableLegsFromInitialState) {
      std::vector<uint32_t> legs = stableLegsFromInitialState(graph);
      dropCoveredMembers(graph, legs, false);
      return legs;
    }
    if (level == Level::ReconstructableFromSignal) {
      std::vector<uint32_t> legs = reconstructableFromSignal(graph);
      dropCoveredMembers(graph, legs, false);
      return legs;
    }
    if (level == Level::ReconstructableFinalState) {
      std::vector<uint32_t> legs = reconstructableFinalState(graph);
      dropCoveredMembers(graph, legs, false);
      return legs;
    }
    if (level == Level::UnderlyingEvent) {
      std::vector<uint32_t> legs = stableLegsFromUnderlyingEvent(graph);
      dropCoveredMembers(graph, legs, false);
      return legs;
    }
    if (level == Level::PartonJets) {
      // Filtered from HardProcess, which dropCoveredMembers already reduced.
      return partonJets(graph);
    }
    std::vector<uint32_t> candidates;
    const uint32_t nParticles = graph.nParticles();
    for (uint32_t id = 0; id < nParticles; ++id) {
      if (atLevel(graph, id, level)) {
        candidates.push_back(id);
      }
    }
    // Which end of a chain of candidates to keep. Earliest by default: a candidate with a
    // candidate ancestor is a duplicate of it.
    // Deepest for HardProcess: the incoming partons also have the flag, and the level is
    // the outgoing particles.
    // Deepest for BHadrons and CHadrons: the member is the weakly decaying hadron, not the
    // B* above it, which decays at its production point. On 200 ttbar and 300 QCD
    // generator events the count is the same, 68.9% and 61.8% of chains keep a different
    // particle, and the median decay displacement is 0.46 cm instead of 0.000 cm.
    const bool keepDeepest = level == Level::HardProcess || level == Level::BHadrons || level == Level::CHadrons;
    dropCoveredMembers(graph, candidates, keepDeepest);
    return candidates;
  }

  // The persisted bit for a level, from kLevelTable. Throws for a level with no row.
  [[nodiscard]] inline LevelFlag levelFlagOf(Level level) {
    for (auto const& row : kLevelTable) {
      if (row.level == level) {
        return row.flag;
      }
    }
    throw cms::Exception("TruthLevels") << "level " << static_cast<int>(level) << " has no row in kLevelTable";
  }

  // The particles on a directed cycle, walking particle to child. Empty on a well-formed
  // graph. On a cycle, dropCoveredMembers lets a member reach itself, so the level loses
  // members. O(nParticles + nEdges), iterative because a shower chain can overflow the
  // call stack.
  [[nodiscard]] inline std::vector<uint32_t> particlesOnCycles(Graph const& graph) {
    enum : uint8_t { kUnseen = 0, kOnStack = 1, kDone = 2 };
    const uint32_t nParticles = graph.nParticles();
    std::vector<uint8_t> state(nParticles, kUnseen);
    std::vector<uint32_t> onCycle;
    // (particle, index of the next child to visit) so the walk can resume after a child.
    std::vector<std::pair<uint32_t, std::size_t>> stack;
    std::vector<uint32_t> children;

    auto childrenOf = [&graph](uint32_t id, std::vector<uint32_t>& out) {
      out.clear();
      for (const uint32_t vertexId : graph.decayVertices(id)) {
        if (vertexId >= graph.nVertices()) {
          continue;
        }
        for (const uint32_t child : graph.outgoingParticles(vertexId)) {
          out.push_back(child);
        }
      }
    };

    for (uint32_t root = 0; root < nParticles; ++root) {
      if (state[root] != kUnseen) {
        continue;
      }
      stack.emplace_back(root, 0);
      state[root] = kOnStack;
      while (!stack.empty()) {
        auto& [id, next] = stack.back();
        childrenOf(id, children);
        if (next >= children.size()) {
          state[id] = kDone;
          stack.pop_back();
          continue;
        }
        const uint32_t child = children[next];
        ++next;
        if (child >= nParticles) {
          continue;
        }
        if (state[child] == kOnStack) {
          // Back edge: every particle on the stack from the child up is on the cycle.
          for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
            onCycle.push_back(it->first);
            if (it->first == child) {
              break;
            }
          }
        } else if (state[child] == kUnseen) {
          state[child] = kOnStack;
          stack.emplace_back(child, 0);
        }
      }
    }
    std::sort(onCycle.begin(), onCycle.end());
    onCycle.erase(std::unique(onCycle.begin(), onCycle.end()), onCycle.end());
    return onCycle;
  }

  // Stamp every particle with the levels it belongs to. Call it on the complete graph.
  // It clears the owned bits first, so it is idempotent.
  inline void fillLevelFlags(Graph& graph) {
    // The walks index the CSR arrays directly, so the particle offset arrays must have
    // nParticles + 1 entries.
    if (graph.nParticles() == 0) {
      return;
    }
    if (graph.particleToDecayVertexOffsets().size() != static_cast<std::size_t>(graph.nParticles()) + 1 ||
        graph.particleToProductionVertexOffsets().size() != static_cast<std::size_t>(graph.nParticles()) + 1) {
      throw cms::Exception("TruthLevels")
          << "fillLevelFlags needs CSR offsets of size nParticles + 1 (" << graph.nParticles() + 1 << "), found "
          << graph.particleToDecayVertexOffsets().size() << " and " << graph.particleToProductionVertexOffsets().size();
    }
    // A cycle thins or empties the levels it touches, so it is reported. The stamping
    // continues, because the levels that no cycle reaches stay correct.
    if (const std::vector<uint32_t> cyclic = particlesOnCycles(graph); !cyclic.empty()) {
      edm::LogWarning("TruthLevels") << cyclic.size() << " particles lie on a directed cycle, first at id "
                                     << cyclic.front() << " (pdgId " << graph.particles()[cyclic.front()].pdgId
                                     << "). A level whose members a cycle reaches erases them and comes out "
                                        "empty or thinned, so treat the level counts of this event as unreliable.";
    }
    // Clear only the owned bits. The selection post-processing sets LevelFlag::Signal.
    for (auto& particle : graph.particles()) {
      particle.levelFlags &= ~kOwnedLevelFlags;
    }
    // The HardProcess antichain feeds two levels, itself and (filtered to partons)
    // the parton jets, so it is computed once.
    const std::vector<uint32_t> hardProcess = levelAntichain(graph, Level::HardProcess);
    for (const Level level : kAllLevels) {
      const LevelFlag flag = levelFlagOf(level);
      std::vector<uint32_t> ids;
      if (level == Level::HardProcess) {
        ids = hardProcess;
      } else if (level == Level::PartonJets) {
        ids = hardProcess;
        ids.erase(std::remove_if(
                      ids.begin(), ids.end(), [&graph](uint32_t id) { return !isParton(graph.particles()[id].pdgId); }),
                  ids.end());
      } else {
        ids = levelAntichain(graph, level);
      }
      for (const uint32_t id : ids) {
        if (id < graph.nParticles()) {
          graph.particles()[id].setLevel(flag);
        }
      }
    }
  }

  // Whether a particle has to be at one of the levels asked for, or at every one.
  enum class LevelMatch : uint8_t { Any, All };

  // The members of several levels, as views, in id order and each once. Any is the
  // union, not reduced to an antichain: levels nest (b quark, B hadron, D hadron). All is
  // the intersection. A repeated level counts once.
  [[nodiscard]] inline std::vector<Particle> particlesAtLevels(Graph const& graph,
                                                               std::vector<Level> const& levels,
                                                               LevelMatch match = LevelMatch::Any) {
    std::vector<Level> wanted = levels;
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());

    std::vector<uint32_t> ids;
    for (const Level level : wanted) {
      const auto members = levelAntichain(graph, level);
      ids.insert(ids.end(), members.begin(), members.end());
    }
    std::sort(ids.begin(), ids.end());

    std::vector<Particle> out;
    for (std::size_t i = 0; i < ids.size();) {
      std::size_t j = i;
      while (j < ids.size() && ids[j] == ids[i]) {
        ++j;
      }
      const bool keep = match == LevelMatch::Any || (j - i) == wanted.size();
      if (keep) {
        out.emplace_back(&graph, ids[i]);
      }
      i = j;
    }
    return out;
  }

  // The members of a level, as particle views.
  [[nodiscard]] inline std::vector<Particle> particlesAtLevel(Graph const& graph, Level level) {
    std::vector<Particle> members;
    for (const uint32_t id : levelAntichain(graph, level)) {
      members.emplace_back(&graph, id);
    }
    return members;
  }

}  // namespace truth

#endif
