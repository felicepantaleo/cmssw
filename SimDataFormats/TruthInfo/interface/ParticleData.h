// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef SimDataFormats_TruthInfo_interface_ParticleData_h
#define SimDataFormats_TruthInfo_interface_ParticleData_h

#include <cstdint>
#include <vector>

#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/TruthInfo/interface/Checkpoint.h"

namespace truth {

  // Membership of the graph levels, so a graph is self-describing about them wherever it
  // is read: the dot dump, a job log, or a consumer outside CMSSW that has the branch but
  // none of the CMSSW headers.
  //
  // Only levels that are pure functions of what the graph already stores live here, and
  // that is the rule for adding one. `signal` and the selected branch roots are
  // deliberately absent: they depend on the signal seed species and on the BranchSelector
  // thresholds, both of which are configured downstream of the graph build, so storing
  // them would make every threshold change a full re-production of the samples. Those
  // stay as the associator's per-event index products.
  //
  // Because every bit here is reproducible from the graph, a consumer that suspects a
  // file predates a change to the level definitions can re-derive with levelAntichain()
  // and compare, which is what LevelFlags_t checks.
  enum class LevelFlag : uint32_t {
    StableLegsFromUpstream = 1u << 0,
    HardProcess = 1u << 1,
    StableDecayProducts = 1u << 2,
    CaloBoundary = 1u << 3,
    // The resonance the sample was generated for: the most upstream particles matching
    // the selection preset's seed species. Two tops, one Z, one Higgs, ten taus.
    //
    // Unlike the four levels above this is NOT a pure function of the graph: it depends
    // on the preset's seed species. That is why Graph records the seed PDG ids that
    // produced it, so the bit stays re-derivable and the dumper audit can still check it.
    // A graph whose recorded seeds are empty carries no Signal bits.
    Signal = 1u << 4,
    // The first RECONSTRUCTABLE decay products of the signal: walk down from each Signal
    // root and stop at the first thing a detector could reconstruct as an object.
    //
    // That is not the same as the first generator-stable particle. A pi0 decays instantly
    // to two photons, but it is the pi0 the analysis reconstructs, so the walk stops
    // there and labels the pi0. A tau to three charged pions labels the three pions.
    // Intermediate resonances the detector cannot see as objects, an a1 or a rho, are
    // walked THROUGH and never labelled, unless their PDG id is added to the graph's
    // reconstructablePdgIds. Neutrinos are dropped, being invisible.
    //
    // The pdg ids that terminate the walk are recorded on the Graph, so like Signal this
    // level stays re-derivable by a reader that has only the graph.
    //
    // An antichain by construction: the walk stops at each leg, so no member can be an
    // ancestor of another. Derived from Signal, so it inherits the same requirement that
    // the seed species be recorded for it to stay re-derivable.
    ReconstructableFromSignal = 1u << 5,
    // The stable legs of the UNDERLYING EVENT, the spectator activity the artificial
    // UnderlyingEvent vertex collects. The counterpart of StableLegsFromUpstream, which
    // holds the ISR and upstream side of the same interaction. Together with the signal
    // levels these partition what the event contains: what the analysis asked for, what
    // radiated into it, and what came along with it.
    //
    // Like StableLegsFromUpstream it exists only when a selection preset ran, since the
    // artificial vertices are what the preset builds. Empty otherwise, not wrong.
    UnderlyingEvent = 1u << 6,
    // One entry per parton-initiated jet: the hard-process legs that are quarks or
    // gluons, each standing for everything downstream of it. There is no clustering and
    // no cone; the jet IS the descendant subgraph of the parton, and its flavour is the
    // parton's PDG id.
    //
    // A subset of HardProcess, so it inherits that level's deepest-element rule and with
    // it the answer to the nesting question: a top has a hard-process b below it, so the
    // b is kept and the top is not. That is the physical rule as well, since the top
    // decays before it hadronises and only the b makes a jet. A top-initiated jet is
    // therefore never a member unless the top is the deepest hard-process parton.
    //
    // An antichain because HardProcess is one and this only removes members.
    PartonJets = 1u << 7,
    // The b hadron and the c hadron each event produced, taking the FIRST one of that
    // flavour along a chain. A B* radiates down to a B, and both are b hadrons, so the
    // earliest-element antichain keeps the B* and drops the B: one member per physical
    // heavy-flavour decay rather than one per generator copy.
    //
    // Beauty and charm are separate levels ON PURPOSE. A B decays to a D, so the two
    // flavours are nested, and a single combined level would keep the B and silently drop
    // every D. Split, each is an antichain in its own right and both are answerable.
    //
    // Distinct from PartonJets, and the difference is the measurement: on no-PU ttbar the
    // b quark subgraph holds 961 and 1932 sim hits while the B hadron below it holds 758
    // and 519, so the quark is the whole jet including fragmentation and the hadron is the
    // displaced-vertex part a b tagger reconstructs.
    BHadrons = 1u << 8,
    CHadrons = 1u << 9,
  };

  // What a particle IS, mirroring VertexRole on the vertex side. Absence of a GEN and a
  // SIM back-reference does NOT identify a synthetic particle: connectors have neither,
  // and so would anything else artificial, so the kind has to be stated rather than
  // inferred. Guessing it from empty fields silently conflated the two.
  enum class ParticleRole : uint8_t {
    // A generator or Geant4 particle.
    Normal = 0,
    // Artificial: produced at an Interaction vertex and decaying at the Upstream or
    // UnderlyingEvent sub-vertex, so those descend from one interaction root.
    Connector = 1,
    // Artificial: stands in for a resonance the generator never wrote, so the signal
    // level is answerable on a non-resonant sample. Its momentum is an ACCOUNTING sum
    // over the hard-process legs and is not a generator quantity.
    SignalStandIn = 2,
  };

  struct ParticleData {
    // Optional provenance/debug back-references to the raw TruthGraph nodes.
    // -1 means "not available".
    int32_t genNode = -1;
    int32_t simNode = -1;

    // Merged metadata.
    int32_t pdgId = 0;
    int16_t status = 0;

    // Packed reco::GenStatusFlags bitfield, when available.
    // 0 means "not available" or "no flags set".
    uint16_t statusFlags = 0;

    // SIM event id when available, 0 otherwise.
    uint64_t eventId = 0;

    // GEN connected component id from the raw TruthGraph, -1 if not applicable.
    int32_t genEvent = -1;

    // Bitwise OR of the LevelFlag values this particle belongs to. The four antichain
    // levels are filled once the graph is complete; Signal is set earlier, by the
    // selection post-processing that knows the seed species, and survives the graph
    // rewrite because it travels on the particle rather than as an index. Placed here on purpose: it occupies the four-byte alignment hole
    // between genEvent and momentum, so sizeof(ParticleData) stays 96 (asserted in
    // LevelFlags_t). Zero means either "belongs to no level" or "written before this
    // member existed", which is why the flags are checkable against levelAntichain().
    uint32_t levelFlags = 0;

    // Standalone payload.
    // Nominal physics four-momentum.
    // For GEN+SIM particles, this is the GEN four-momentum.
    // For SIM-only particles, this is the SimTrack four-momentum.
    math::XYZTLorentzVectorD momentum;

    // Optional trajectory checkpoints.
    std::vector<Checkpoint> checkpoints;

    // True for SIM particles that Geant4 flagged as back-scattered (albedo): the
    // track crossed the Tracker<->CALO boundary inward. From SimTrack::isFromBack-
    // Scattering(); always false for GEN-only particles.
    bool backscattered = false;

    // Real particle, connector, or synthetic stand-in. Sits in the tail padding after
    // backscattered, so carrying it keeps sizeof(ParticleData) at 96.
    ParticleRole role = ParticleRole::Normal;

    [[nodiscard]] bool hasGen() const { return genNode >= 0; }
    [[nodiscard]] bool hasSim() const { return simNode >= 0; }
    [[nodiscard]] bool valid() const { return hasGen() || hasSim(); }

    // True for anything the graph invented. Never read the momentum of such a particle
    // as a generator quantity.
    [[nodiscard]] bool isSynthetic() const { return role != ParticleRole::Normal; }

    [[nodiscard]] bool isAtLevel(LevelFlag flag) const { return (levelFlags & static_cast<uint32_t>(flag)) != 0; }
    void setLevel(LevelFlag flag) { levelFlags |= static_cast<uint32_t>(flag); }
  };

}  // namespace truth

#endif
