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

    // Bitwise OR of the LevelFlag values this particle belongs to, filled once the graph
    // is complete. Placed here on purpose: it occupies the four-byte alignment hole
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

    [[nodiscard]] bool hasGen() const { return genNode >= 0; }
    [[nodiscard]] bool hasSim() const { return simNode >= 0; }
    [[nodiscard]] bool valid() const { return hasGen() || hasSim(); }

    [[nodiscard]] bool isAtLevel(LevelFlag flag) const { return (levelFlags & static_cast<uint32_t>(flag)) != 0; }
    void setLevel(LevelFlag flag) { levelFlags |= static_cast<uint32_t>(flag); }
  };

}  // namespace truth

#endif
