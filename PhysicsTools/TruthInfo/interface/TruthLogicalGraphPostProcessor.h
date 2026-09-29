// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_interface_TruthLogicalGraphPostProcessor_h
#define PhysicsTools_TruthInfo_interface_TruthLogicalGraphPostProcessor_h

#include <cstdint>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SimDataFormats/TruthInfo/interface/Graph.h"

namespace truth {

  struct LogicalGraphPostProcessingConfig {
    bool collapseIntermediateGenParticles = true;

    // If true, remove every SIM logical particle with no positive-energy calorimeter or
    // tracker sim-hit at or below it, with its whole downstream subtree. GEN-only
    // descendants (neutrinos) go with it. The GEN skeleton outside removed subtrees is
    // kept. A no-op when the producer supplies no per-particle direct-hit vector.
    bool dropHitlessSimSubgraphs = true;

    // The most upstream particle of each matching chain becomes a root of the selected
    // graph. Empty applies no seed cut. The value 0 disables the selection and keeps the
    // full graph.
    std::vector<int32_t> seedPdgIds;

    // Seed on hadrons by heavy-flavor content: a hadron that contains any of these quark
    // flavors becomes a seed (5 for b, 4 for c). OR-ed with seedPdgIds.
    std::vector<int32_t> seedHadronFlavors;

    // Species that a detector reconstructs as an object although they decay (pi0). The
    // walk from the signal to its reconstructable products stops at them. The walk goes
    // through a species that is not listed (a1, rho).
    std::vector<int32_t> reconstructablePdgIds;

    // For each selected root, keep this many generations of ancestors above it
    // as context only: the ancestors and connecting vertices are kept, but not
    // their other descendants.
    uint32_t seedParentDepth = 0;

    // If true, stable final-state GEN particles outside the selection are kept, each with
    // its SIM subgraph, and attached to an artificial UnderlyingEvent source vertex. Used
    // only when a selection is active.
    bool keepStableSpectators = true;

    // If true, kept particles whose production vertices are all outside the selection are
    // attached to an artificial InitialState, UnderlyingEvent or BeamSideInput source
    // vertex. If false, they become graph roots, so each seed gives a separate component
    // (the ten taus of TenTau give ten components). Used only when a selection is active.
    bool attachSelectionSources = true;

    // If true, also keep the production vertex of each selected root and the other
    // outgoing particles of that vertex, with their decay subtrees (the VBF tagging
    // quarks). seedParentDepth does not reach these siblings. Used only when a selection
    // is active.
    bool keepProductionSiblings = false;

    // Pile-up filter, independent of the seed selection. It drops particles by the
    // EncodedEventId of their pp collision, before the seed selection.
    // If true, keep only the signal interaction (bunchCrossing 0 and event 0).
    bool signalOnly = false;

    // If not empty, keep only particles whose bunchCrossing is in this list ({0} keeps
    // in-time only). AND-ed with signalOnly.
    std::vector<int32_t> keepBunchCrossings;

    // Decay patterns of interest. Each group is an unordered, charge-sensitive
    // multiset of PDG ids; groups are OR-ed.
    //
    // Without seedPdgIds: a vertex whose outgoing PDG ids contain a group as a
    // sub-multiset is selected, and the matched particles plus their downstream
    // subgraphs are kept.
    //
    // With seedPdgIds: only seed roots whose effective decay products (after
    // following same-PDG radiating copy chains) contain a group are kept. If
    // the event contains no particle with a seed PDG id at all, the direct
    // vertex search is used as a fallback.
    std::vector<std::vector<int32_t>> decayPdgIdGroups;

    // Particles with these exact PDG ids are removed from the final logical graph.
    // If such a particle is internal, its production and decay vertices are merged
    // so that the graph remains navigable.
    std::vector<int32_t> ignoredPdgIds;

    // Exact logical particle ids to remove from the final logical graph.
    // These ids refer to the graph state at the moment the ignored-particle
    // collapsing step is applied.
    std::vector<uint32_t> ignoredParticleIds;
  };

  // Gives a GEN-only particle with no momentum the sum of its decay products that have
  // one. A decay here is a GEN vertex whose only incoming particle is this one, so a vertex
  // fed by several partons, a string or the beams adds nothing. The sum is exact when
  // every product has a momentum. Otherwise it misses the products Geant4 did not track,
  // which are those outside the g4SimHits Generator primary cuts. A decay chain resolves
  // from the bottom up, and a product that closes a cycle is not added.
  void fillMomentumFromDecayProducts(Graph& graph);

  class TruthLogicalGraphPostProcessor {
  public:
    TruthLogicalGraphPostProcessor() = default;
    explicit TruthLogicalGraphPostProcessor(LogicalGraphPostProcessingConfig config);

    static edm::ParameterSetDescription psetDescription();
    static LogicalGraphPostProcessingConfig configFromPSet(edm::ParameterSet const& pset);

    // The configuration of this instance.
    [[nodiscard]] LogicalGraphPostProcessingConfig const& config() const { return config_; }

    // particleDirectHit[i] != 0 marks that logical particle i has a positive-energy
    // calorimeter or tracker sim-hit on its own SimTrack. It is aligned to the input
    // particle ids. An empty vector disables dropHitlessSimSubgraphs.
    [[nodiscard]] Graph process(Graph input, std::vector<uint8_t> const& particleDirectHit = {}) const;

  private:
    LogicalGraphPostProcessingConfig config_;
  };

}  // namespace truth

#endif
