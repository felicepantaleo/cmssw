// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Which truth particles a reco object may be assigned to.
//
// A candidate root carries the hits of its whole subgraph, so an ancestor always covers
// its descendants and can win the association on score alone. Some of those ancestors are
// not particles a detector could see: the nodes the graph invents to summarise an
// interaction, the beam particles, the partons of the hard scatter and the electroweak
// bosons. These are barred as assignment targets.
//
// The rule is about WHAT the ancestor is, not about a vertex. A merged pi0 is reached by
// crossing a decay vertex, so no vertex process can be a barrier. A gun particle attached
// to the InitialState vertex and a spectator attached to the UnderlyingEvent vertex stay
// assignable.
//
// This is the assignment rule only. The barred particles stay candidate roots, because
// the hard-process and parton-jet denominators are made of these particles.

#ifndef PhysicsTools_TruthInfo_interface_AssignableTarget_h
#define PhysicsTools_TruthInfo_interface_AssignableTarget_h

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

namespace truth {

  // The W, the Z and the Higgs. No reco object is one of them: their branch is the pair
  // of legs they decayed to. The photon is a detector particle and is not one of these.
  [[nodiscard]] inline bool isElectroweakBoson(int32_t pdgId) {
    const int64_t a = std::abs(static_cast<int64_t>(pdgId));
    return a == 23 || a == 24 || a == 25;
  }

  // Each clause bars one class of non-detector particle. Defaults bar all of them;
  // extraBarredPdgIds is matched on the absolute value, so one entry covers a particle
  // and its antiparticle.
  struct AssignableTargetConfig {
    bool excludeSynthetic = true;
    bool excludeBeamParticles = true;
    bool excludePartons = true;
    bool excludeElectroweakBosons = true;
    std::vector<int32_t> extraBarredPdgIds;
  };

  [[nodiscard]] inline bool isAssignableTarget(Graph const& graph,
                                               uint32_t particleId,
                                               AssignableTargetConfig const& config) {
    if (particleId >= graph.nParticles()) {
      return false;
    }
    auto const& particle = graph.particles()[particleId];

    // A connector or a signal stand-in. Its momentum is an accounting sum and its
    // subgraph is everything below the node the graph invented.
    if (config.excludeSynthetic && particle.isSynthetic()) {
      return false;
    }
    if (config.excludePartons && isParton(particle.pdgId)) {
      return false;
    }
    if (config.excludeElectroweakBosons && isElectroweakBoson(particle.pdgId)) {
      return false;
    }
    if (!config.extraBarredPdgIds.empty()) {
      const int32_t absPdgId = static_cast<int32_t>(std::abs(static_cast<int64_t>(particle.pdgId)));
      if (std::find(config.extraBarredPdgIds.begin(), config.extraBarredPdgIds.end(), absPdgId) !=
          config.extraBarredPdgIds.end()) {
        return false;
      }
    }

    const auto production = graph.productionVertices(particleId);
    // Nothing produced it, so it is a beam particle and everything it covers is the
    // whole interaction.
    if (config.excludeBeamParticles && production.empty()) {
      return false;
    }
    return true;
  }

}  // namespace truth

#endif
