// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#ifndef PhysicsTools_TruthInfo_interface_BranchSelector_h
#define PhysicsTools_TruthInfo_interface_BranchSelector_h

#include <cstdint>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/Branch.h"

namespace truth {

  // Kinematic / provenance selection of truth Branches, mirroring the cut
  // surface of TrackingParticleSelector / CaloParticleSelector but applied to a
  // Branch (the dynamic successor of TrackingParticle/CaloParticle). The branch
  // kinematics are taken from its defining root particle.
  class BranchSelector {
  public:
    struct Config {
      double ptMin = 0.;
      double ptMax = 1e100;
      double etaMin = -1e100;
      double etaMax = 1e100;
      std::vector<int32_t> pdgIds;  // empty = accept all; matched on signed PDG id
      bool signalOnly = false;      // bunchCrossing == 0 and event == 0
      bool intimeOnly = false;      // bunchCrossing == 0
      bool chargedOnly = false;     // root particle electrically charged
      bool invertEta = false;       // keep |eta| OUTSIDE [etaMin, etaMax]
      // Apply the pt and eta cuts ONLY to a root Geant4 actually tracked. The cuts describe what a detector can see, and the momentum of a
      // root that decayed is not a detector observable: a resonance produced at rest
      // carries pt about 0 and therefore |eta| -> infinity, so a pt>1 GeV or |eta|<4 cut
      // throws it away while its decay products are all over the calorimeter. Measured on
      // DYToLL: 43% of the Z bosons were rejected that way, every one of them with a
      // subgraph full of sim-hits.
      bool kinematicsOnStableOnly = true;
    };

    BranchSelector() = default;
    explicit BranchSelector(Config config) : config_(std::move(config)) {}

    [[nodiscard]] bool operator()(Branch const& branch) const;

    [[nodiscard]] Config const& config() const { return config_; }

  private:
    Config config_;
  };

}  // namespace truth

#endif
