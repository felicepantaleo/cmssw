// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_interface_BranchSelector_h
#define PhysicsTools_TruthInfo_interface_BranchSelector_h

#include <cstdint>
#include <limits>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/Branch.h"

namespace truth {

  // Kinematic and provenance selection of truth Branches, with the cuts of
  // TrackingParticleSelector and CaloParticleSelector. The branch kinematics are
  // taken from its root particle.
  class BranchSelector {
  public:
    struct Config {
      // Unbounded by default: the sentinels are the limits of the type.
      float ptMin = 0.f;
      float ptMax = std::numeric_limits<float>::max();
      float etaMin = std::numeric_limits<float>::lowest();
      float etaMax = std::numeric_limits<float>::max();
      std::vector<int32_t> pdgIds;  // empty = accept all; matched on signed PDG id
      bool signalOnly = false;      // bunchCrossing == 0 and event == 0
      bool intimeOnly = false;      // bunchCrossing == 0
      bool chargedOnly = false;     // root particle electrically charged
      bool invertEta = false;       // keep eta OUTSIDE [etaMin, etaMax]
      // Skip the pt and eta cuts for a root whose momentum is not a detector observable:
      // a parton, a top quark, a boson or another resonance that Geant4 never tracked and
      // the generator did not emit as final state. A resonance at rest has pt about 0 and
      // a large |eta|, so a pt or eta cut rejects it. Measured on DYToLL: the cuts reject
      // 43% of the Z bosons. A hadron, a lepton and a photon keep the cuts, tracked or not,
      // and so does every generator final-state particle.
      bool kinematicsOnStableOnly = true;
    };

    // The cuts that are ALSO efficiency-plot axes. An efficiency against pt must not apply
    // the pt cut to its own denominator, or the cut deforms the turn-on. Measured on no-PU
    // ttbar: the caloBoundary denominator in the first pt bin is 10024 with the cut and
    // 144529 without, a factor 14.4; the second bin changes by a factor 1.05. So these two
    // are reported per branch, and the consumer decides per axis. No plot uses the other
    // cuts as an x axis.
    enum class CutBit : uint32_t { None = 0, Pt = 1u << 0, Eta = 1u << 1 };

    BranchSelector() = default;
    explicit BranchSelector(Config config) : config_(std::move(config)) {}

    // True when the branch passes every cut.
    [[nodiscard]] bool operator()(Branch const& branch) const;

    // The cuts that are not plot axes. A branch failing any of them is not a candidate at
    // all and may not enter any plot, whatever its kinematics.
    [[nodiscard]] bool passesNonKinematic(Branch const& branch) const;

    // The CutBit mask of the plotted-axis cuts this branch FAILS; 0 means it passes both.
    // With kinematicsOnStableOnly, a branch whose root momentum is not observable fails
    // neither.
    [[nodiscard]] uint32_t failedKinematicCuts(Branch const& branch) const;

    [[nodiscard]] Config const& config() const { return config_; }

  private:
    Config config_;
  };

}  // namespace truth

#endif
