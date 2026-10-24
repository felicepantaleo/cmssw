#ifndef RecoTICL_Linking_TracksterLinkingbyCornetto_h
#define RecoTICL_Linking_TracksterLinkingbyCornetto_h

// Trackster linking by a parent tree.
//
// Pair test with anchor a and other trackster b (RecoTICL/Linking/interface/CornettoGeometry.h): the
// axis points from the origin to the barycenter of a. With D = bary_b - bary_a, s = D . axis and
// dT = |D - s axis|, the pair is compatible when -maxBackwardDistance <= s <= maxLongitudinalDistance,
// dT <= transverseRadius0 + transverseSlope * |s|, |deta| and |dphi| <= etaWindow, both are in the same
// endcap, and:
// - EM anchor: when typeVetoProbability > 0 and the EM probability of a is above typeVetoProbability, the
//   transverse limit is emTransverseRadius0 + emTransverseSlope * |s| instead.
// - time: when timeCompatibilityNSigma > 0 and both time errors are positive, t_b - t_a - s / c agrees
//   with 0 within timeCompatibilityNSigma times the combined error, with timeResolutionFloor added in
//   quadrature. The trackster times must be local to the barycenter.
// - type: when typeVetoProbability > 0 and both carry a PID, the pair is refused if one EM probability is
//   above typeVetoProbability and the other below 1 - typeVetoProbability.
//
// Grouping, each step independent per trackster:
// 1. The parent of b is the highest-energy trackster a with more energy than b (equal energy: smaller
//    index) for which the pair (a, b) is compatible. A trackster without a parent is a root. A trackster
//    with a non-finite barycenter or energy is a root and is never a parent.
// 2. Each trackster follows its parents to its root.
// 3. A trackster stays with its root only if the pair (root, trackster) is compatible; else it is a group
//    of its own.
//
// Optional second pass on the groups (mergeFragments, mergeDropped): the same parent tree on the groups, with the
// pair test of the merge cone. A group has the raw energy, the energy-weighted barycenter, the energy-weighted EM
// probability of its members with a PID, and the error-weighted time of its members moved to its barycenter. Only a
// group that passes the emission rule is a parent. A group that passes it can have a parent when mergeFragments is
// true, a group that does not pass it when mergeDropped is true. With mergeHadronicOnly and a positive
// typeVetoProbability, a pair with an EM group (EM probability above typeVetoProbability) is refused.
//
// Emission rule: a group is emitted when its raw energy is at least minEmittedEnergy and its raw pt is at least
// minEmittedPt, or its raw energy is at least minEmittedEnergyAnyPt (when positive). A group with an EM probability
// above typeVetoProbability uses minEmittedEnergyEM and minEmittedPtEM instead. The pt uses the direction of the
// energy-weighted barycenter of the members.

#include <array>
#include <vector>

#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "RecoTICL/Linking/interface/CornettoGeometry.h"
#include "RecoTICL/Linking/interface/CornettoGrouping.h"
#include "RecoTICL/Linking/interface/TracksterLinkingAlgoBase.h"

namespace ticl {

  class TracksterLinkingbyCornetto : public TracksterLinkingAlgoBase {
  public:
    TracksterLinkingbyCornetto(const edm::ParameterSet& conf,
                               edm::ConsumesCollector iC,
                               cms::Ort::ONNXRuntime const* onnxRuntime = nullptr);
    ~TracksterLinkingbyCornetto() override = default;

    void linkTracksters(const Inputs& input,
                        std::vector<Trackster>& resultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) override;

    void initialize(const HGCalDDDConstants* hgcons,
                    const ticlgeom::Tools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override {}

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
      iDesc.add<float>("etaWindow", 0.3f)
          ->setComment("Max |deta| and |dphi| between the barycenters. Must be in (0, pi).");
      iDesc.add<float>("maxLongitudinalDistance", 60.f)
          ->setComment("Max separation [cm] downstream of the anchor along its axis. Must be positive.");
      iDesc.add<float>("maxBackwardDistance", 20.f)
          ->setComment("Max separation [cm] upstream of the anchor along its axis. Must not be negative.");
      iDesc.add<float>("transverseRadius0", 4.f)
          ->setComment("Max transverse distance at zero separation [cm]. Must not be negative.");
      iDesc.add<float>("transverseSlope", 0.05f)
          ->setComment("Growth of the max transverse distance per cm of separation. Must not be negative.");
      iDesc.add<float>("emTransverseRadius0", 1.5f)
          ->setComment(
              "Max transverse distance at zero separation [cm] for an EM anchor. Must not be negative or above "
              "transverseRadius0.");
      iDesc.add<float>("emTransverseSlope", 0.05f)
          ->setComment(
              "Growth of the max transverse distance per cm of separation for an EM anchor. Must not be negative or "
              "above transverseSlope.");
      iDesc.add<float>("timeCompatibilityNSigma", 3.f)
          ->setComment(
              "Max |time difference at the anchor barycenter depth| in combined sigmas. Applies when both time "
              "errors are positive. Values <= 0 disable the gate.");
      iDesc.add<float>("timeResolutionFloor", 0.05f)
          ->setComment("Time resolution [ns] added in quadrature to the combined time error. Must not be negative.");
      iDesc.add<float>("typeVetoProbability", 0.6f)
          ->setComment(
              "EM/hadronic veto threshold: 0 disables the veto, else it must be in [0.5, 1]. A pair is refused if "
              "one EM probability is above this value and the other below 1 - this value. Applies only when both "
              "carry a PID.");
      iDesc.add<float>("minEmittedEnergy", 1.f)
          ->setComment("Min raw energy [GeV] of an emitted group. Applies after linking.");
      iDesc.add<float>("minEmittedPt", 0.5f)
          ->setComment("Min raw pt [GeV] of an emitted group. Applies after linking.");
      iDesc.add<float>("minEmittedEnergyEM", 1.f)
          ->setComment("Min raw energy [GeV] of an emitted EM group (EM probability above typeVetoProbability).");
      iDesc.add<float>("minEmittedPtEM", 0.5f)->setComment("Min raw pt [GeV] of an emitted EM group.");
      iDesc.add<float>("minEmittedEnergyAnyPt", 0.f)
          ->setComment(
              "A group at or above this raw energy [GeV] is emitted at any pt. Values <= 0 disable it. The minimum "
              "energy still applies.");
      iDesc.add<bool>("mergeFragments", false)
          ->setComment("Second pass: an emitted group can join an emitted group of higher energy.");
      iDesc.add<bool>("mergeDropped", false)
          ->setComment("Second pass: a group that is not emitted can join an emitted group.");
      iDesc.add<bool>("mergeHadronicOnly", true)->setComment("Second pass: refuse a pair with an EM group.");
      iDesc.add<float>("mergeMinParentEnergy", 0.f)->setComment("Second pass: min raw energy [GeV] of a parent.");
      iDesc.add<float>("mergeMaxLongitudinalDistance", 60.f)
          ->setComment("Second pass: max separation [cm] downstream of the parent. Must be positive.");
      iDesc.add<float>("mergeMaxBackwardDistance", 20.f)
          ->setComment("Second pass: max separation [cm] upstream of the parent. Must not be negative.");
      iDesc.add<float>("mergeTransverseRadius0", 4.f)
          ->setComment("Second pass: max transverse distance at zero separation [cm]. Must not be negative.");
      iDesc.add<float>("mergeTransverseSlope", 0.05f)
          ->setComment(
              "Second pass: growth of the max transverse distance per cm of separation. Must not be negative.");
      iDesc.add<float>("mergeEMTransverseRadius0", 1.5f)
          ->setComment("Second pass: max transverse distance at zero separation [cm] for an EM parent.");
      iDesc.add<float>("mergeEMTransverseSlope", 0.05f)
          ->setComment("Second pass: growth of the max transverse distance per cm of separation for an EM parent.");
      TracksterLinkingAlgoBase::fillPSetDescription(iDesc);
    }

  private:
    using Tiles = std::array<TICLLayerTile, 2>;  // one per endcap, index 1 for z > 0

    // Per-trackster quantities the pair test reads.
    struct Features {
      std::vector<math::XYZVectorF> bary;
      std::vector<float> energy;
      std::vector<float> eta;
      std::vector<float> phi;
      std::vector<float> rad;   // |bary|
      std::vector<float> em;    // EM probability, negative without a PID
      std::vector<float> time;  // local to the barycenter [ns]
      std::vector<float> timeError;
      std::vector<bool> valid;  // finite barycenter and energy
    };

    Features computeFeatures(const Inputs& input) const;
    // Features of the groups, from the features f of their members.
    Features groupFeatures(const Features& f, const std::vector<std::vector<unsigned int>>& groups) const;
    bool compatible(const Features& f, unsigned int a, unsigned int b) const {
      return compatibleIn(cone_, emCone_, f, a, b);
    }
    bool compatibleIn(const cornetto::Cone& cone,
                      const cornetto::Cone& emCone,
                      const Features& f,
                      unsigned int a,
                      unsigned int b) const;
    bool mergeCompatible(const Features& g, unsigned int a, unsigned int b) const;
    bool timeCompatible(float tAnchor, float errAnchor, float tOther, float errOther, float path) const;
    bool emits(float energy, float pt, float em) const;
    // The parent of each node: the highest-ranked node a above it, with allowed(a, node) true, that passes
    // test(a, node).
    template <typename Allowed, typename Test>
    std::vector<unsigned int> findParentsIn(
        const Features& f, const Tiles& tiles, const cornetto::Cone& cone, Allowed&& allowed, Test&& test) const;
    std::vector<unsigned int> findParents(const Features& f, const Tiles& tiles) const;
    // The groups after the second pass.
    std::vector<std::vector<unsigned int>> mergeGroups(const Features& f,
                                                       std::vector<std::vector<unsigned int>> groups) const;

    const float etaWindow_;
    const cornetto::Cone cone_;
    const cornetto::Cone emCone_;  // cone of an EM anchor, inside cone_
    const bool useTimeGate_;
    const float timeNSigma2_;
    const float timeFloor2_;
    const float typeVetoProbability_;
    const float minEmittedEnergy_;
    const float minEmittedPt_;
    const float minEmittedEnergyEM_;
    const float minEmittedPtEM_;
    const float minEmittedEnergyAnyPt_;
    const bool mergeFragments_;
    const bool mergeDropped_;
    const bool mergeHadronicOnly_;
    const float mergeMinParentEnergy_;
    const cornetto::Cone mergeCone_;
    const cornetto::Cone mergeEMCone_;  // merge cone of an EM parent, inside mergeCone_
  };

}  // namespace ticl

#endif
