#ifndef RecoTICL_Linking_TracksterLinkingbyCornetto_h
#define RecoTICL_Linking_TracksterLinkingbyCornetto_h

// Trackster linking by a parent tree.
//
// Pair test with anchor a and other trackster b (RecoTICL/Linking/interface/CornettoGeometry.h): the
// axis points from the origin to the barycenter of a. With D = bary_b - bary_a, s = D . axis and
// dT = |D - s axis|, the pair is compatible when -maxBackwardDistance <= s <= maxLongitudinalDistance,
// dT <= transverseRadius0 + transverseSlope * |s|, |deta| and |dphi| <= etaWindow, both are in the same
// endcap, and:
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
// A group is not emitted when its raw energy is below minEmittedEnergy or its raw pt is below minEmittedPt.
// The pt uses the direction of the energy-weighted barycenter of the members.

#include <array>
#include <vector>

#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "RecoTICL/Linking/interface/CornettoGeometry.h"
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
      std::vector<bool> valid;  // finite barycenter and energy
    };

    Features computeFeatures(const Inputs& input) const;
    bool compatible(const Inputs& input, const Features& f, unsigned int a, unsigned int b) const;
    bool timeCompatible(float tAnchor, float errAnchor, float tOther, float errOther, float path) const;
    std::vector<unsigned int> findParents(const Inputs& input, const Features& f, const Tiles& tiles) const;
    // Groups ordered by their smallest member. Each input trackster is in exactly one group.
    std::vector<std::vector<unsigned int>> buildGroups(const Inputs& input,
                                                       const Features& f,
                                                       const std::vector<unsigned int>& parents) const;

    const float etaWindow_;
    const cornetto::Cone cone_;
    const bool useTimeGate_;
    const float timeNSigma2_;
    const float timeFloor2_;
    const float typeVetoProbability_;
    const float minEmittedEnergy_;
    const float minEmittedPt_;
  };

}  // namespace ticl

#endif
