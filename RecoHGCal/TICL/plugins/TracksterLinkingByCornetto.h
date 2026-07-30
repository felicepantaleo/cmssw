#ifndef RecoHGCal_TICL_TracksterLinkingByCornetto_H
#define RecoHGCal_TICL_TracksterLinkingByCornetto_H

// Cornetto: trackster linking by axis compatibility and union-find. The cone
// opening around the shower axis collects the fragments, like the ice-cream
// cone collects the scoops.
//
// Design goals, in order: NO discontinuities (a single continuous geometric rule,
// no energy thresholds, no behavior classes, no dropped components: the v5
// Skeletons plugin gates linking on min_trackster_energy/min_num_lcs, switches
// window sizes at energy thresholds and drops small singleton components
// entirely, which carves visible cliffs into the candidate momentum spectrum);
// simplicity (one pair test + connected components); pileup robustness hooks
// (timing gate now, the pair test is local so density or vertex compatibility
// can be added without restructuring); parallel friendliness (the three stages
// map onto standard GPU kernels for a future Alpaka port: eta binning =
// histogram fill, pair tests = independent map over tile neighborhoods,
// union-find = iterative label propagation).
//
// The pair test: for tracksters i, j the anchor a is the higher-energy one.
// With D = bary_j - bary_i, s = D dot axis_a (longitudinal separation along the
// anchor shower axis) and dT = |D - s axis_a| (transverse distance from the
// axis), the pair is linked when |s| < maxLongitudinalDistance + maxLongitudinalSlope
// * max(0, |z_anchor| - longitudinalZRef) (a longitudinal window that widens with
// calorimeter depth, since hadronic showers reach deeper into CE-H) and
// dT < transverseRadius0 + transverseSlope * |s| (a linearly opening cone) and
// the trackster times are compatible within timeCompatibilityNSigma when both
// are valid. All windows are continuous in energy by construction.
//
// Grouping mode. The original path feeds those pairs to union-find, i.e. it takes
// CONNECTED COMPONENTS, which are transitive: A-B and B-C put A and C together even
// when A and C fail the pair test. At PU200 the endcap holds O(3000) CLUE3D
// tracksters and that transitivity percolates, stitching a chain of individually
// reasonable links into one giant trackster (measured on SinglePi PU200: a >1 TeV
// trackster in 100% of events, median 8.2 TeV, holding ~57% of the endcap energy and
// ~44% of its layer clusters). seededGrowth replaces the closure with CLUE3D-style
// seeds and followers: tracksters above seedEnergy become cores, in decreasing
// energy, and each core attaches unowned satellites that sit inside its window,
// downstream of it (forwardOnly, since a shower develops in depth) and that do not
// swing the core axis by more than axisToleranceDeg. Satellites never merge with each
// other, so no chain can form and the giant component cannot appear (same sample:
// leading trackster 8.2 TeV -> O(240) GeV, in line with the Skeletons baseline).

#include <vector>
#include "RecoHGCal/TICL/interface/TracksterLinkingAlgoBase.h"

namespace ticl {

  class TracksterLinkingByCornetto : public TracksterLinkingAlgoBase {
  public:
    TracksterLinkingByCornetto(const edm::ParameterSet& conf,
                               edm::ConsumesCollector iC,
                               cms::Ort::ONNXRuntime const* onnxRuntime = nullptr);
    ~TracksterLinkingByCornetto() override = default;

    void linkTracksters(const Inputs& input,
                        std::vector<Trackster>& resultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                        std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) override;

    void initialize(const HGCalDDDConstants* hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override {}

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
      iDesc.add<double>("etaWindow", 0.3)
          ->setComment("Barycenter |deta| candidate window; pairs farther apart are never tested.");
      iDesc.add<double>("maxLongitudinalDistance", 60.0)
          ->setComment("Longitudinal window at the reference depth: max |separation along the anchor axis| [cm].");
      iDesc.add<double>("maxLongitudinalSlope", 0.0)
          ->setComment("Growth of the longitudinal window per cm of anchor |z| beyond longitudinalZRef; 0 = flat window.");
      iDesc.add<double>("longitudinalZRef", 320.0)
          ->setComment("Reference |z| [cm] (HGCAL front face); the window stays at maxLongitudinalDistance for |z| below it.");
      iDesc.add<double>("transverseRadius0", 5.0)->setComment("Cone transverse radius at zero separation [cm].");
      iDesc.add<double>("transverseSlope", 0.05)->setComment("Cone opening: radius growth per cm of separation.");
      iDesc.add<double>("timeCompatibilityNSigma", 3.0)
          ->setComment("Max |time difference| in combined sigmas when both tracksters have valid time.");
      iDesc.add<bool>("seededGrowth", false)
          ->setComment(
              "Grow energetic cores instead of taking connected components. False keeps the union-find path "
              "(bit-identical to the Alpaka kernel); true is percolation-proof at PU200.");
      iDesc.add<double>("seedEnergy", 5.0)
          ->setComment("Min raw energy [GeV] for a trackster to seed a core (seededGrowth only).");
      iDesc.add<double>("axisToleranceDeg", 5.0)
          ->setComment("Max swing [deg] of the core axis a single attachment may cause (seededGrowth only).");
      iDesc.add<double>("minEmittedEnergy", 0.0)
          ->setComment(
              "Drop a GROUP below this raw energy [GeV] when emitting. This is an emit-side cut, applied after "
              "the linking decisions, so it introduces no linking cliff; a link-side energy gate is what causes "
              "the spectrum discontinuities Cornetto exists to avoid. 0 emits everything.");
      iDesc.add<bool>("forwardOnly", true)
          ->setComment(
              "Attach only downstream of the core centroid (a shower develops in depth); seededGrowth only.");
      TracksterLinkingAlgoBase::fillPSetDescription(iDesc);
    }

  private:
    // Connected components over the pair predicate (the original path).
    void linkByUnionFind(const Inputs& input, std::vector<std::vector<unsigned int>>& components) const;
    // Energetic cores grown by attachment; satellites never merge with each other, which
    // is what makes it percolation-proof (see the class comment).
    void linkBySeededGrowth(const Inputs& input, std::vector<std::vector<unsigned int>>& components) const;

    const float etaWindow_;
    const float maxLongitudinalDistance_;
    const float transverseRadius0_;
    const float transverseSlope_;
    const float timeCompatibilityNSigma_;
    const float maxLongitudinalSlope_;
    const float longitudinalZRef_;
    const bool seededGrowth_;
    const float seedEnergy_;
    const float axisToleranceCos_;
    const bool forwardOnly_;
    const float minEmittedEnergy_;
  };

}  // namespace ticl

#endif
