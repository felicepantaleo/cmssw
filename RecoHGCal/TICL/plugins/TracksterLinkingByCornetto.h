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
// axis), the pair is linked when |s| < maxLongitudinalDistance and
// dT < transverseRadius0 + transverseSlope * |s| (a linearly opening cone) and
// the trackster times are compatible within timeCompatibilityNSigma when both
// are valid. All windows are continuous in energy by construction.

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
          ->setComment("Max |separation along the anchor axis| [cm].");
      iDesc.add<double>("transverseRadius0", 5.0)->setComment("Cone transverse radius at zero separation [cm].");
      iDesc.add<double>("transverseSlope", 0.05)->setComment("Cone opening: radius growth per cm of separation.");
      iDesc.add<double>("timeCompatibilityNSigma", 3.0)
          ->setComment("Max |time difference| in combined sigmas when both tracksters have valid time.");
      TracksterLinkingAlgoBase::fillPSetDescription(iDesc);
    }

  private:
    const float etaWindow_;
    const float maxLongitudinalDistance_;
    const float transverseRadius0_;
    const float transverseSlope_;
    const float timeCompatibilityNSigma_;
  };

}  // namespace ticl

#endif
