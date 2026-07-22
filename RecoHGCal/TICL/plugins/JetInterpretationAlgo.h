#ifndef RecoHGCal_TICL_JetInterpretationAlgo_h
#define RecoHGCal_TICL_JetInterpretationAlgo_h

// Jet (multi-particle) interpretation for TICL candidates (opinion-emitting,
// arbitration mode). In jets and dense environments SEVERAL tracks enter the same
// merged trackster: no single-track hypothesis can claim it (the charged-hadron
// energy gate E < p + slack correctly vetoes each individual track), so without a
// multi-track advocate the tracks are lost and the whole blob becomes one neutral.
// For each trackster with at least two compatible tracks this algorithm emits a Jet
// hypothesis carrying ALL those tracks, scored by how well the summed track momenta
// balance the calorimetric energy. A winning jet is resolved by the candidate
// assembly into one charged candidate per track (kinematics from the track, as in
// particle flow) plus one neutral residual for the calorimetric excess.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoHGCal/TICL/interface/TICLInterpretationAlgoBase.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace ticl {

  class JetInterpretationAlgo : public TICLInterpretationAlgoBase<reco::Track> {
  public:
    JetInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC);
    ~JetInterpretationAlgo() override;

    // Opinion-only algorithm: makeCandidates is intentionally a no-op.
    void makeCandidates(const Inputs &input,
                        edm::Handle<MtdHostCollection> inputTiming_h,
                        std::vector<Trackster> &resultTracksters,
                        std::vector<int> &resultCandidate,
                        std::vector<bool> &maskedTracksters) override;

    void makeOpinions(const Inputs &input,
                      edm::Handle<MtdHostCollection> inputTiming_h,
                      std::vector<Trackster> &hypothesisTracksters,
                      std::vector<Hypothesis> &hypotheses) override;

    void initialize(const HGCalDDDConstants *hgcons,
                    const hgcal::RecHitTools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override;

    static void fillPSetDescription(edm::ParameterSetDescription &iDesc);

  private:
    void makeRecoveryOpinions(const Inputs &input,
                              std::vector<Trackster> &hypothesisTracksters,
                              std::vector<Hypothesis> &hypotheses);

    // (eta,phi) window to associate tracks to a trackster.
    const double delta_tk_ts_;
    // Min trackster raw energy for a jet reading to be considered.
    const double min_trackster_energy_;
    // Single-track recovery (claim-and-attach): E/p band for the merged nearby
    // tracksters of a track the tight charged-hadron gates rejected.
    const double recovery_min_eop_;
    const double recovery_max_eop_;

    const HGCalDDDConstants *hgcons_;
    hgcal::RecHitTools rhtools_;
    edm::ESHandle<MagneticField> bfield_;
    edm::ESHandle<Propagator> propagator_;
  };

}  // namespace ticl

#endif
