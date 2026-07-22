#ifndef RecoHGCal_TICL_EGammaInterpretationAlgo_h
#define RecoHGCal_TICL_EGammaInterpretationAlgo_h

// e/gamma interpretation for TICL candidates (opinion-emitting, arbitration mode).
//
// EM showers are reconstructed twice in TICL v5: their CLUE3D tracksters flow both
// into the Skeletons linking (the hadronic view, ticlTracksterLinks) and, in
// parallel, into the superclustering (the EM view). This algorithm reads the
// SUPERCLUSTERED tracksters plus the tracks and asserts scored hypotheses:
//   - electron: a track matched to a supercluster (the KF general track: the HGCAL
//     GSF chain is structurally DOWNSTREAM of ticlCandidate, via
//     particleFlowClusterHGCal -> particleFlowSuperClusterHGCal -> electron seeds,
//     so GSF tracks cannot be consumed here without a circular dependency);
//   - photon:   an EM-like supercluster (also emitted for GSF-matched ones, so the
//               photon interpretation naturally takes over when the electron
//               hypothesis loses arbitration).
// It consumes nothing: the TICLCandidateProducer arbitrates the hypotheses of all
// interpretations, enforcing layer-cluster-level energy exclusivity, which is what
// resolves the same shower appearing in both the hadronic and the EM collections.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoHGCal/TICL/interface/TICLInterpretationAlgoBase.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace ticl {

  class EGammaInterpretationAlgo : public TICLInterpretationAlgoBase<reco::Track> {
  public:
    EGammaInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC);
    ~EGammaInterpretationAlgo() override;

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
    // (eta,phi) window for the track <-> supercluster match.
    const double delta_tk_sc_;
    // Electron identity gate: the matched supercluster must carry most of the track
    // momentum (E/p window) and be EM-like. Without it a charged hadron's EM subshower
    // outranks the charged-hadron hypothesis and hijacks the track.
    const double eop_min_;
    const double eop_max_;
    // Min EM energy fraction for a supercluster to yield a photon hypothesis.
    const double min_em_fraction_;
    // Min supercluster raw energy considered at all.
    const double min_supercluster_energy_;

    const HGCalDDDConstants *hgcons_;
    hgcal::RecHitTools rhtools_;
    edm::ESHandle<MagneticField> bfield_;
    edm::ESHandle<Propagator> propagator_;
  };

}  // namespace ticl

#endif
