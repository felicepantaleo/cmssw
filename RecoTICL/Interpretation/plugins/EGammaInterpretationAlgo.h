#ifndef RecoHGCal_TICL_EGammaInterpretationAlgo_h
#define RecoHGCal_TICL_EGammaInterpretationAlgo_h

// e/gamma interpretation: an electron hypothesis per track matched to a supercluster, and a photon hypothesis per
// EM-like supercluster. The input tracksters are the superclusters.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoTICL/Interpretation/interface/TICLInterpretationAlgoBase.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace ticl {

  class EGammaInterpretationAlgo : public TICLInterpretationAlgoBase<reco::Track> {
  public:
    EGammaInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC);
    ~EGammaInterpretationAlgo() override;

    // The algorithm gives hypotheses only: makeCandidates does nothing.
    void makeCandidates(const Inputs &input,
                        edm::Handle<MtdHostCollection> inputTiming_h,
                        std::vector<Trackster> &resultTracksters,
                        std::vector<int> &resultCandidate,
                        std::vector<bool> &maskedTracksters,
                        std::vector<std::vector<unsigned int>> &linkedResultTracksters) override;

    void makeOpinions(const Inputs &input,
                      edm::Handle<MtdHostCollection> inputTiming_h,
                      std::vector<Trackster> &hypothesisTracksters,
                      std::vector<Hypothesis> &hypotheses) override;

    void initialize(const HGCalDDDConstants *hgcons,
                    const ticlgeom::Tools rhtools,
                    const edm::ESHandle<MagneticField> bfieldH,
                    const edm::ESHandle<Propagator> propH) override;

    static void fillPSetDescription(edm::ParameterSetDescription &iDesc);

  private:
    // (eta, phi) window at the HGCAL front between the track impact and the supercluster.
    const float delta_tk_sc_;
    // Electron identity: E/p window and min EM fraction of the supercluster.
    const float eop_min_;
    const float eop_max_;
    // Min EM energy fraction of a supercluster for an electron or a photon hypothesis.
    const float min_em_fraction_;
    // Electron hypotheses per track: the nearest superclusters in the window, at most this number.
    const unsigned int max_matches_;
    // Min supercluster raw energy [GeV].
    const float min_supercluster_energy_;
  };

}  // namespace ticl

#endif
