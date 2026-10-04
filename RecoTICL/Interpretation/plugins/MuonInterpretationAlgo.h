#ifndef RecoHGCal_TICL_MuonInterpretationAlgo_h
#define RecoHGCal_TICL_MuonInterpretationAlgo_h

// Muon interpretation. A muon crosses HGCAL as a MIP: its candidate takes the track momentum.
// makeCandidates: a muon track with MIP-like energy in its (eta, phi) window takes the tracksters of the window.
// makeOpinions: a muon hypothesis for every muon track, with the tracksters near the propagated track as footprint.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoTICL/Interpretation/interface/TICLInterpretationAlgoBase.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <memory>
#include <string>

namespace ticl {

  class MuonInterpretationAlgo : public TICLInterpretationAlgoBase<reco::Track> {
  public:
    MuonInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC);
    ~MuonInterpretationAlgo() override;

    void makeCandidates(const Inputs &input,
                        edm::Handle<MtdHostCollection> inputTiming_h,
                        std::vector<Trackster> &resultTracksters,
                        std::vector<int> &resultCandidate,
                        std::vector<bool> &maskedTracksters,
                        std::vector<std::vector<unsigned int>> &linkedResultTracksters) override;

    // One muon hypothesis per muon track: the footprint is the tracksters near the track, the score falls with
    // their energy.
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
    // True when the energy around the track is MIP-like.
    bool isMuonLike(float nearbyEnergy, unsigned nNearbyTracksters) const;

    // (eta,phi) window used to collect tracksters around the track direction.
    const float delta_tk_ts_;
    // Max summed raw energy of the tracksters around the trajectory for a MIP-like
    // (muon) signature; above this the track points to a shower and is not a muon.
    const float mip_energy_max_;
    // Hypotheses: the tracksters in the delta_tk_ts window within this transverse distance [cm] of the track.
    const float max_distance_;

    const HGCalDDDConstants *hgcons_;
    ticlgeom::Tools rhtools_;
  };

}  // namespace ticl

#endif
