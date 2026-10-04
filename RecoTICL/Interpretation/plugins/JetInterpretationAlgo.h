#ifndef RecoHGCal_TICL_JetInterpretationAlgo_h
#define RecoHGCal_TICL_JetInterpretationAlgo_h

// Jet and recovery opinions. A Jet hypothesis holds several tracks and the trackster (or the union of footprints) they
// point to, scored by the balance of E and the summed p. A recovery hypothesis holds one track and its footprint.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoTICL/Interpretation/interface/TICLInterpretationAlgoBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"

#include <array>
#include <memory>
#include <vector>

namespace ticl {

  class JetInterpretationAlgo : public TICLInterpretationAlgoBase<reco::Track> {
  public:
    JetInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC);
    ~JetInterpretationAlgo() override;

    // Opinion-only algorithm: makeCandidates does nothing.
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
    // (eta, phi, z) of a trackster barycenter.
    struct TracksterAxis {
      float eta, phi, z;
    };
    // Centre of the (eta, phi) window of a track: the HGCAL impact (float), or the outermost momentum (double).
    struct TrackWindow {
      bool atImpact = false;
      float etaImpact = 0.f, phiImpact = 0.f, zImpact = 0.f;
      double etaMomentum = 0., phiMomentum = 0., zMomentum = 0.;
    };

    // Windows, trackster axes and footprints of the current event.
    void fillEventCache(const Inputs &input);
    void makeRecoveryOpinions(const Inputs &input,
                              std::vector<Trackster> &hypothesisTracksters,
                              std::vector<Hypothesis> &hypotheses) const;
    void makeSharedFootprintOpinions(const Inputs &input,
                                     std::vector<Trackster> &hypothesisTracksters,
                                     std::vector<Hypothesis> &hypotheses) const;
    // Recovery footprint of a track: indices of the input tracksters and their summed raw energy.
    std::vector<unsigned> footprint(const Inputs &input, size_t iTrack, float &sumE) const;
    // Transverse distance [cm] from a track to a point: through the HGCAL impact when the propagation succeeded, else
    // along the straight line from the outermost state.
    float trackDistance(const Inputs &input, const reco::Track &tk, size_t iTrack, const Vector &point) const;
    // True when the trackster barycenter is in the delta_tk_ts window of the track.
    bool inTrackWindow(size_t iTrack, unsigned iTs) const;

    const float delta_tk_ts_;
    // Min trackster raw energy for a jet on one trackster.
    const float min_trackster_energy_;
    // E/p band of a recovery footprint.
    const float recovery_min_eop_;
    const float recovery_max_eop_;
    // Recovery footprint: the tracksters in the window within this transverse distance [cm] of the track, nearest
    // first, up to the track momentum.
    const float recovery_max_distance_;

    const HGCalDDDConstants *hgcons_;
    ticlgeom::Tools rhtools_;
    std::vector<TrackWindow> windows_;
    std::vector<TracksterAxis> tracksterAxes_;
    std::vector<std::vector<unsigned>> footprints_;
    std::vector<float> footprintEnergies_;
  };

}  // namespace ticl

#endif
