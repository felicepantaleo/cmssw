#ifndef RecoTICL_Interpretation_ChargedHadronInterpretationAlgo_h
#define RecoTICL_Interpretation_ChargedHadronInterpretationAlgo_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "RecoTICL/Interpretation/interface/TICLInterpretationAlgoBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

namespace ticl {

  class ChargedHadronInterpretationAlgo : public TICLInterpretationAlgoBase<reco::Track> {
  public:
    ChargedHadronInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC);

    ~ChargedHadronInterpretationAlgo() override;

    void makeCandidates(const Inputs &input,
                        edm::Handle<MtdHostCollection> inputTiming_h,
                        std::vector<Trackster> &resultTracksters,
                        std::vector<int> &resultCandidate,
                        std::vector<bool> &maskedTracksters) override;

    // Arbitration mode: one charged-hadron hypothesis per track-linked (merged)
    // trackster, reusing the geometric association of makeCandidates. Neutral
    // leftovers are not emitted; the producer derives neutrals from unclaimed
    // tracksters after arbitration.
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
    void buildLayers();

    Vector propagateTrackster(const Trackster &t,
                              const unsigned idx,
                              float zVal,
                              std::array<TICLLayerTile, 2> &tracksterTiles);

    void findTrackstersInWindow(const edm::MultiSpan<Trackster> &tracksters,
                                const std::vector<std::pair<Vector, unsigned>> &seedingCollection,
                                const std::array<TICLLayerTile, 2> &tracksterTiles,
                                const std::vector<Vector> &tracksterPropPoints,
                                float delta,
                                unsigned trackstersSize,
                                std::vector<std::vector<unsigned>> &resultCollection,
                                bool useMask);

    bool timeAndEnergyCompatible(float &total_raw_energy,
                                 const reco::Track &track,
                                 const Trackster &trackster,
                                 const float &tkTime,
                                 const float &tkTimeErr,
                                 const float &tkQual,
                                 const float &tkBeta,
                                 const GlobalPoint &tkMtdPos,
                                 bool useMTDTiming);

    const float tkEnergyCut_ = 2.0f;
    const float maxDeltaT_ = 3.0f;
    const float del_tk_ts_layer1_;
    const float del_tk_ts_int_;
    const float timing_quality_threshold_;
    // Track<->trackster energy-compatibility veto: a trackster is absorbed by a track
    // only while the cumulative reco raw energy stays below track.p() plus a slack of
    // min(fraction * E_trackster, max). Tight defaults were tuned at PU200; expose them
    // so the veto can be relaxed and re-tuned per pileup scenario.
    const double energy_overshoot_fraction_;
    const double energy_overshoot_max_;

    const HGCalDDDConstants *hgcons_;

    std::unique_ptr<GeomDet> firstDisk_[2];
    std::unique_ptr<GeomDet> interfaceDisk_[2];

    hgcal::RecHitTools rhtools_;

    edm::ESHandle<MagneticField> bfield_;
    edm::ESHandle<Propagator> propagator_;
  };

}  // namespace ticl

#endif
