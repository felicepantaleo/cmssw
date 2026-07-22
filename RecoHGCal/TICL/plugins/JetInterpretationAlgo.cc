#include "RecoHGCal/TICL/plugins/JetInterpretationAlgo.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// v0 of the jet (multi-particle) interpretation, opinion-emitting (see the header).
// The track <-> trackster association uses the track outer momentum direction against
// the trackster barycenter; the upgrade path is a full propagation to the HGCAL face.

using namespace ticl;

JetInterpretationAlgo::JetInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : TICLInterpretationAlgoBase(conf, iC),
      delta_tk_ts_(conf.getParameter<double>("delta_tk_ts")),
      min_trackster_energy_(conf.getParameter<double>("min_trackster_energy")),
      recovery_min_eop_(conf.getParameter<double>("recovery_min_eop")),
      recovery_max_eop_(conf.getParameter<double>("recovery_max_eop")),
      hgcons_(nullptr) {}

JetInterpretationAlgo::~JetInterpretationAlgo() {}

void JetInterpretationAlgo::initialize(const HGCalDDDConstants *hgcons,
                                       const hgcal::RecHitTools rhtools,
                                       const edm::ESHandle<MagneticField> bfieldH,
                                       const edm::ESHandle<Propagator> propH) {
  hgcons_ = hgcons;
  rhtools_ = rhtools;
  bfield_ = bfieldH;
  propagator_ = propH;
}

void JetInterpretationAlgo::makeCandidates(const Inputs & /*input*/,
                                           edm::Handle<MtdHostCollection> /*inputTiming_h*/,
                                           std::vector<Trackster> & /*resultTracksters*/,
                                           std::vector<int> & /*resultCandidate*/,
                                           std::vector<bool> & /*maskedTracksters*/) {
  // Opinion-only algorithm: the jet interpretation participates through
  // makeOpinions + the producer arbitration, never through greedy masking.
}

void JetInterpretationAlgo::makeOpinions(const Inputs &input,
                                         edm::Handle<MtdHostCollection> /*inputTiming_h*/,
                                         std::vector<Trackster> &hypothesisTracksters,
                                         std::vector<Hypothesis> &hypotheses) {
  const auto &tracks = *input.tracksHandle;
  const auto &maskTracks = input.maskedTracks;
  const auto &tracksters = input.tracksters;

  for (unsigned iTs = 0; iTs < tracksters.size(); ++iTs) {
    const auto &ts = tracksters[iTs];
    if (ts.raw_energy() < min_trackster_energy_)
      continue;
    const auto &bary = ts.barycenter();

    // All selected tracks pointing into this trackster.
    std::vector<int> inTracks;
    double sumP = 0.;
    for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
      if (!maskTracks[iTrack])
        continue;
      const auto &tk = tracks[iTrack];
      const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
      if (bary.eta() * dir.eta() < 0.)  // same endcap
        continue;
      if (reco::deltaR(bary.eta(), bary.phi(), dir.eta(), dir.phi()) < delta_tk_ts_) {
        inTracks.push_back(static_cast<int>(iTrack));
        sumP += tk.p();
      }
    }
    // A jet reading needs at least two tracks; single tracks belong to the
    // charged-hadron (or electron / muon) hypotheses.
    if (inTracks.size() < 2)
      continue;

    // Score: how well the summed track momenta balance the calorimetric energy.
    const double e = ts.raw_energy();
    const double balance = 1. - std::abs(e - sumP) / std::max(e, sumP);
    if (balance <= 0.)
      continue;

    Hypothesis h;
    h.type = Hypothesis::Type::Jet;
    h.score = static_cast<float>(balance);
    h.trackIdxs = inTracks;
    h.tracksterIdx = static_cast<int>(hypothesisTracksters.size());
    hypothesisTracksters.push_back(ts);
    hypotheses.push_back(h);
  }

  makeRecoveryOpinions(input, hypothesisTracksters, hypotheses);
}

void JetInterpretationAlgo::makeRecoveryOpinions(const Inputs &input,
                                                 std::vector<Trackster> &hypothesisTracksters,
                                                 std::vector<Hypothesis> &hypotheses) {
  // Single-track claim-and-attach recovery: a track the tight charged-hadron gates
  // rejected, pointing at unclaimed calorimetric energy comparable to its momentum
  // (the energy currently becomes neutral candidates while the track momentum is
  // lost). Merge the nearby tracksters and emit a low-tier hypothesis; the
  // arbitration's exclusivity rejects it wherever a tight winner already claimed the
  // track or the energy, and the assembly's residual mechanism handles any excess.
  const auto &tracks = *input.tracksHandle;
  const auto &maskTracks = input.maskedTracks;
  const auto &tracksters = input.tracksters;

  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    if (!maskTracks[iTrack])
      continue;
    const auto &tk = tracks[iTrack];
    const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
    std::vector<unsigned> nearby;
    double sumE = 0.;
    for (unsigned iTs = 0; iTs < tracksters.size(); ++iTs) {
      const auto &bary = tracksters[iTs].barycenter();
      if (bary.eta() * dir.eta() < 0.)
        continue;
      if (reco::deltaR(bary.eta(), bary.phi(), dir.eta(), dir.phi()) < delta_tk_ts_) {
        nearby.push_back(iTs);
        sumE += tracksters[iTs].raw_energy();
      }
    }
    if (nearby.empty())
      continue;
    const double p = tk.p();
    if (p <= 0. || sumE < recovery_min_eop_ * p || sumE > recovery_max_eop_ * p)
      continue;
    Hypothesis h;
    h.type = Hypothesis::Type::RecoveryChargedHadron;
    h.score = static_cast<float>(1. - std::abs(sumE - p) / std::max(sumE, p));
    h.trackIdx = static_cast<int>(iTrack);
    Trackster merged;
    for (unsigned iTs : nearby)
      merged.mergeTracksters(tracksters[iTs]);
    h.tracksterIdx = static_cast<int>(hypothesisTracksters.size());
    hypothesisTracksters.push_back(merged);
    hypotheses.push_back(h);
  }
}

void JetInterpretationAlgo::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<double>("delta_tk_ts", 0.1)->setComment("(eta,phi) window to associate tracks to a trackster.");
  desc.add<double>("min_trackster_energy", 5.0)
      ->setComment("Min trackster raw energy [GeV] for a jet (multi-track) reading.");
  desc.add<double>("recovery_min_eop", 0.2)->setComment("Min nearby E / track p for a single-track recovery.");
  desc.add<double>("recovery_max_eop", 3.0)->setComment("Max nearby E / track p for a single-track recovery.");
  TICLInterpretationAlgoBase::fillPSetDescription(desc);
}
