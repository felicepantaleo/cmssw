#include "RecoTICL/Interpretation/interface/EGammaInterpretationAlgo.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// v0 of the e/gamma interpretation, opinion-emitting (see the header). The track <->
// supercluster association uses the track outer momentum direction against the
// supercluster barycenter; the upgrade path is a full propagation to the HGCAL face,
// as in the general interpretation.

using namespace ticl;

EGammaInterpretationAlgo::EGammaInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : TICLInterpretationAlgoBase(conf, iC),
      delta_tk_sc_(conf.getParameter<double>("delta_tk_sc")),
      eop_min_(conf.getParameter<double>("eop_min")),
      eop_max_(conf.getParameter<double>("eop_max")),
      min_em_fraction_(conf.getParameter<double>("min_em_fraction")),
      min_supercluster_energy_(conf.getParameter<double>("min_supercluster_energy")),
      hgcons_(nullptr) {}

EGammaInterpretationAlgo::~EGammaInterpretationAlgo() {}

void EGammaInterpretationAlgo::initialize(const HGCalDDDConstants *hgcons,
                                          const hgcal::RecHitTools rhtools,
                                          const edm::ESHandle<MagneticField> bfieldH,
                                          const edm::ESHandle<Propagator> propH) {
  hgcons_ = hgcons;
  rhtools_ = rhtools;
  bfield_ = bfieldH;
  propagator_ = propH;
}

void EGammaInterpretationAlgo::makeCandidates(const Inputs & /*input*/,
                                              edm::Handle<MtdHostCollection> /*inputTiming_h*/,
                                              std::vector<Trackster> & /*resultTracksters*/,
                                              std::vector<int> & /*resultCandidate*/,
                                              std::vector<bool> & /*maskedTracksters*/) {
  // Opinion-only algorithm: the e/gamma interpretation participates through
  // makeOpinions + the producer arbitration, never through greedy masking.
}

void EGammaInterpretationAlgo::makeOpinions(const Inputs &input,
                                            edm::Handle<MtdHostCollection> /*inputTiming_h*/,
                                            std::vector<Trackster> &hypothesisTracksters,
                                            std::vector<Hypothesis> &hypotheses) {
  const auto &tracks = *input.tracksHandle;
  const auto &maskTracks = input.maskedTracks;
  const auto &superclusters = input.tracksters;

  // One hypothesis trackster per supercluster used by any hypothesis; appended lazily
  // and shared between the electron and photon opinions on the same supercluster.
  std::vector<int> hypoTracksterOf(superclusters.size(), -1);
  auto hypoTracksterFor = [&](unsigned scIdx) {
    if (hypoTracksterOf[scIdx] < 0) {
      hypoTracksterOf[scIdx] = static_cast<int>(hypothesisTracksters.size());
      hypothesisTracksters.push_back(superclusters[scIdx]);
    }
    return hypoTracksterOf[scIdx];
  };

  // Electron opinions: best supercluster within the (eta,phi) window of each track.
  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    if (!maskTracks[iTrack])
      continue;
    const auto &tk = tracks[iTrack];
    const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
    int best = -1;
    double bestDR = delta_tk_sc_;
    for (unsigned iSc = 0; iSc < superclusters.size(); ++iSc) {
      const auto &sc = superclusters[iSc];
      if (sc.raw_energy() < min_supercluster_energy_)
        continue;
      const auto &bary = sc.barycenter();
      if (bary.eta() * dir.eta() < 0.)  // same endcap
        continue;
      const double dR = reco::deltaR(bary.eta(), bary.phi(), dir.eta(), dir.phi());
      if (dR < bestDR) {
        bestDR = dR;
        best = static_cast<int>(iSc);
      }
    }
    if (best >= 0) {
      // Electron identity: E/p compatibility and an EM-like supercluster. A charged
      // hadron's EM subshower has E/p well below 1 and fails here, so the
      // charged-hadron hypothesis keeps the track.
      const auto &sc = superclusters[best];
      const double eop = tk.p() > 0. ? sc.raw_energy() / tk.p() : 0.;
      const double emFraction = sc.raw_energy() > 0.f ? sc.raw_em_energy() / sc.raw_energy() : 0.;
      if (eop < eop_min_ || eop > eop_max_ || emFraction < min_em_fraction_)
        continue;
      Hypothesis h;
      h.type = Hypothesis::Type::Electron;
      h.score = static_cast<float>(1. - bestDR / delta_tk_sc_);
      h.trackIdx = static_cast<int>(iTrack);
      h.tracksterIdx = hypoTracksterFor(best);
      hypotheses.push_back(h);
    }
  }

  // Photon opinions: every EM-like supercluster, including track-matched ones. The
  // arbiter prefers the electron hypothesis (type priority); the photon one on the
  // same supercluster then fails the energy-exclusivity check and is dropped, or
  // takes over if the electron loses.
  for (unsigned iSc = 0; iSc < superclusters.size(); ++iSc) {
    const auto &sc = superclusters[iSc];
    if (sc.raw_energy() < min_supercluster_energy_)
      continue;
    const double emFraction = sc.raw_energy() > 0.f ? sc.raw_em_energy() / sc.raw_energy() : 0.;
    if (emFraction < min_em_fraction_)
      continue;
    Hypothesis h;
    h.type = Hypothesis::Type::Photon;
    h.score = static_cast<float>(std::min(emFraction, 1.));
    h.tracksterIdx = hypoTracksterFor(iSc);
    hypotheses.push_back(h);
  }
}

void EGammaInterpretationAlgo::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<double>("delta_tk_sc", 0.1)->setComment("(eta,phi) window for the track <-> supercluster match.");
  desc.add<double>("eop_min", 0.5)->setComment("Min supercluster E / track p for an electron hypothesis.");
  desc.add<double>("eop_max", 1.5)->setComment("Max supercluster E / track p for an electron hypothesis.");
  desc.add<double>("min_em_fraction", 0.8)
      ->setComment("Min EM energy fraction for a supercluster to yield a photon hypothesis.");
  desc.add<double>("min_supercluster_energy", 1.0)->setComment("Min supercluster raw energy [GeV].");
  TICLInterpretationAlgoBase::fillPSetDescription(desc);
}
