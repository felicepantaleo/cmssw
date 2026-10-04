#include "RecoTICL/Interpretation/plugins/JetInterpretationAlgo.h"

#include <algorithm>
#include <cmath>

#include "DataFormats/Math/interface/deltaR.h"
#include "RecoTICL/Interpretation/interface/TrackStraightLine.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace ticl;

JetInterpretationAlgo::JetInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : TICLInterpretationAlgoBase(conf, iC),
      delta_tk_ts_(conf.getParameter<float>("delta_tk_ts")),
      min_trackster_energy_(conf.getParameter<float>("min_trackster_energy")),
      recovery_min_eop_(conf.getParameter<float>("recovery_min_eop")),
      recovery_max_eop_(conf.getParameter<float>("recovery_max_eop")),
      recovery_max_distance_(conf.getParameter<float>("recovery_max_distance")),
      hgcons_(nullptr) {}

JetInterpretationAlgo::~JetInterpretationAlgo() {}

void JetInterpretationAlgo::initialize(const HGCalDDDConstants *hgcons,
                                       const ticlgeom::Tools rhtools,
                                       const edm::ESHandle<MagneticField> /*bfieldH*/,
                                       const edm::ESHandle<Propagator> /*propH*/) {
  hgcons_ = hgcons;
  rhtools_ = rhtools;
}

float JetInterpretationAlgo::trackDistance(const Inputs &input,
                                           const reco::Track &tk,
                                           size_t iTrack,
                                           const Vector &point) const {
  const auto &impact = (*input.impacts)[iTrack];
  if (impact.valid)
    return impactTransverseDistance(impact, point);
  return straightLineTransverseDistance(tk, point);
}

bool JetInterpretationAlgo::inTrackWindow(size_t iTrack, unsigned iTs) const {
  const auto &w = windows_[iTrack];
  const auto &t = tracksterAxes_[iTs];
  if (w.atImpact)
    return t.z * w.zImpact > 0.f && reco::deltaR(t.eta, t.phi, w.etaImpact, w.phiImpact) < delta_tk_ts_;
  return t.z * w.zMomentum > 0.f && reco::deltaR(t.eta, t.phi, w.etaMomentum, w.phiMomentum) < delta_tk_ts_;
}

void JetInterpretationAlgo::fillEventCache(const Inputs &input) {
  const auto &tracks = *input.tracksHandle;
  const auto &maskTracks = input.maskedTracks;
  const auto &tracksters = input.tracksters;
  if (input.impacts == nullptr)
    throw cms::Exception("Configuration") << "JetInterpretationAlgo: the track impacts are required";
  windows_.assign(tracks.size(), TrackWindow());
  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    if (!maskTracks[iTrack])
      continue;
    const auto &tk = tracks[iTrack];
    auto &w = windows_[iTrack];
    if ((*input.impacts)[iTrack].valid) {
      const auto &pos = (*input.impacts)[iTrack].position;
      w.atImpact = true;
      w.etaImpact = pos.eta();
      w.phiImpact = pos.barePhi();
      w.zImpact = pos.z();
    } else {
      const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
      w.etaMomentum = dir.eta();
      w.phiMomentum = dir.phi();
      w.zMomentum = dir.z();
    }
  }
  tracksterAxes_.resize(tracksters.size());
  for (unsigned iTs = 0; iTs < tracksters.size(); ++iTs) {
    const auto &bary = tracksters[iTs].barycenter();
    tracksterAxes_[iTs] = {bary.eta(), bary.phi(), bary.z()};
  }
  footprints_.assign(tracks.size(), {});
  footprintEnergies_.assign(tracks.size(), 0.f);
  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack)
    if (maskTracks[iTrack])
      footprints_[iTrack] = footprint(input, iTrack, footprintEnergies_[iTrack]);
}

void JetInterpretationAlgo::makeCandidates(const Inputs & /*input*/,
                                           edm::Handle<MtdHostCollection> /*inputTiming_h*/,
                                           std::vector<Trackster> & /*resultTracksters*/,
                                           std::vector<int> & /*resultCandidate*/,
                                           std::vector<bool> & /*maskedTracksters*/,
                                           std::vector<std::vector<unsigned int>> & /*linkedResultTracksters*/) {}

void JetInterpretationAlgo::makeOpinions(const Inputs &input,
                                         edm::Handle<MtdHostCollection> /*inputTiming_h*/,
                                         std::vector<Trackster> &hypothesisTracksters,
                                         std::vector<Hypothesis> &hypotheses) {
  const auto &tracks = *input.tracksHandle;
  const auto &maskTracks = input.maskedTracks;
  const auto &tracksters = input.tracksters;
  fillEventCache(input);

  // One jet per trackster with at least two tracks in its window, scored by the balance of E and the summed p.
  for (unsigned iTs = 0; iTs < tracksters.size(); ++iTs) {
    const auto &ts = tracksters[iTs];
    if (ts.raw_energy() < min_trackster_energy_)
      continue;
    std::vector<int> inTracks;
    float sumP = 0.f;
    for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
      if (maskTracks[iTrack] && inTrackWindow(iTrack, iTs)) {
        inTracks.push_back(static_cast<int>(iTrack));
        sumP += tracks[iTrack].p();
      }
    }
    if (inTracks.size() < 2)
      continue;
    const float e = ts.raw_energy();
    const float balance = 1.f - std::abs(e - sumP) / std::max(e, sumP);
    if (balance <= 0.f)
      continue;
    Hypothesis h;
    h.type = Hypothesis::Type::Jet;
    h.score = balance;
    h.trackIdxs = std::move(inTracks);
    h.tracksterIdx = static_cast<int>(hypothesisTracksters.size());
    hypothesisTracksters.push_back(ts);
    hypotheses.push_back(std::move(h));
  }

  makeSharedFootprintOpinions(input, hypothesisTracksters, hypotheses);
  makeRecoveryOpinions(input, hypothesisTracksters, hypotheses);
}

void JetInterpretationAlgo::makeRecoveryOpinions(const Inputs &input,
                                                 std::vector<Trackster> &hypothesisTracksters,
                                                 std::vector<Hypothesis> &hypotheses) const {
  // A track with a footprint of E/p in [recovery_min_eop, recovery_max_eop] gives a charged hadron on that footprint.
  const auto &tracks = *input.tracksHandle;
  const auto &maskTracks = input.maskedTracks;
  const auto &tracksters = input.tracksters;
  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    const auto &nearby = footprints_[iTrack];
    if (!maskTracks[iTrack] || nearby.empty())
      continue;
    const float p = tracks[iTrack].p();
    const float sumE = footprintEnergies_[iTrack];
    if (p <= 0.f || sumE < recovery_min_eop_ * p || sumE > recovery_max_eop_ * p)
      continue;
    Hypothesis h;
    h.type = Hypothesis::Type::RecoveryChargedHadron;
    h.score = 1.f - std::abs(sumE - p) / std::max(sumE, p);
    h.trackIdx = static_cast<int>(iTrack);
    Trackster merged;
    for (unsigned iTs : nearby)
      merged.mergeTracksters(tracksters[iTs]);
    h.tracksterIdx = static_cast<int>(hypothesisTracksters.size());
    hypothesisTracksters.push_back(std::move(merged));
    hypotheses.push_back(std::move(h));
  }
}

std::vector<unsigned> JetInterpretationAlgo::footprint(const Inputs &input, size_t iTrack, float &sumE) const {
  const auto &tk = (*input.tracksHandle)[iTrack];
  const auto &tracksters = input.tracksters;
  const float p = tk.p();
  std::vector<unsigned> nearby;
  sumE = 0.f;
  // Nearest first. Skip a trackster that takes the sum above recovery_max_eop * p. Stop at p.
  std::vector<std::pair<float, unsigned>> byDistance;
  for (unsigned iTs = 0; iTs < tracksters.size(); ++iTs) {
    if (!inTrackWindow(iTrack, iTs))
      continue;
    const float dist = trackDistance(input, tk, iTrack, tracksters[iTs].barycenter());
    if (dist < recovery_max_distance_)
      byDistance.emplace_back(dist, iTs);
  }
  std::sort(byDistance.begin(), byDistance.end());
  for (auto const &[dist, iTs] : byDistance) {
    if (sumE >= p)
      break;
    const float e = tracksters[iTs].raw_energy();
    if (sumE + e > recovery_max_eop_ * p)
      continue;
    nearby.push_back(iTs);
    sumE += e;
  }
  return nearby;
}

void JetInterpretationAlgo::makeSharedFootprintOpinions(const Inputs &input,
                                                        std::vector<Trackster> &hypothesisTracksters,
                                                        std::vector<Hypothesis> &hypotheses) const {
  // Tracks whose footprints share a trackster form one group. A group with at least two tracks gives one Jet
  // hypothesis with all its tracks and the union of their footprints.
  const auto &tracks = *input.tracksHandle;
  const auto &tracksters = input.tracksters;

  std::vector<int> parent(tracks.size());
  for (size_t i = 0; i < parent.size(); ++i)
    parent[i] = static_cast<int>(i);
  auto find = [&parent](int i) {
    while (parent[i] != i)
      i = parent[i] = parent[parent[i]];
    return i;
  };
  std::vector<int> ownerOf(tracksters.size(), -1);
  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    for (unsigned iTs : footprints_[iTrack]) {
      if (ownerOf[iTs] < 0)
        ownerOf[iTs] = static_cast<int>(iTrack);
      else
        parent[find(static_cast<int>(iTrack))] = find(ownerOf[iTs]);
    }
  }

  std::vector<std::vector<int>> groupTracks(tracks.size());
  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack)
    if (!footprints_[iTrack].empty())
      groupTracks[find(static_cast<int>(iTrack))].push_back(static_cast<int>(iTrack));
  std::vector<bool> inGroup(tracksters.size(), false);
  for (auto const &members : groupTracks) {
    if (members.size() < 2)
      continue;
    float sumP = 0.f;
    for (int iTk : members)
      sumP += tracks[iTk].p();
    std::vector<unsigned> groupTracksters;
    for (int iTk : members)
      for (unsigned iTs : footprints_[iTk])
        if (!inGroup[iTs]) {
          inGroup[iTs] = true;
          groupTracksters.push_back(iTs);
        }
    Trackster merged;
    float e = 0.f;
    for (unsigned iTs : groupTracksters) {
      merged.mergeTracksters(tracksters[iTs]);
      e += tracksters[iTs].raw_energy();
      inGroup[iTs] = false;
    }
    const float balance = 1.f - std::abs(e - sumP) / std::max(e, sumP);
    if (balance <= 0.f)
      continue;
    Hypothesis h;
    h.type = Hypothesis::Type::Jet;
    h.score = balance;
    h.trackIdxs = members;
    h.tracksterIdx = static_cast<int>(hypothesisTracksters.size());
    hypothesisTracksters.push_back(std::move(merged));
    hypotheses.push_back(std::move(h));
  }
}

void JetInterpretationAlgo::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<float>("delta_tk_ts", 0.1f)->setComment("(eta,phi) window to associate tracks to a trackster.");
  desc.add<float>("min_trackster_energy", 5.0f)
      ->setComment("Min trackster raw energy [GeV] for a jet (multi-track) reading.");
  desc.add<float>("recovery_min_eop", 0.2f)->setComment("Min nearby E / track p for a single-track recovery.");
  desc.add<float>("recovery_max_eop", 1.5f)->setComment("Max nearby E / track p for a single-track recovery.");
  desc.add<float>("recovery_max_distance", 5.f)
      ->setComment(
          "Recovery footprint: the tracksters in the window within this transverse distance [cm] of the track, nearest "
          "first, up to the track momentum.");
  TICLInterpretationAlgoBase::fillPSetDescription(desc);
}
