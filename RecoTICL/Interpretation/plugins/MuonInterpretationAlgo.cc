#include "MuonInterpretationAlgo.h"

#include <algorithm>

#include "DataFormats/Math/interface/deltaR.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"
#include "RecoTICL/Interpretation/interface/TrackStraightLine.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

// Muon interpretation: a track whose trajectory in HGCAL meets only MIP-like energy is a muon; it takes the
// tracksters along the trajectory.

using namespace ticl;

MuonInterpretationAlgo::MuonInterpretationAlgo(const edm::ParameterSet &conf, edm::ConsumesCollector iC)
    : TICLInterpretationAlgoBase(conf, iC),
      delta_tk_ts_(conf.getParameter<float>("delta_tk_ts")),
      mip_energy_max_(conf.getParameter<float>("mip_energy_max")),
      max_distance_(conf.getParameter<float>("max_distance")),
      hgcons_(nullptr) {
  if (!(max_distance_ > 0.f))
    throw cms::Exception("Configuration") << "MuonInterpretationAlgo: max_distance must be positive";
}

MuonInterpretationAlgo::~MuonInterpretationAlgo() {}

void MuonInterpretationAlgo::initialize(const HGCalDDDConstants *hgcons,
                                        const ticlgeom::Tools rhtools,
                                        const edm::ESHandle<MagneticField> /*bfieldH*/,
                                        const edm::ESHandle<Propagator> /*propH*/) {
  hgcons_ = hgcons;
  rhtools_ = rhtools;
}

bool MuonInterpretationAlgo::isMuonLike(float nearbyEnergy, unsigned /*nNearbyTracksters*/) const {
  // A muon deposits a MIP: the tracksters around its trajectory carry little energy.
  return nearbyEnergy < mip_energy_max_;
}

void MuonInterpretationAlgo::makeCandidates(const Inputs &input,
                                            edm::Handle<MtdHostCollection> /*inputTiming_h*/,
                                            std::vector<Trackster> &resultTracksters,
                                            std::vector<int> &resultCandidate,
                                            std::vector<bool> &maskedTracksters,
                                            std::vector<std::vector<unsigned int>> &linkedResultTracksters) {
  const auto &tracks = *input.tracksHandle;
  const auto &maskTracks = input.maskedTracks;
  const auto &tracksters = input.tracksters;
  if (maskedTracksters.size() < tracksters.size())
    maskedTracksters.resize(tracksters.size(), false);

  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    if (!maskTracks[iTrack])
      continue;
    const auto &tk = tracks[iTrack];
    // The track enters HGCAL along its outermost momentum direction.
    const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
    const float tkEta = dir.eta();
    const float tkPhi = dir.phi();

    // Collect the tracksters whose barycenter lies within the (eta,phi) window around
    // the trajectory, and sum their raw energy (the "is it energetic?" measure).
    std::vector<unsigned> nearby;
    float nearbyEnergy = 0.f;
    for (unsigned iTs = 0; iTs < tracksters.size(); ++iTs) {
      if (maskedTracksters[iTs])
        continue;
      const auto &bary = tracksters[iTs].barycenter();
      if (bary.eta() * tkEta < 0.f)  // same endcap
        continue;
      if (reco::deltaR(bary.eta(), bary.phi(), tkEta, tkPhi) < delta_tk_ts_) {
        nearby.push_back(iTs);
        nearbyEnergy += tracksters[iTs].raw_energy();
      }
    }

    if (!isMuonLike(nearbyEnergy, nearby.size())) {
      // Trajectory points to a shower: this is not a muon. Flag it so the producer
      // routes it to the general interpretation instead of building a muon candidate.
      resultCandidate[iTrack] = kMuonRejected;
      continue;
    }

    // Muon: consume the MIP tracksters (mask them) and merge them into one trackster so
    // the producer can attach it to the muon candidate; the candidate energy itself is
    // taken from the track momentum by the producer.
    if (!nearby.empty()) {
      Trackster muonTrackster;
      for (unsigned iTs : nearby) {
        muonTrackster.mergeTracksters(tracksters[iTs]);
        maskedTracksters[iTs] = true;
      }
      resultCandidate[iTrack] = static_cast<int>(resultTracksters.size());
      resultTracksters.push_back(muonTrackster);
      linkedResultTracksters.push_back(std::move(nearby));
    } else {
      resultCandidate[iTrack] = -1;  // muon with no HGCAL deposit: track-only candidate
    }
  }
}

void MuonInterpretationAlgo::makeOpinions(const Inputs &input,
                                          edm::Handle<MtdHostCollection> /*inputTiming_h*/,
                                          std::vector<Trackster> &hypothesisTracksters,
                                          std::vector<Hypothesis> &hypotheses) {
  const auto &tracks = *input.tracksHandle;
  const auto &maskTracks = input.maskedTracks;
  const auto &tracksters = input.tracksters;
  if (std::none_of(maskTracks.begin(), maskTracks.end(), [](bool b) { return b; }))
    return;
  if (input.impacts == nullptr)
    throw cms::Exception("Configuration") << "MuonInterpretationAlgo: the track impacts are required";
  // (eta, phi) of the trackster barycenters.
  std::vector<float> tsEta(tracksters.size()), tsPhi(tracksters.size());
  for (unsigned iTs = 0; iTs < tracksters.size(); ++iTs) {
    tsEta[iTs] = tracksters[iTs].barycenter().eta();
    tsPhi[iTs] = tracksters[iTs].barycenter().phi();
  }

  for (size_t iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    if (!maskTracks[iTrack])
      continue;
    const auto &tk = tracks[iTrack];
    const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
    const float tkEta = dir.eta();
    const float tkPhi = dir.phi();
    // Distance to the propagated track; to the straight line from the outermost state when the propagation failed.
    const auto &impact = (*input.impacts)[iTrack];
    auto distance = [&](const Vector &point) {
      return impact.valid ? impactTransverseDistance(impact, point) : straightLineTransverseDistance(tk, point);
    };

    std::vector<unsigned> nearby;
    float nearbyEnergy = 0.f;
    for (unsigned iTs = 0; iTs < tracksters.size(); ++iTs) {
      if (tsEta[iTs] * tkEta < 0.f)  // same endcap
        continue;
      if (!(reco::deltaR(tsEta[iTs], tsPhi[iTs], tkEta, tkPhi) < delta_tk_ts_))
        continue;
      if (!(distance(tracksters[iTs].barycenter()) < max_distance_))
        continue;
      nearby.push_back(iTs);
      nearbyEnergy += tracksters[iTs].raw_energy();
    }

    Hypothesis h;
    h.type = Hypothesis::Type::Muon;
    h.trackIdx = static_cast<int>(iTrack);
    h.score = static_cast<float>(std::max(0.f, 1.f - nearbyEnergy / mip_energy_max_));
    if (!nearby.empty()) {
      Trackster muonTrackster;
      muonTrackster.mergeTracksters(tracksters, nearby);
      h.tracksterIdx = static_cast<int>(hypothesisTracksters.size());
      hypothesisTracksters.push_back(std::move(muonTrackster));
    }
    hypotheses.push_back(std::move(h));
  }
}

void MuonInterpretationAlgo::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<float>("delta_tk_ts", 0.1f)->setComment("(eta,phi) window to collect tracksters around the trajectory.");
  desc.add<float>("mip_energy_max", 10.0f)
      ->setComment("Max summed raw energy [GeV] around the trajectory for a MIP-like (muon) signature.");
  desc.add<float>("max_distance", 3.f)
      ->setComment("Hypotheses: max transverse distance [cm] between a trackster barycenter and the track.");
  TICLInterpretationAlgoBase::fillPSetDescription(desc);
}
