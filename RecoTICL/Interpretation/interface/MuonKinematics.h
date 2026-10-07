#ifndef RecoTICL_Interpretation_MuonKinematics_h
#define RecoTICL_Interpretation_MuonKinematics_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

#include <vector>

namespace ticl {

  // True when a charged candidate takes the kinematics of the muon of its track: a muon candidate, a PF muon without
  // the tracker-muon flag, or a global muon when the candidate has no trackster.
  inline bool takesMuonKinematics(const reco::MuonRef &muon, bool muonCandidate, bool hasTracksters) {
    return muonCandidate || (PFMuonAlgo::isMuon(muon) && !muon->isTrackerMuon()) ||
           (!hasTracksters && muon->isGlobalMuon());
  }

  // For each track, the index of the first muon whose track it is, as PFMuonAlgo::muAssocToTrack finds it; -1 for none.
  std::vector<int> muonOfTrack(const edm::Handle<std::vector<reco::Track>> &tracks, const reco::MuonCollection &muons);

}  // namespace ticl

#endif
