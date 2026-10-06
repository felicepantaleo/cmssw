#ifndef RecoTICL_Interpretation_MuonKinematics_h
#define RecoTICL_Interpretation_MuonKinematics_h

#include "DataFormats/MuonReco/interface/Muon.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

namespace ticl {

  // True when a charged candidate takes the kinematics of the muon of its track: a PF muon without the tracker-muon
  // flag, or a global muon when the candidate has no trackster.
  inline bool takesMuonKinematics(const reco::MuonRef &muon, bool hasTracksters) {
    return (PFMuonAlgo::isMuon(muon) && !muon->isTrackerMuon()) || (!hasTracksters && muon->isGlobalMuon());
  }

}  // namespace ticl

#endif
