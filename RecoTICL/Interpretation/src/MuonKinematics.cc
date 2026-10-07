#include "RecoTICL/Interpretation/interface/MuonKinematics.h"

std::vector<int> ticl::muonOfTrack(const edm::Handle<std::vector<reco::Track>> &tracks,
                                   const reco::MuonCollection &muons) {
  std::vector<int> muonOf(tracks->size(), -1);
  // In reverse order, so that the first muon of a track is the last one written.
  for (int iMu = static_cast<int>(muons.size()) - 1; iMu >= 0; --iMu) {
    const auto &track = muons[iMu].track();
    if (track.isNonnull() && track.id() == tracks.id() && track.key() < muonOf.size())
      muonOf[track.key()] = iMu;
  }
  return muonOf;
}
