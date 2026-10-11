#ifndef RecoTICL_Interpretation_AssignmentMaps_h
#define RecoTICL_Interpretation_AssignmentMaps_h

#include <algorithm>
#include <vector>

namespace ticl {

  // Values of the trackMode map of TICLInterpretationProducer: what the accepted hypothesis of each track is.
  enum class TrackMode : int {
    kNotSelected = -1,
    kMuon = 1,
    kChargedHadron = 2,
    kElectron = 3,
    kJetMember = 4,
    kRecovery = 5
  };

  // Indices, sorted and unique, of the final tracksters that the interpretation types as electrons (the trackster of a
  // track in mode kElectron) or as neutrals with a pdg id in neutralPdgs.
  inline std::vector<int> eGammaTracksterIndices(const std::vector<int> &trackToTrackster,
                                                 const std::vector<int> &trackMode,
                                                 const std::vector<int> &neutralIdx,
                                                 const std::vector<int> &neutralPdg,
                                                 const std::vector<int> &neutralPdgs) {
    std::vector<int> selected;
    for (std::size_t iTrack = 0; iTrack < trackMode.size(); ++iTrack)
      if (trackMode[iTrack] == static_cast<int>(TrackMode::kElectron) && trackToTrackster[iTrack] >= 0)
        selected.push_back(trackToTrackster[iTrack]);
    for (std::size_t k = 0; k < neutralIdx.size(); ++k)
      if (std::find(neutralPdgs.begin(), neutralPdgs.end(), neutralPdg[k]) != neutralPdgs.end())
        selected.push_back(neutralIdx[k]);
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    return selected;
  }

}  // namespace ticl

#endif
