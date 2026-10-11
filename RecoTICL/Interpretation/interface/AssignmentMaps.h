#ifndef RecoTICL_Interpretation_AssignmentMaps_h
#define RecoTICL_Interpretation_AssignmentMaps_h

#include <algorithm>
#include <cstddef>
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

  // Electron information of the e/gamma tracksters, where selected[k] is the final trackster of e/gamma trackster k:
  // the track in mode kElectron on it (-1: none), and whether the Electron hypothesis of a track has its footprint on
  // it (electronTrackster, the map of TICLInterpretationProducer).
  struct EGammaElectrons {
    std::vector<int> electronTrack;
    std::vector<bool> electronHypothesis;
  };

  inline EGammaElectrons eGammaElectrons(const std::vector<int> &selected,
                                         const std::vector<int> &trackToTrackster,
                                         const std::vector<int> &trackMode,
                                         const std::vector<int> &electronTrackster) {
    EGammaElectrons result{std::vector<int>(selected.size(), -1), std::vector<bool>(selected.size(), false)};
    const int nFinal = selected.empty() ? 0 : *std::max_element(selected.begin(), selected.end()) + 1;
    std::vector<int> position(nFinal, -1);
    for (std::size_t k = 0; k < selected.size(); ++k)
      position[selected[k]] = static_cast<int>(k);
    auto eGammaIndex = [&](int iFinal) { return iFinal >= 0 && iFinal < nFinal ? position[iFinal] : -1; };
    for (std::size_t iTrack = 0; iTrack < trackMode.size(); ++iTrack) {
      if (const int k = eGammaIndex(trackToTrackster[iTrack]);
          k >= 0 && trackMode[iTrack] == static_cast<int>(TrackMode::kElectron))
        result.electronTrack[k] = static_cast<int>(iTrack);
      if (const int k = eGammaIndex(electronTrackster[iTrack]); k >= 0)
        result.electronHypothesis[k] = true;
    }
    return result;
  }

  // Seeds of an HGCAL supercluster whose PF clusters come from the e/gamma tracksters clusterTracksters: a seed from the
  // KF track of each trackster typed electron (tracks, sorted and unique), and seeds from the supercluster itself
  // (fromSuperCluster) when no trackster is typed electron and the Electron hypothesis of a track is on one of them.
  // With superClusterSeedsForElectrons, a supercluster with a trackster typed electron also gets them.
  struct SuperClusterSeeding {
    std::vector<int> tracks;
    bool fromSuperCluster = false;
  };

  inline SuperClusterSeeding superClusterSeeding(const std::vector<int> &clusterTracksters,
                                                 const EGammaElectrons &electrons,
                                                 bool superClusterSeedsForElectrons) {
    SuperClusterSeeding seeding;
    bool electronHypothesis = false;
    for (int k : clusterTracksters) {
      if (electrons.electronTrack[k] >= 0)
        seeding.tracks.push_back(electrons.electronTrack[k]);
      electronHypothesis = electronHypothesis || electrons.electronHypothesis[k];
    }
    std::sort(seeding.tracks.begin(), seeding.tracks.end());
    seeding.tracks.erase(std::unique(seeding.tracks.begin(), seeding.tracks.end()), seeding.tracks.end());
    seeding.fromSuperCluster = seeding.tracks.empty() ? electronHypothesis : superClusterSeedsForElectrons;
    return seeding;
  }

}  // namespace ticl

#endif
