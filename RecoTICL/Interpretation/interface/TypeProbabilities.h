#ifndef RecoTICL_Interpretation_TypeProbabilities_h
#define RecoTICL_Interpretation_TypeProbabilities_h

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "RecoTICL/Interpretation/interface/TICLInterpretationAlgoBase.h"

namespace ticl {

  // Track types of the per-track probabilities: an electron, a muon or a charged hadron (ChargedHadron, Recovery or Jet
  // hypothesis).
  enum class TrackType : unsigned int { kElectron = 0, kMuon = 1, kChargedHadron = 2 };
  constexpr unsigned int kTrackTypes = 3;

  // For each track, kTrackTypes values: the probability of the best hypothesis of each type that holds the track, the
  // logistic of the weight that the solver compares (0 when the track has no hypothesis of the type). A track-only
  // hypothesis has no model weight and does not count.
  inline std::vector<float> trackTypeProbabilities(const std::vector<Hypothesis> &hypotheses,
                                                   const std::vector<float> &weights,
                                                   std::size_t nTracks) {
    std::vector<float> p(kTrackTypes * nTracks, 0.f);
    auto keep = [&](int iTrack, TrackType type, float w) {
      if (iTrack < 0 || static_cast<std::size_t>(iTrack) >= nTracks)
        return;
      float &q = p[kTrackTypes * iTrack + static_cast<unsigned int>(type)];
      q = std::max(q, 1.f / (1.f + std::exp(-w)));
    };
    for (std::size_t i = 0; i < hypotheses.size(); ++i) {
      const auto &h = hypotheses[i];
      switch (h.type) {
        case Hypothesis::Type::Muon:
          keep(h.trackIdx, TrackType::kMuon, weights[i]);
          break;
        case Hypothesis::Type::Electron:
          keep(h.trackIdx, TrackType::kElectron, weights[i]);
          break;
        case Hypothesis::Type::ChargedHadron:
        case Hypothesis::Type::RecoveryChargedHadron:
          if (!isTrackOnly(h))
            keep(h.trackIdx, TrackType::kChargedHadron, weights[i]);
          break;
        case Hypothesis::Type::Jet:
          for (int iTrack : h.trackIdxs)
            keep(iTrack, TrackType::kChargedHadron, weights[i]);
          break;
        default:
          break;
      }
    }
    return p;
  }

}  // namespace ticl

#endif
