#ifndef RecoTICL_Interpretation_CandidateArbitration_h
#define RecoTICL_Interpretation_CandidateArbitration_h

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace ticl {

  // Trackster energy resolution sigma/E = sqrt(S^2/E + C^2), with the EM or the hadronic terms.
  struct TracksterResolution {
    float emStochastic;
    float emConstant;
    float hadStochastic;
    float hadConstant;
    // Energy error of a trackster that measures the energy e.
    float error(bool hadronic, float e) const;
  };

  struct EnergyWithError {
    float energy;
    float sigma;
  };

  // Energy and error of a hadronic trackster from the 16, 50 and 84% quantiles q of its true energy over its regressed
  // energy: regressedEnergy x q50 and regressedEnergy x |q84 - q16| / 2.
  EnergyWithError calibratedEnergy(float regressedEnergy, std::span<const float, 3> q);

  // Energy of a track (p, sigmaP, mass) on a trackster with its regressed energy and the resolution at the track energy:
  // the inverse-variance mean when they agree within nSigma combined errors (compatible), else the track energy.
  struct CombinedEnergy {
    float energy;
    bool compatible;
  };
  CombinedEnergy combinedEnergy(
      float p, float sigmaP, float mass, const Trackster &ts, const TracksterResolution &resolution, float nSigma);

  // A charged hadron that shares the energy of a trackster with the other charged hadrons on it: its candidate, its
  // track energy and variance, and the energy and variance of the claim trackster of its track (0 when none).
  struct ChargedOnTrackster {
    std::size_t candidate;
    float energy;
    float variance;
    bool jetMember;
    float claimEnergy;
    float claimVariance;
  };

  // Energy of a trackster that charged hadrons share: calibrated (else regressed) plus the claim energies. Error: the
  // calibrated error with the claim variances, else the resolution at the sum of the track energies.
  EnergyWithError sharedTracksterEnergy(const Trackster &ts,
                                        const std::optional<EnergyWithError> &calibrated,
                                        std::span<const ChargedOnTrackster> charged,
                                        const TracksterResolution &resolution);

  // Energies of the charged hadrons on a trackster: when their sum agrees with it within nSigma, each takes a share of
  // the difference in proportion to its variance (combined), unless one falls below minEnergy; else they do not change.
  struct SharedEnergies {
    std::vector<float> energies;
    bool combined;
  };
  SharedEnergies shareTracksterEnergy(std::span<const ChargedOnTrackster> charged,
                                      EnergyWithError trackster,
                                      float nSigma,
                                      float minEnergy);

  // The total energy of a trackster that holds charged candidates and the energy that they take from it.
  struct TracksterEnergyBalance {
    float energy = 0.f;
    float given = 0.f;
  };

  // Total energy of a trackster that charged hadrons share: the regressed energy plus the claim energies. The
  // calibrated energy of sharedTracksterEnergy estimates the charged part only.
  float sharedTracksterTotalEnergy(const Trackster &ts, std::span<const ChargedOnTrackster> charged);

  // Energy of the neutral residual of a trackster: energy - given when it is at least max(floor, fraction x energy).
  std::optional<float> neutralResidualEnergy(const TracksterEnergyBalance &balance, float floor, float fraction);

  // Energy scale of a neutral candidate (e, pt): 1 when shareMinPt > 0 and pt >= shareMinPt, else share. No value
  // when the scaled energy is not positive or below minEnergy: the candidate is not produced.
  std::optional<float> neutralEnergyScale(float e, float pt, float share, float shareMinPt, float minEnergy);

  // The nearest GSF track to the track in (eta, phi), with the charge of the track, within maxDeltaR and not used;
  // -1 when none.
  int matchGsfTrack(const reco::Track &track,
                    const std::vector<reco::GsfTrack> &gsfTracks,
                    const std::vector<bool> &used,
                    float maxDeltaR);

  // Candidate candidateIndex[i] holds the rebuilt trackster ptrOf(i) alone, with raw energy rawEnergies[i]. Every other
  // candidate that holds tracksters then holds none and has raw energy 0.
  template <typename PtrOf>
  void setCandidateTracksters(std::vector<TICLCandidate> &candidates,
                              const std::vector<int> &candidateIndex,
                              const std::vector<float> &rawEnergies,
                              PtrOf &&ptrOf) {
    for (auto &cand : candidates) {
      if (cand.tracksters().empty())
        continue;
      cand.setTracksters({});
      cand.setRawEnergy(0.f);
    }
    for (std::size_t i = 0; i < candidateIndex.size(); ++i) {
      auto &cand = candidates[candidateIndex[i]];
      cand.setTracksters({ptrOf(i)});
      cand.setRawEnergy(rawEnergies[i]);
    }
  }

  // Removes the candidates without track and without trackster. Returns the index before the removal of each kept
  // candidate.
  std::vector<std::size_t> removeEmptyCandidates(std::vector<TICLCandidate> &candidates);

}  // namespace ticl

#endif
