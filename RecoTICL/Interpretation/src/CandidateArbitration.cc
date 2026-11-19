#include "RecoTICL/Interpretation/interface/CandidateArbitration.h"

#include <algorithm>
#include <cmath>

#include "DataFormats/Math/interface/deltaR.h"

namespace ticl {

  float TracksterResolution::error(bool hadronic, float e) const {
    const float s = hadronic ? hadStochastic : emStochastic;
    const float c = hadronic ? hadConstant : emConstant;
    return e * std::sqrt(s * s / std::max(e, 0.1f) + c * c);
  }

  EnergyWithError calibratedEnergy(float regressedEnergy, std::span<const float, 3> q) {
    return {regressedEnergy * q[1], regressedEnergy * 0.5f * std::abs(q[2] - q[0])};
  }

  CombinedEnergy combinedEnergy(
      float p, float sigmaP, float mass, const Trackster &ts, const TracksterResolution &resolution, float nSigma) {
    const float eTrack = std::sqrt(p * p + mass * mass);
    const float eTrackster = ts.regressed_energy();
    const float sigmaTrackster = resolution.error(ts.isHadronic(), eTrack);
    if (!(std::abs(eTrackster - eTrack) < nSigma * std::hypot(sigmaP, sigmaTrackster)))
      return {eTrack, false};
    constexpr float kMinVariance = 1e-12f;
    const float wTrack = 1.f / std::max(sigmaP * sigmaP, kMinVariance);
    const float wTrackster = 1.f / std::max(sigmaTrackster * sigmaTrackster, kMinVariance);
    return {(wTrack * eTrack + wTrackster * eTrackster) / (wTrack + wTrackster), true};
  }

  EnergyWithError sharedTracksterEnergy(const Trackster &ts,
                                        const std::optional<EnergyWithError> &calibrated,
                                        std::span<const ChargedOnTrackster> charged,
                                        const TracksterResolution &resolution) {
    float eSum = 0.f, eClaim = 0.f, varClaim = 0.f;
    for (auto const &c : charged) {
      eSum += c.energy;
      eClaim += c.claimEnergy;
      varClaim += c.claimVariance;
    }
    if (!calibrated)
      return {ts.regressed_energy() + eClaim, resolution.error(ts.isHadronic(), eSum)};
    return {calibrated->energy + eClaim, std::sqrt(calibrated->sigma * calibrated->sigma + varClaim)};
  }

  float sharedTracksterTotalEnergy(const Trackster &ts, std::span<const ChargedOnTrackster> charged) {
    float energy = ts.regressed_energy();
    for (auto const &c : charged)
      energy += c.claimEnergy;
    return energy;
  }

  SharedEnergies shareTracksterEnergy(std::span<const ChargedOnTrackster> charged,
                                      EnergyWithError trackster,
                                      float nSigma,
                                      float minEnergy) {
    float eSum = 0.f, varSum = 0.f;
    for (auto const &c : charged) {
      eSum += c.energy;
      varSum += c.variance;
    }
    const float varTotal = varSum + trackster.sigma * trackster.sigma;
    const float diff = trackster.energy - eSum;
    bool combine = std::abs(diff) < nSigma * std::sqrt(varTotal);
    for (auto const &c : charged)
      combine = combine && c.energy + c.variance / varTotal * diff > minEnergy;
    SharedEnergies result{{}, combine};
    result.energies.reserve(charged.size());
    for (auto const &c : charged)
      result.energies.push_back(combine ? c.energy + c.variance / varTotal * diff : c.energy);
    return result;
  }

  std::optional<float> neutralResidualEnergy(const TracksterEnergyBalance &balance, float floor, float fraction) {
    const float residual = balance.energy - balance.given;
    if (!(residual >= std::max(floor, fraction * balance.energy)))
      return std::nullopt;
    return residual;
  }

  std::optional<float> neutralEnergyScale(float e, float pt, float share, float shareMinPt, float minEnergy) {
    const float scale = shareMinPt > 0.f && pt >= shareMinPt ? 1.f : share;
    const float scaled = e * scale;
    if (!(scaled > 0.f) || scaled < minEnergy)
      return std::nullopt;
    return scale;
  }

  std::vector<std::size_t> removeEmptyCandidates(std::vector<TICLCandidate> &candidates) {
    std::vector<std::size_t> kept;
    std::vector<TICLCandidate> out;
    out.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (candidates[i].trackPtr().isNull() && candidates[i].tracksters().empty())
        continue;
      kept.push_back(i);
      out.push_back(std::move(candidates[i]));
    }
    candidates = std::move(out);
    return kept;
  }

  int matchGsfTrack(const reco::Track &track,
                    const std::vector<reco::GsfTrack> &gsfTracks,
                    const std::vector<bool> &used,
                    float maxDeltaR) {
    int best = -1;
    float bestDR2 = maxDeltaR * maxDeltaR;
    for (std::size_t i = 0; i < gsfTracks.size(); ++i) {
      const auto &gsf = gsfTracks[i];
      if (used[i] || gsf.charge() != track.charge())
        continue;
      const float dR2 = reco::deltaR2(gsf.eta(), gsf.phi(), track.eta(), track.phi());
      if (dR2 < bestDR2) {
        bestDR2 = dR2;
        best = static_cast<int>(i);
      }
    }
    return best;
  }

}  // namespace ticl
