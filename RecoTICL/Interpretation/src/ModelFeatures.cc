#include "RecoTICL/Interpretation/interface/ModelFeatures.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/Vector3D.h"

namespace ticl::modelfeatures {

  using Probabilities = std::remove_cvref_t<decltype(std::declval<const Trackster &>().id_probabilities())>;
  static_assert(std::tuple_size_v<Probabilities> == 8, "the models read 8 PID classes");

  void fillHypothesisFeatures(const Hypothesis &h,
                              const Trackster *footprint,
                              const std::vector<reco::Track> &tracks,
                              const HypothesisContext &context,
                              std::span<float, hypothesis::kN> x) {
    using namespace hypothesis;
    x[kType + static_cast<int>(h.type)] = 1.f;
    x[kScore] = h.score;
    const float rawE = footprint != nullptr ? footprint->raw_energy() : -1.f;
    const bool hasE = rawE > 0.f;
    const float tsEta = hasE ? footprint->barycenter().eta() : -99.f;
    const float tsPhi = hasE ? footprint->barycenter().phi() : -99.f;
    x[kLogRawEnergy] = hasE ? std::log1p(rawE) : -1.f;
    x[kEMFraction] = hasE ? footprint->raw_em_energy() / std::max(rawE, 1e-6f) : -1.f;
    x[kAbsEta] = hasE ? std::abs(tsEta) : -1.f;
    if (footprint != nullptr)
      std::copy(footprint->id_probabilities().begin(), footprint->id_probabilities().end(), x.begin() + kPID);
    else
      std::fill(x.begin() + kPID, x.begin() + kNLayerClusters, -1.f);
    x[kNLayerClusters] = footprint != nullptr ? static_cast<float>(footprint->vertices().size()) : 0.f;
    x[kNJetTracks] = static_cast<float>(h.trackIdxs.size());
    float trackP = -1.f, trackPt = -1.f, trackEta = -1.f, dR = -1.f;
    if (h.trackIdx >= 0) {
      const auto &tk = tracks[h.trackIdx];
      trackP = tk.p();
      trackPt = tk.pt();
      trackEta = tk.eta();
      const auto dir = tk.outerOk() ? tk.outerMomentum() : tk.momentum();
      dR = hasE ? reco::deltaR(dir.eta(), dir.phi(), tsEta, tsPhi) : -1.f;
    } else if (!h.trackIdxs.empty()) {
      // Jet: scalar sums of p and pt, direction of the summed outer momenta.
      math::XYZVector sum;
      trackP = trackPt = 0.f;
      for (int iTk : h.trackIdxs) {
        const auto &tk = tracks[iTk];
        trackP += tk.p();
        trackPt += tk.pt();
        sum += tk.outerOk() ? tk.outerMomentum() : tk.momentum();
      }
      trackEta = sum.eta();
      dR = hasE ? reco::deltaR(sum.eta(), sum.phi(), tsEta, tsPhi) : -1.f;
    }
    x[kLogTrackP] = trackP > 0.f ? std::log1p(trackP) : -1.f;
    x[kTrackPt] = trackPt;
    x[kTrackAbsEta] = trackP > 0.f ? std::abs(trackEta) : -1.f;
    const float eOverP = (hasE && trackP > 0.f) ? rawE / trackP : -1.f;
    x[kEOverP] = std::clamp(eOverP, -1.f, 20.f);
    x[kDeltaR] = dR;
    x[kComponentSize] = static_cast<float>(context.componentSize);
    auto countTrack = [&](int iTk) {
      if (iTk >= 0) {
        x[kMuonTracks] += context.muonTrack[iTk] ? 1.f : 0.f;
        x[kElectronTracks] += context.electronTrack[iTk] ? 1.f : 0.f;
        x[kEMRecoveryTracks] += context.emRecoveryTrack[iTk] ? 1.f : 0.f;
        x[kMissingInnerHitTracks] +=
            tracks[iTk].hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) > 0 ? 1.f : 0.f;
      }
    };
    if (h.trackIdxs.empty())
      countTrack(h.trackIdx);
    else
      for (int iTk : h.trackIdxs)
        countTrack(iTk);
  }

  void fillNeutralSpeciesFeatures(const Trackster &ts, std::span<float, neutralSpecies::kN> x) {
    using namespace neutralSpecies;
    std::copy(ts.id_probabilities().begin(), ts.id_probabilities().end(), x.begin() + kPID);
    x[kEMFraction] = ts.raw_em_energy() / ts.raw_energy();
    x[kLogRawEnergy] = std::log1p(ts.raw_energy());
    x[kAbsEta] = std::abs(ts.barycenter().eta());
    x[kNLayerClusters] = static_cast<float>(ts.vertices().size());
  }

  void fillNeutralShareFeatures(const Trackster &ts,
                                const std::vector<reco::CaloCluster> &layerClusters,
                                const NeutralShareContext &context,
                                std::span<float, neutralShare::kN> x) {
    using namespace neutralShare;
    const auto &b = ts.barycenter();
    x[kLogRegressedEnergy] = std::log(std::max(ts.regressed_energy(), 1e-3f));
    x[kLogRawEnergy] = std::log(std::max(ts.raw_energy(), 1e-3f));
    x[kEMFraction] = ts.raw_em_energy() / std::max(ts.raw_energy(), 1e-6f);
    x[kAbsEta] = std::abs(b.eta());
    x[kLogNLayerClusters] = std::log1p(static_cast<float>(ts.vertices().size()));
    std::copy(ts.id_probabilities().begin(), ts.id_probabilities().end(), x.begin() + kPID);
    x[kLogNearEnergy] = std::log1p(context.nearEnergy);
    x[kLogSideEnergy] = std::log1p(context.sideEnergy);
    x[kNNear] = static_cast<float>(context.nNear);
    for (int k = 0; k < 3; ++k) {
      x[kLogEigenvalues + k] = std::log(std::max(ts.eigenvalues()[k], 1e-3f));
      x[kSigmasPCA + k] = ts.sigmasPCA()[k];
      x[kSigmas + k] = ts.sigmas()[k];
    }
    // Layer clusters: the largest energy share, the energy share within 2 and 5 cm of the main axis, the
    // energy-weighted transverse rms (cm), the smallest |z| and the |z| extent.
    auto axis = ts.eigenvectors(0);
    if (!(axis.mag2() > 0.f) || !std::isfinite(axis.mag2()))
      axis = b;
    axis = axis.unit();
    float eSum = 0.f, eMax = 0.f, eCore2 = 0.f, eCore5 = 0.f, r2Sum = 0.f;
    float zMin = std::numeric_limits<float>::max(), zMax = 0.f;
    for (size_t i = 0; i < ts.vertices().size(); ++i) {
      const auto &lc = layerClusters[ts.vertices(i)];
      const float e = lc.energy() / std::max<float>(1.f, ts.vertex_multiplicity(i));
      const Trackster::Vector d(lc.x() - b.x(), lc.y() - b.y(), lc.z() - b.z());
      const float along = d.Dot(axis);
      const float r = std::sqrt(std::max(0.f, d.mag2() - along * along));
      eSum += e;
      eMax = std::max(eMax, e);
      eCore2 += r < 2.f ? e : 0.f;
      eCore5 += r < 5.f ? e : 0.f;
      r2Sum += e * r * r;
      zMin = std::min(zMin, std::abs(static_cast<float>(lc.z())));
      zMax = std::max(zMax, std::abs(static_cast<float>(lc.z())));
    }
    if (eSum > 0.f) {
      x[kMaxClusterShare] = eMax / eSum;
      x[kCoreShare2] = eCore2 / eSum;
      x[kCoreShare5] = eCore5 / eSum;
      x[kTransverseRMS] = std::sqrt(r2Sum / eSum);
      x[kMinAbsZ] = zMin;
      x[kAbsZExtent] = zMax - zMin;
    } else {
      std::fill(x.begin() + kMaxClusterShare, x.begin() + kAbsZExtent, -1.f);
      x[kAbsZExtent] = 0.f;
    }
    for (auto &v : x)
      if (!std::isfinite(v))
        v = 0.f;
  }

}  // namespace ticl::modelfeatures
