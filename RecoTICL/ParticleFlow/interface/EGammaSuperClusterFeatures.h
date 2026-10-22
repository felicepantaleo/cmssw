#ifndef RecoTICL_ParticleFlow_EGammaSuperClusterFeatures_h
#define RecoTICL_ParticleFlow_EGammaSuperClusterFeatures_h

// Input features of the energy regression of a TICL e/gamma supercluster: quantities of the supercluster and of its
// e/gamma trackster only. The regression predicts the median ratio of the generated photon energy to the raw energy.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

namespace ticl::egammaSuperCluster {

  inline constexpr std::size_t kNFeatures = 11;

  // log raw energy, |eta|, eta width, phi width, number of clusters, seed energy over raw energy, CE-E energy over
  // raw energy of the trackster, the three PCA sigmas of the trackster, log(1 + number of layer clusters).
  inline std::array<float, kNFeatures> regressionFeatures(const reco::SuperCluster &sc, const ticl::Trackster &ts) {
    const float raw = std::max(static_cast<float>(sc.rawEnergy()), 1e-3f);
    const auto &sigmas = ts.sigmasPCA();
    return {std::log(raw),
            std::abs(static_cast<float>(sc.eta())),
            static_cast<float>(sc.etaWidth()),
            static_cast<float>(sc.phiWidth()),
            static_cast<float>(sc.clustersSize()),
            static_cast<float>(sc.seed()->energy()) / raw,
            ts.raw_em_energy() / std::max(ts.raw_energy(), 1e-3f),
            sigmas[0],
            sigmas[1],
            sigmas[2],
            std::log(1.f + static_cast<float>(ts.vertices().size()))};
  }

}  // namespace ticl::egammaSuperCluster

#endif
