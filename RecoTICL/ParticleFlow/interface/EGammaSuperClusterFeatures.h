#ifndef RecoTICL_ParticleFlow_EGammaSuperClusterFeatures_h
#define RecoTICL_ParticleFlow_EGammaSuperClusterFeatures_h

// Input features of the energy regression of a TICL e/gamma supercluster. The regression predicts the median ratio of
// the generated photon energy to the raw energy, and reads the first N features of regressionFeatures: N = 11 for
// regression_v2, kNFeatures for regression_v3.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "DataFormats/Common/interface/MultiSpan.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

namespace ticl::egammaSuperCluster {

  inline constexpr std::size_t kNFeaturesV2 = 11;
  inline constexpr std::size_t kNFeatures = 19;
  // Radius [cm] of the core cylinder, as the R variable of the TICLv5 supercluster regression.
  inline constexpr float kCoreRadius = 2.8f;
  // Radius [cm] of the wide cylinder.
  inline constexpr float kWideRadius = 5.f;

  // Cell-level quantities of a supercluster, from its CE-E hits.
  struct CellFeatures {
    // Energy within kCoreRadius plus the cell size of the energy centroid of each layer, over the raw energy.
    float coreFraction = 0.f;
    // Energy within kWideRadius plus the cell size of the energy centroid of each layer, over the raw energy.
    float wideFraction = 0.f;
    // Energies of the cells of largest and second largest energy, over the raw energy.
    float maxCellFraction = 0.f;
    float secondCellFraction = 0.f;
    // Energy-weighted RMS [cm] of the transverse distance of the hits to the energy centroid of their layer.
    float transverseRMS = 0.f;
  };

  // The energy of a hit is the rechit energy times the hit fraction; the layer centroids weight the hits with the
  // rechit energy. Hits without a rechit are skipped.
  inline CellFeatures cellFeatures(const reco::SuperCluster &sc,
                                   const std::unordered_map<DetId, const unsigned int> &hitMap,
                                   const edm::MultiSpan<HGCRecHit> &hits,
                                   const ticlgeom::Tools &tools) {
    constexpr float kHDCellSize = 0.465f;
    constexpr float kLDCellSize = 0.698f;
    struct Hit {
      unsigned int layer;
      float x, y, energy, cellSize;
    };
    const unsigned int nLayers = tools.lastLayerEE();
    std::vector<Hit> eeHits;
    eeHits.reserve(sc.hitsAndFractions().size());
    std::vector<std::array<double, 3>> centroid(nLayers, {0., 0., 0.});
    for (auto const &[id, fraction] : sc.hitsAndFractions()) {
      if (id.det() != DetId::HGCalEE)
        continue;
      const unsigned int layer = tools.getLayer(id);
      if (layer < 1 || layer > nLayers)
        continue;
      const auto it = hitMap.find(id);
      if (it == hitMap.end())
        continue;
      const float energy = hits[it->second].energy();
      const auto position = tools.getPosition(id);
      auto &c = centroid[layer - 1];
      c[0] += energy * position.x();
      c[1] += energy * position.y();
      c[2] += energy;
      eeHits.push_back({layer - 1,
                        position.x(),
                        position.y(),
                        energy * fraction,
                        tools.getSiThickIndex(id) == 0 ? kHDCellSize : kLDCellSize});
    }
    CellFeatures result;
    const double raw = sc.rawEnergy();
    if (raw <= 0.)
      return result;
    double core = 0., wide = 0., sum = 0., sumR2 = 0.;
    float first = 0.f, second = 0.f;
    for (auto const &h : eeHits) {
      const auto &c = centroid[h.layer];
      const double dx = h.x - c[0] / c[2];
      const double dy = h.y - c[1] / c[2];
      const double r2 = dx * dx + dy * dy;
      const double r = std::sqrt(r2);
      if (r <= kCoreRadius + h.cellSize)
        core += h.energy;
      if (r <= kWideRadius + h.cellSize)
        wide += h.energy;
      sum += h.energy;
      sumR2 += h.energy * r2;
      if (h.energy > first) {
        second = first;
        first = h.energy;
      } else if (h.energy > second) {
        second = h.energy;
      }
    }
    result.coreFraction = static_cast<float>(core / raw);
    result.wideFraction = static_cast<float>(wide / raw);
    result.maxCellFraction = static_cast<float>(first / raw);
    result.secondCellFraction = static_cast<float>(second / raw);
    result.transverseRMS = sum > 0. ? static_cast<float>(std::sqrt(sumR2 / sum)) : 0.f;
    return result;
  }

  // 0-10 (regression_v2): log raw energy, |eta|, eta width, phi width, number of clusters, seed energy over raw energy,
  // CE-E energy over raw energy of the trackster, the three PCA sigmas of the trackster, log(1 + number of layer
  // clusters of the trackster).
  // 11-14: core fraction and largest cell fraction (cellFeatures), log of the trackster raw energy over the
  // supercluster raw energy, log(1 + number of HGCAL rechits in the event).
  // 15-18: log(1 + number of hits of the supercluster), second largest cell fraction, transverse RMS and wide fraction
  // (cellFeatures).
  inline std::array<float, kNFeatures> regressionFeatures(const reco::SuperCluster &sc,
                                                          const ticl::Trackster &ts,
                                                          const CellFeatures &cell,
                                                          std::size_t nEventHits) {
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
            std::log(1.f + static_cast<float>(ts.vertices().size())),
            cell.coreFraction,
            cell.maxCellFraction,
            std::log(std::max(ts.raw_energy(), raw) / raw),
            std::log(1.f + static_cast<float>(nEventHits)),
            std::log(1.f + static_cast<float>(sc.hitsAndFractions().size())),
            cell.secondCellFraction,
            cell.transverseRMS,
            cell.wideFraction};
  }

}  // namespace ticl::egammaSuperCluster

#endif
