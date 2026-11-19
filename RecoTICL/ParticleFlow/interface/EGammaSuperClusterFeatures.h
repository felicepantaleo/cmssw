#ifndef RecoTICL_ParticleFlow_EGammaSuperClusterFeatures_h
#define RecoTICL_ParticleFlow_EGammaSuperClusterFeatures_h

// Input features of the energy regression (regressionFeatures) and of the score (scoreFeatures) of a TICL e/gamma
// supercluster. The regression predicts the median ratio of the generated photon energy to the raw energy.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/MultiSpan.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"

namespace ticl::egammaSuperCluster {

  inline constexpr std::size_t kNFeatures = 19;
  // Radius [cm] of the core cylinder.
  inline constexpr float kCoreRadius = 2.8f;
  // Radius [cm] of the wide cylinder.
  inline constexpr float kWideRadius = 5.f;
  // Cell size [cm] added to the cylinder radii: kHDCellSize for the wafer type HGCalHD120 (thickness index 0), else
  // kLDCellSize, also for HGCalHD200. regression_v3 is trained with these cell sizes.
  inline constexpr float kHDCellSize = 0.465f;
  inline constexpr float kLDCellSize = 0.698f;
  // Lower limit [GeV] of the energies in the denominators of the features.
  inline constexpr float kMinEnergy = 1e-3f;

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

  // A CE-E hit of a supercluster: layer index from 0, position [cm], rechit energy [GeV], hit fraction, cell size [cm].
  struct CellHit {
    unsigned int layer;
    float x, y, energy, fraction, cellSize;
  };

  // The energy of a hit is the rechit energy times the hit fraction; the layer centroids weight the hits with the
  // rechit energy. A hit with layer >= nLayers is skipped.
  inline CellFeatures cellFeatures(const std::vector<CellHit> &hits, unsigned int nLayers, double rawEnergy) {
    std::vector<std::array<double, 3>> centroid(nLayers, {0., 0., 0.});
    for (auto const &h : hits) {
      if (h.layer >= nLayers)
        continue;
      auto &c = centroid[h.layer];
      c[0] += h.energy * h.x;
      c[1] += h.energy * h.y;
      c[2] += h.energy;
    }
    CellFeatures result;
    if (rawEnergy <= 0.)
      return result;
    double core = 0., wide = 0., sum = 0., sumR2 = 0.;
    float first = 0.f, second = 0.f;
    for (auto const &h : hits) {
      if (h.layer >= nLayers)
        continue;
      const float energy = h.energy * h.fraction;
      const auto &c = centroid[h.layer];
      const double dx = h.x - c[0] / c[2];
      const double dy = h.y - c[1] / c[2];
      const double r2 = dx * dx + dy * dy;
      const double r = std::sqrt(r2);
      if (r <= kCoreRadius + h.cellSize)
        core += energy;
      if (r <= kWideRadius + h.cellSize)
        wide += energy;
      sum += energy;
      sumR2 += energy * r2;
      if (energy > first) {
        second = first;
        first = energy;
      } else if (energy > second) {
        second = energy;
      }
    }
    result.coreFraction = static_cast<float>(core / rawEnergy);
    result.wideFraction = static_cast<float>(wide / rawEnergy);
    result.maxCellFraction = static_cast<float>(first / rawEnergy);
    result.secondCellFraction = static_cast<float>(second / rawEnergy);
    result.transverseRMS = sum > 0. ? static_cast<float>(std::sqrt(sumR2 / sum)) : 0.f;
    return result;
  }

  // cellFeatures of the CE-E hits of the supercluster that have a rechit.
  inline CellFeatures cellFeatures(const reco::SuperCluster &sc,
                                   const std::unordered_map<DetId, const unsigned int> &hitMap,
                                   const edm::MultiSpan<HGCRecHit> &hits,
                                   const ticlgeom::Tools &tools) {
    const unsigned int nLayers = tools.lastLayerEE();
    std::vector<CellHit> eeHits;
    eeHits.reserve(sc.hitsAndFractions().size());
    for (auto const &[id, fraction] : sc.hitsAndFractions()) {
      if (id.det() != DetId::HGCalEE)
        continue;
      const unsigned int layer = tools.getLayer(id);
      if (layer < 1 || layer > nLayers)
        continue;
      const auto it = hitMap.find(id);
      if (it == hitMap.end())
        continue;
      const auto position = tools.getPosition(id);
      eeHits.push_back({layer - 1,
                        position.x(),
                        position.y(),
                        hits[it->second].energy(),
                        fraction,
                        tools.getSiThickIndex(id) == 0 ? kHDCellSize : kLDCellSize});
    }
    return cellFeatures(eeHits, nLayers, sc.rawEnergy());
  }

  // 0-10: log raw energy, |eta|, eta width, phi width, number of clusters, seed energy over raw energy, CE-E energy over
  // raw energy of the trackster, the three PCA sigmas of the trackster, log(1 + number of layer clusters of the
  // trackster).
  // 11-14: core fraction and largest cell fraction (cellFeatures), log of the trackster raw energy over the
  // supercluster raw energy, log(1 + nEventHits).
  // 15-18: log(1 + number of hits of the supercluster), second largest cell fraction, transverse RMS and wide fraction
  // (cellFeatures).
  inline std::array<float, kNFeatures> regressionFeatures(const reco::SuperCluster &sc,
                                                          const ticl::Trackster &ts,
                                                          const CellFeatures &cell,
                                                          std::size_t nEventHits) {
    const float raw = std::max(static_cast<float>(sc.rawEnergy()), kMinEnergy);
    const auto &sigmas = ts.sigmasPCA();
    return {std::log(raw),
            std::abs(static_cast<float>(sc.eta())),
            static_cast<float>(sc.etaWidth()),
            static_cast<float>(sc.phiWidth()),
            static_cast<float>(sc.clustersSize()),
            static_cast<float>(sc.seed()->energy()) / raw,
            ts.raw_em_energy() / std::max(ts.raw_energy(), kMinEnergy),
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

  inline constexpr std::size_t kNIdProbabilities = 8;
  inline constexpr std::size_t kNScoreFeatures = kNFeatures + kNIdProbabilities + 8;
  // Radii of the inner and outer isolation cones.
  inline constexpr float kIsolationInnerDR = 0.15f;
  inline constexpr float kIsolationOuterDR = 0.3f;

  // Transverse energy [GeV] of the layer clusters within innerDR and outerDR of the supercluster position whose seed hit
  // is not a hit of the supercluster. Only the tiles of the endcap of the supercluster are read: the layers
  // [0, lastLayer) for eta < 0 and [lastLayer, 2 lastLayer) for eta > 0.
  inline std::array<double, 2> isolationEt(const reco::SuperCluster &sc,
                                           const TICLLayerTiles &tiles,
                                           const std::vector<reco::CaloCluster> &layerClusters,
                                           unsigned int lastLayer,
                                           float innerDR,
                                           float outerDR) {
    std::unordered_set<uint32_t> own;
    own.reserve(sc.hitsAndFractions().size());
    for (auto const &hf : sc.hitsAndFractions())
      own.insert(hf.first.rawId());
    const float eta = sc.position().eta(), phi = sc.position().phi();
    std::array<double, 2> result{0., 0.};
    const unsigned int firstLayer = eta > 0.f ? lastLayer : 0;
    const unsigned int endLayer = std::min(firstLayer + lastLayer, static_cast<unsigned int>(TileConstants::nLayers));
    for (unsigned int layer = firstLayer; layer < endLayer; ++layer) {
      const auto &tile = tiles[layer];
      const auto box = tile.searchBoxEtaPhi(eta - outerDR, eta + outerDR, phi - outerDR, phi + outerDR);
      for (int iEta = box[0]; iEta <= box[1]; ++iEta)
        for (int iPhi = box[2]; iPhi <= box[3]; ++iPhi)
          for (auto i : tile[tile.globalBin(iEta, iPhi % TileConstants::nPhiBins)]) {
            const auto &lc = layerClusters[i];
            const float d2 = reco::deltaR2(lc.eta(), lc.phi(), eta, phi);
            if (d2 >= outerDR * outerDR || own.count(lc.seed().rawId()) > 0)
              continue;
            const double et = lc.energy() / std::cosh(lc.eta());
            result[1] += et;
            if (d2 < innerDR * innerDR)
              result[0] += et;
          }
    }
    return result;
  }

  // Tiles of the tracks with pT >= kTrackMinPt and |eta| >= kTrackMinAbsEta, one per endcap side (0: eta < 0).
  inline constexpr float kTrackMinPt = 1.f;
  inline constexpr float kTrackMinAbsEta = 1.15f;
  using TrackTiles = std::array<TICLLayerTile, 2>;
  inline void fillTrackTiles(const std::vector<reco::Track> &tracks, TrackTiles &tiles) {
    for (unsigned int i = 0; i < tracks.size(); ++i) {
      const auto &tk = tracks[i];
      if (tk.pt() >= kTrackMinPt && std::abs(tk.eta()) >= kTrackMinAbsEta)
        tiles[tk.eta() > 0. ? 1 : 0].fill(tk.eta(), tk.phi(), i);
    }
  }

  // Tracks around the supercluster position, with the track direction at the vertex: the number of tracks with
  // pT > kNearMinPt within kNearDR, and the sum of the pT of the tracks within kInnerDR < dR < kOuterDR.
  inline constexpr float kNearDR = 0.05f;
  inline constexpr float kNearMinPt = 2.f;
  inline constexpr float kInnerDR = 0.04f;
  inline constexpr float kOuterDR = 0.3f;
  struct TrackFeatures {
    int nNear = 0;
    double isolation = 0.;
  };
  inline TrackFeatures trackFeatures(const reco::SuperCluster &sc,
                                     const std::vector<reco::Track> &tracks,
                                     const TrackTiles &tiles) {
    const float eta = sc.position().eta(), phi = sc.position().phi();
    const auto &tile = tiles[eta > 0.f ? 1 : 0];
    TrackFeatures result;
    const auto box = tile.searchBoxEtaPhi(eta - kOuterDR, eta + kOuterDR, phi - kOuterDR, phi + kOuterDR);
    for (int iEta = box[0]; iEta <= box[1]; ++iEta)
      for (int iPhi = box[2]; iPhi <= box[3]; ++iPhi)
        for (auto i : tile[tile.globalBin(iEta, iPhi % TileConstants::nPhiBins)]) {
          const auto &tk = tracks[i];
          const float d2 = reco::deltaR2(tk.eta(), tk.phi(), eta, phi);
          if (d2 < kNearDR * kNearDR && tk.pt() > kNearMinPt)
            ++result.nNear;
          if (d2 < kOuterDR * kOuterDR && d2 > kInnerDR * kInnerDR)
            result.isolation += tk.pt();
        }
    return result;
  }

  // 0-18: regressionFeatures; 19-26: the PID probabilities of the trackster; 27: log of the transverse energy after
  // the regression; 28-29: isolationEt in the inner and outer cones over the raw transverse energy; 30: energy after the
  // regression over the raw energy; 31: the trackster is the trackster of a track in electron mode; 32: the trackster
  // is a neutral pion; 33: nNear of trackFeatures; 34: isolation of trackFeatures over the raw transverse energy.
  inline std::array<float, kNScoreFeatures> scoreFeatures(const std::array<float, kNFeatures> &regression,
                                                          const reco::SuperCluster &sc,
                                                          const ticl::Trackster &ts,
                                                          const std::array<double, 2> &isolation,
                                                          bool electronTrack,
                                                          bool neutralPion,
                                                          const TrackFeatures &tracks) {
    std::array<float, kNScoreFeatures> f{};
    std::copy(regression.begin(), regression.end(), f.begin());
    const auto &pid = ts.id_probabilities();
    static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(pid)>> == kNIdProbabilities);
    std::copy(pid.begin(), pid.end(), f.begin() + kNFeatures);
    constexpr std::size_t k = kNFeatures + kNIdProbabilities;
    const float raw = std::max(static_cast<float>(sc.rawEnergy()), kMinEnergy);
    const float sinTheta = std::sin(static_cast<float>(sc.position().theta()));
    const float rawEt = std::max(raw * sinTheta, kMinEnergy);
    f[k] = std::log(std::max(static_cast<float>(sc.energy()) * sinTheta, kMinEnergy));
    f[k + 1] = static_cast<float>(isolation[0]) / rawEt;
    f[k + 2] = static_cast<float>(isolation[1]) / rawEt;
    f[k + 3] = static_cast<float>(sc.energy()) / raw;
    f[k + 4] = electronTrack ? 1.f : 0.f;
    f[k + 5] = neutralPion ? 1.f : 0.f;
    f[k + 6] = static_cast<float>(tracks.nNear);
    f[k + 7] = static_cast<float>(tracks.isolation) / rawEt;
    return f;
  }

}  // namespace ticl::egammaSuperCluster

#endif
