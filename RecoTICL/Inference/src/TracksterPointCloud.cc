#include "RecoTICL/Inference/interface/TracksterPointCloud.h"

#include <algorithm>
#include <cmath>

#include "DataFormats/Math/interface/deltaR.h"
#include "RecoTICL/Inference/interface/TracksterLayerClusterProfile.h"

namespace ticl {

  namespace {
    constexpr std::array<const char*, TracksterPointCloud::kNPointFeatures> kPointNames{{"dx",
                                                                                         "dy",
                                                                                         "dz",
                                                                                         "layer",
                                                                                         "logE",
                                                                                         "eFrac",
                                                                                         "dt",
                                                                                         "timeValid",
                                                                                         "nHits",
                                                                                         "scintillator",
                                                                                         "code",
                                                                                         "share",
                                                                                         "dtPV",
                                                                                         "dtPVValid"}};
    constexpr std::array<const char*, TracksterPointCloud::kNGlobalFeatures> kGlobalNames{
        {"rawE",      "rawEmE",     "bx",         "by",          "bz",        "eta",           "phi",
         "ax",        "ay",         "az",         "axisPCA",     "axisAngle", "eig0",          "eig1",
         "eig2",      "sigPCA0",    "sigPCA1",    "sigPCA2",     "sig0",      "sig1",          "sig2",
         "nLC",       "nLayers",    "firstLayer", "lastLayer",   "time",      "timeErr",       "lcMaxFrac",
         "lcCore2",   "lcCore5",    "lcRms",      "lcZ0",        "lcZExtent", "nearRawE",      "nNear",
         "sideRawE",  "cylEThis",   "cylEOther",  "cylEMasked",  "cylENone",  "ringEOther",    "ringEMasked",
         "ringENone", "nPointsAll", "nVertices",  "tkValid",     "tkP",       "tkPt",          "tkEta",
         "tkPhi",     "tkDist",     "tkDt",       "tkTimeValid", "nOwnAll",   "nNeighbourAll", "pvValid",
         "pvZ",       "pvT",        "pvTErr",     "pvTimeValid", "dtPV",      "dtPVSig",       "dtPVValid"}};
    constexpr unsigned int kMaxLayer = 64;
    // eta of the direction from the origin to transverse radius r and |z|.
    float etaOf(float r, float absZ) { return std::asinh(absZ / std::max(r, 1e-3f)); }
    // Number of phi bins of the layer-cluster tiles.
    constexpr int kNPhiBins = ticl::TileConstants::nPhiBins;
    // Speed of light [cm/ns].
    constexpr float kC = 29.9792458f;
    // Time of flight [ns] from the vertex v to the point x: |x - v| / c.
    float flightTime(float x, float y, float z, const TracksterPointCloud::PrimaryVertex& v) {
      return std::sqrt((x - v.x) * (x - v.x) + (y - v.y) * (y - v.y) + (z - v.z) * (z - v.z)) / kC;
    }
  }  // namespace

  const char* TracksterPointCloud::pointFeatureName(unsigned int i) { return kPointNames[i]; }
  const char* TracksterPointCloud::globalFeatureName(unsigned int i) { return kGlobalNames[i]; }

  int TracksterPointCloud::tileLayer(const reco::CaloCluster& layerCluster, const ticlgeom::Tools& rhtools) {
    const auto firstHit = layerCluster.hitsAndFractions()[0].first;
    return static_cast<int>(rhtools.getLayerWithOffset(firstHit)) +
           static_cast<int>(rhtools.lastLayer()) * ((rhtools.zside(firstHit) + 1) >> 1) - 1;
  }

  bool TracksterPointCloud::maskedByCLUE3DHighFilter(const reco::CaloCluster& layerCluster,
                                                     const ticlgeom::Tools& rhtools) {
    const int algo = static_cast<int>(layerCluster.algo());
    const auto& hits = layerCluster.hitsAndFractions();
    return (algo != 6 && algo != 7 && algo != 8) || hits.size() > 9999 ||
           (hits.size() < 2 && rhtools.isSilicon(hits[0].first));
  }

  TracksterPointCloud::Event::Event(const std::vector<Trackster>& stage,
                                    const std::vector<reco::CaloCluster>& layerClusters,
                                    const edm::ValueMap<std::pair<float, float>>* layerClusterTimes,
                                    const TICLLayerTiles& tiles,
                                    const ticlgeom::Tools& rhtools,
                                    int nVertices,
                                    const PrimaryVertex& primaryVertex)
      : stage_(&stage),
        layerClusters_(&layerClusters),
        tiles_(&tiles),
        nVertices_(nVertices),
        pv_(primaryVertex),
        lastLayerEE_(static_cast<int>(rhtools.lastLayerEE())),
        lastLayer_(static_cast<int>(rhtools.lastLayer())) {
    layerZ_.assign(lastLayer_ + 1, 0.f);
    for (int l = 1; l <= lastLayer_; ++l)
      layerZ_[l] = std::abs(rhtools.getPositionLayer(l).z());
    const unsigned int n = layerClusters.size();
    owner_.assign(n, -1);
    std::vector<float> ownerShare(n, 0.f);
    for (unsigned int i = 0; i < stage.size(); ++i) {
      const auto& t = stage[i];
      for (size_t j = 0; j < t.vertices().size(); ++j) {
        const auto v = t.vertices(j);
        if (v >= n)
          continue;
        const float m = 1.f / std::max(1.f, t.vertex_multiplicity(j));
        if (m > ownerShare[v]) {
          ownerShare[v] = m;
          owner_[v] = static_cast<int>(i);
        }
      }
      const auto& b = t.barycenter();
      const int side = b.z() > 0.f ? 1 : 0;
      sideRawE_[side] += t.raw_energy();
      tracksterTiles_[side].fill(b.eta(), b.phi(), i);
    }
    masked_.resize(n);
    layer_.resize(n);
    nHits_.resize(n);
    scintillator_.resize(n);
    time_.assign(n, 0.f);
    timeError_.assign(n, -1.f);
    for (unsigned int k = 0; k < n; ++k) {
      const auto& lc = layerClusters[k];
      const auto firstHit = lc.hitsAndFractions()[0].first;
      masked_[k] = maskedByCLUE3DHighFilter(lc, rhtools);
      layer_[k] = static_cast<short>(rhtools.getLayerWithOffset(firstHit));
      nHits_[k] = static_cast<short>(lc.hitsAndFractions().size());
      scintillator_[k] = rhtools.isScintillator(firstHit) ? 1 : 0;
      if (layerClusterTimes != nullptr) {
        const auto& t = layerClusterTimes->get(k);
        time_[k] = t.first;
        timeError_[k] = t.second;
      }
    }
  }

  TracksterPointCloud::Result TracksterPointCloud::build(const Event& event,
                                                         unsigned int trackster,
                                                         const Track* track) const {
    const auto& stage = event.stage();
    const auto& lcs = event.layerClusters();
    const auto& t = stage[trackster];
    const auto& b = t.barycenter();
    const int side = b.z() > 0.f ? 1 : 0;
    Result result;
    auto& g = result.global;
    g.fill(0.f);

    // Axis.
    Trackster::Vector bdir(b.x(), b.y(), b.z());
    bdir = bdir.unit();
    Trackster::Vector axis = t.eigenvectors(0);
    bool pca = false;
    float angle = -1.f;
    if (t.vertices().size() >= 3 && std::isfinite(axis.mag2()) && axis.mag2() > 0.f) {
      axis = axis.unit();
      if (axis.z() * b.z() < 0.f)
        axis = -axis;
      angle = std::acos(std::clamp(static_cast<float>(axis.Dot(bdir)), -1.f, 1.f));
      pca = angle < config_.maxAxisAngle;
    }
    if (!pca)
      axis = bdir;

    g[kRawE] = t.raw_energy();
    g[kRawEmE] = t.raw_em_energy();
    g[kBx] = b.x();
    g[kBy] = b.y();
    g[kBz] = b.z();
    g[kEta] = b.eta();
    g[kPhi] = b.phi();
    g[kAx] = axis.x();
    g[kAy] = axis.y();
    g[kAz] = axis.z();
    g[kAxisPCA] = pca ? 1.f : 0.f;
    g[kAxisAngle] = angle;
    for (int k = 0; k < 3; ++k) {
      g[kEig0 + k] = t.eigenvalues()[k];
      g[kSigPCA0 + k] = t.sigmasPCA()[k];
      g[kSig0 + k] = t.sigmas()[k];
    }
    g[kNLC] = t.vertices().size();
    g[kTime] = t.time();
    g[kTimeErr] = t.timeError();
    const auto profile = layerClusterProfile(t, lcs);
    for (int k = 0; k < 6; ++k)
      g[kLcMaxFrac + k] = profile[k];
    g[kSideRawE] = event.sideRawE_[side];
    g[kNVertices] = event.nVertices_;

    // Own layer clusters (index, share) and the layers.
    std::vector<std::pair<unsigned int, float>> own;
    own.reserve(t.vertices().size());
    std::array<bool, kMaxLayer> seen{};
    int nLayers = 0, first = 999, last = -1;
    for (size_t j = 0; j < t.vertices().size(); ++j) {
      const auto v = t.vertices(j);
      if (v >= lcs.size())
        continue;
      own.emplace_back(v, 1.f / std::max(1.f, t.vertex_multiplicity(j)));
      const int l = event.layer_[v];
      if (l >= 0 && static_cast<unsigned int>(l) < kMaxLayer && !seen[l]) {
        seen[l] = true;
        ++nLayers;
      }
      first = std::min(first, l);
      last = std::max(last, l);
    }
    std::sort(own.begin(), own.end());
    // One entry per layer cluster, with the sum of its shares.
    {
      size_t w = 0;
      for (size_t r = 0; r < own.size(); ++r) {
        if (w > 0 && own[w - 1].first == own[r].first)
          own[w - 1].second += own[r].second;
        else
          own[w++] = own[r];
      }
      own.resize(w);
    }
    g[kNLayers] = nLayers;
    g[kFirstLayer] = last >= 0 ? first : -1;
    g[kLastLayer] = last;

    // Other tracksters of the stage within dR 0.2 on the same side (trackster tiles).
    {
      const auto& tile = event.tracksterTiles_[side];
      const float eta = b.eta(), phi = b.phi();
      const auto box = tile.searchBoxEtaPhi(eta - 0.2f, eta + 0.2f, phi - 0.2f, phi + 0.2f);
      for (int ie = box[0]; ie <= box[1]; ++ie)
        for (int ip = box[2]; ip <= box[3]; ++ip)
          for (auto o : tile[tile.globalBin(ie, ip % kNPhiBins)]) {
            const auto& c = stage[o].barycenter();
            if (o == trackster || reco::deltaR(c.eta(), c.phi(), eta, phi) >= 0.2f)
              continue;
            g[kNearRawE] += stage[o].raw_energy();
            g[kNNear] += 1.f;
          }
    }

    // Points and rings.
    struct Candidate {
      float energy, dx, dy, share;
      unsigned int lc;
      int code;
    };
    std::vector<Candidate> ownCands, cands;
    const bool timeValid = t.timeError() > 0.f;
    if (axis.z() != 0.f) {
      const float tx = axis.x() / axis.z(), ty = axis.y() / axis.z();
      // Every layer cluster of the trackster is a point, in the cylinder or not.
      ownCands.reserve(own.size());
      for (auto const& [k, share] : own) {
        const auto& lc = lcs[k];
        const float dzl = lc.z() - b.z();
        ownCands.push_back({static_cast<float>(lc.energy()),
                            static_cast<float>(lc.x() - (b.x() + dzl * tx)),
                            static_cast<float>(lc.y() - (b.y() + dzl * ty)),
                            share,
                            k,
                            kThis});
      }
      const float sign = side == 1 ? 1.f : -1.f;
      for (int l = 1; l <= event.lastLayer_; ++l) {
        const auto& tile = (*event.tiles_)[l - 1 + side * event.lastLayer_];
        const float r = radius(l, event.lastLayerEE_);
        const float rOut = config_.ringOuterFactor * r;
        // Axis point at the layer z, and the eta-phi box of the circle of radius rOut (1 cm of margin for the depth of
        // the layer).
        const float zl = sign * event.layerZ_[l];
        const float px = b.x() + (zl - b.z()) * tx, py = b.y() + (zl - b.z()) * ty;
        const float rho = std::hypot(px, py);
        const float rBox = rOut + 1.f;
        const float etaLo = etaOf(rho + rBox, event.layerZ_[l]);
        const float etaHi = etaOf(std::max(rho - rBox, 0.f), event.layerZ_[l]);
        const float phi = std::atan2(py, px);
        const float dphi = rBox < rho ? std::asin(rBox / rho) : static_cast<float>(M_PI);
        const auto box = side == 1 ? tile.searchBoxEtaPhi(etaLo, etaHi, phi - dphi, phi + dphi)
                                   : tile.searchBoxEtaPhi(-etaHi, -etaLo, phi - dphi, phi + dphi);
        const int phiMax = dphi >= static_cast<float>(M_PI) ? box[2] + kNPhiBins - 1 : box[3];
        for (int ie = box[0]; ie <= box[1]; ++ie)
          for (int ip = box[2]; ip <= phiMax; ++ip)
            for (auto k : tile[tile.globalBin(ie, ip % kNPhiBins)]) {
              const auto& lc = lcs[k];
              const float dzl = lc.z() - b.z();
              const float dx = lc.x() - (b.x() + dzl * tx);
              const float dy = lc.y() - (b.y() + dzl * ty);
              const float rr = dx * dx + dy * dy;
              if (rr >= rOut * rOut)
                continue;
              auto ownIt =
                  std::lower_bound(own.begin(), own.end(), std::make_pair(k, 0.f), [](auto const& a, auto const& c) {
                    return a.first < c.first;
                  });
              const bool isOwn = ownIt != own.end() && ownIt->first == k;
              int code;
              if (isOwn)
                code = kThis;
              else if (event.owner_[k] >= 0)
                code = kOther;
              else if (event.masked_[k])
                code = kMaskedSingleSilicon;
              else
                code = kNone;
              const float e = lc.energy();
              if (rr >= r * r) {
                if (code == kOther)
                  g[kRingEOther] += e;
                else if (code == kMaskedSingleSilicon)
                  g[kRingEMasked] += e;
                else if (code == kNone)
                  g[kRingENone] += e;
                continue;
              }
              g[kCylEThis + code] += e;
              g[kNPointsAll] += 1.f;
              if (!isOwn)
                cands.push_back({e, dx, dy, 0.f, k, code});
            }
      }
    }
    g[kNOwnAll] = ownCands.size();
    g[kNNeighbourAll] = cands.size();
    auto higher = [](auto const& a, auto const& c) {
      return a.energy > c.energy || (a.energy == c.energy && a.lc < c.lc);
    };
    auto cap = [&](std::vector<Candidate>& v, unsigned int n) {
      if (v.size() > n) {
        std::partial_sort(v.begin(), v.begin() + n, v.end(), higher);
        v.resize(n);
      } else {
        std::sort(v.begin(), v.end(), higher);
      }
    };
    cap(ownCands, config_.maxOwnPoints);
    cap(cands, config_.maxNeighbourPoints);
    cands.insert(cands.begin(), ownCands.begin(), ownCands.end());
    result.points.reserve(cands.size() * kNPointFeatures);
    result.layerCluster.reserve(cands.size());
    result.owner.reserve(cands.size());
    const float raw = t.raw_energy();
    const auto& pv = event.pv_;
    const bool pvTimeValid = pv.valid && pv.tError > 0.f;
    for (auto const& c : cands) {
      const auto& lc = lcs[c.lc];
      const bool tv = timeValid && event.timeError_[c.lc] > 0.f;
      std::array<float, kNPointFeatures> p;
      p[kDx] = c.dx;
      p[kDy] = c.dy;
      p[kDz] = lc.z() - b.z();
      p[kLayer] = event.layer_[c.lc];
      p[kLogE] = std::log(std::max(c.energy, 1e-6f));
      p[kEFrac] = raw > 0.f ? c.energy / raw : 0.f;
      p[kDt] = tv ? event.time_[c.lc] - t.time() : 0.f;
      p[kTimeValid] = tv ? 1.f : 0.f;
      p[kNHits] = event.nHits_[c.lc];
      p[kScintillator] = event.scintillator_[c.lc];
      p[kCode] = c.code;
      p[kShare] = c.share;
      const bool pvt = pvTimeValid && event.timeError_[c.lc] > 0.f;
      p[kPointDtPV] = pvt ? event.time_[c.lc] - pv.t - flightTime(lc.x(), lc.y(), lc.z(), pv) : -99.f;
      p[kPointDtPVValid] = pvt ? 1.f : 0.f;
      result.points.insert(result.points.end(), p.begin(), p.end());
      result.layerCluster.push_back(static_cast<int>(c.lc));
      result.owner.push_back(c.code == kThis ? static_cast<int>(trackster) : event.owner_[c.lc]);
    }

    // Track at the HGCAL front face.
    if (track != nullptr) {
      const float pMag = std::sqrt(track->px * track->px + track->py * track->py + track->pz * track->pz);
      const float pt = std::hypot(track->px, track->py);
      g[kTkValid] = 1.f;
      g[kTkP] = pMag;
      g[kTkPt] = pt;
      g[kTkEta] = pt > 0.f ? std::asinh(track->pz / pt) : 0.f;
      g[kTkPhi] = std::atan2(track->py, track->px);
      if (track->pz != 0.f) {
        const float s = (b.z() - track->z) / track->pz;
        g[kTkDist] = std::hypot(track->x + s * track->px - b.x(), track->y + s * track->py - b.y());
      } else {
        g[kTkDist] = -1.f;
      }
      const bool tv = timeValid && track->timeError > 0.f;
      g[kTkDt] = tv ? track->time - t.time() : 0.f;
      g[kTkTimeValid] = tv ? 1.f : 0.f;
    }
    // Primary vertex and the trackster time against it.
    g[kPvValid] = pv.valid ? 1.f : 0.f;
    g[kPvZ] = pv.valid ? pv.z : 0.f;
    g[kPvT] = pvTimeValid ? pv.t : 0.f;
    g[kPvTErr] = pvTimeValid ? pv.tError : -1.f;
    g[kPvTimeValid] = pvTimeValid ? 1.f : 0.f;
    const bool dtv = pvTimeValid && timeValid;
    g[kDtPV] = dtv ? t.time() - pv.t - flightTime(b.x(), b.y(), b.z(), pv) : -99.f;
    g[kDtPVSig] = dtv ? g[kDtPV] / std::sqrt(t.timeError() * t.timeError() + pv.tError * pv.tError) : -99.f;
    g[kDtPVValid] = dtv ? 1.f : 0.f;
    for (auto& v : g)
      if (!std::isfinite(v))
        v = 0.f;
    for (auto& v : result.points)
      if (!std::isfinite(v))
        v = 0.f;
    return result;
  }

}  // namespace ticl
