#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoHGCal/TICL/plugins/TracksterLinkingByCornetto.h"

using namespace ticl;

TracksterLinkingByCornetto::TracksterLinkingByCornetto(const edm::ParameterSet& conf,
                                                       edm::ConsumesCollector iC,
                                                       cms::Ort::ONNXRuntime const* onnxRuntime)
    : TracksterLinkingAlgoBase(conf, iC, onnxRuntime),
      etaWindow_(conf.getParameter<double>("etaWindow")),
      maxLongitudinalDistance_(conf.getParameter<double>("maxLongitudinalDistance")),
      transverseRadius0_(conf.getParameter<double>("transverseRadius0")),
      transverseSlope_(conf.getParameter<double>("transverseSlope")),
      timeCompatibilityNSigma_(conf.getParameter<double>("timeCompatibilityNSigma")),
      maxLongitudinalSlope_(conf.getParameter<double>("maxLongitudinalSlope")),
      longitudinalZRef_(conf.getParameter<double>("longitudinalZRef")),
      seededGrowth_(conf.getParameter<bool>("seededGrowth")),
      seedEnergy_(conf.getParameter<double>("seedEnergy")),
      axisToleranceCos_(std::cos(conf.getParameter<double>("axisToleranceDeg") * M_PI / 180.)),
      forwardOnly_(conf.getParameter<bool>("forwardOnly")),
      minEmittedEnergy_(conf.getParameter<double>("minEmittedEnergy")) {}

namespace {
  // Union-find with path halving. On GPU this stage becomes iterative label
  // propagation over the edge list; the emitted components are identical.
  unsigned int findRoot(std::vector<unsigned int>& parent, unsigned int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  void unite(std::vector<unsigned int>& parent, unsigned int a, unsigned int b) {
    a = findRoot(parent, a);
    b = findRoot(parent, b);
    if (a != b)
      parent[std::max(a, b)] = std::min(a, b);
  }
}  // namespace

void TracksterLinkingByCornetto::linkTracksters(
    const Inputs& input,
    std::vector<Trackster>& resultTracksters,
    std::vector<std::vector<unsigned int>>& linkedResultTracksters,
    std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) {
  // Grouping: connected components (default, bit-identical to the Alpaka kernel) or
  // seeded growth (percolation-proof at PU200). Both emit EVERY group, singletons
  // included, so the spectrum stays continuous either way.
  std::vector<std::vector<unsigned int>> groups;
  if (seededGrowth_)
    linkBySeededGrowth(input, groups);
  else
    linkByUnionFind(input, groups);

  for (auto const& group : groups) {
    if (group.empty())
      continue;
    Trackster merged;
    merged.mergeTracksters(input.tracksters, group);
    // Emit-side threshold: the pileup singletons seeded growth refuses to chain are
    // numerous but carry little energy (measured PU200: 91% of the groups hold 10% of
    // the energy). Dropping them here keeps the collection at the Skeletons size
    // without gating any LINKING decision.
    if (merged.raw_energy() < minEmittedEnergy_)
      continue;
    linkedResultTracksters.push_back({static_cast<unsigned int>(resultTracksters.size())});
    resultTracksters.push_back(merged);
    linkedTracksterIdToInputTracksterId.push_back(group);
  }
}

void TracksterLinkingByCornetto::linkBySeededGrowth(const Inputs& input,
                                                    std::vector<std::vector<unsigned int>>& groups) const {
  const auto& tracksters = input.tracksters;
  const unsigned int n = tracksters.size();

  std::vector<float> energy(n);
  std::vector<math::XYZVectorF> bary(n);
  for (unsigned int i = 0; i < n; ++i) {
    bary[i] = tracksters[i].barycenter();
    energy[i] = tracksters[i].raw_energy();
  }

  // Cores are seeded in place: a trackster above seedEnergy owns itself. No ordering is
  // imposed, because the growth below is SATELLITE centric and order independent.
  constexpr int kUnowned = -1;
  std::vector<int> owner(n, kUnowned);
  std::vector<double> coreE(n, 0.);
  std::vector<math::XYZVectorF> coreSum(n);
  for (unsigned int i = 0; i < n; ++i)
    if (energy[i] >= seedEnergy_) {
      owner[i] = static_cast<int>(i);
      coreE[i] = energy[i];
      coreSum[i] = bary[i] * energy[i];
    }

  auto unit = [](const math::XYZVectorF& v) {
    const float m = std::sqrt(v.Mag2());
    return m > 0.f ? v / m : v;
  };

  // Batch growth. Every round each unowned satellite independently picks the best core
  // that accepts it, judged against the core state as it was at the START of the round.
  // Because no satellite reads another satellite's decision, the outcome does not depend
  // on the order the satellites are visited: the round is a pure map, which is what lets
  // the same algorithm run as one kernel per round on device, and what keeps the host and
  // the Alpaka backend bit identical. A satellite that has joined a core never moves
  // again and never links to another satellite, so no chain can form (see the class
  // comment on percolation).
  std::vector<int> claim(n, kUnowned);
  for (int round = 0; round < kMaxGrowthRounds; ++round) {
    std::fill(claim.begin(), claim.end(), kUnowned);
    bool anyClaim = false;

    for (unsigned int c = 0; c < n; ++c) {
      if (owner[c] != kUnowned)
        continue;
      int best = kUnowned;
      float bestE = -1.f;
      for (unsigned int s = 0; s < n; ++s) {
        if (owner[s] != static_cast<int>(s))
          continue;  // s is not a core root
        if (bary[c].z() * bary[s].z() < 0.f)
          continue;  // same endcap only
        // Highest-energy core wins, ties broken by the smaller index, so the choice is a
        // function of the pair alone (the same rule the union-find anchor uses).
        if (coreE[s] < bestE or (coreE[s] == bestE and best != kUnowned and s > static_cast<unsigned int>(best)))
          continue;

        const math::XYZVectorF core = coreSum[s] / coreE[s];
        const math::XYZVectorF axis = unit(core);
        if (std::abs(bary[c].eta() - core.eta()) > etaWindow_)
          continue;
        if (std::abs(reco::deltaPhi(bary[c].phi(), core.phi())) > etaWindow_)
          continue;
        const auto D = bary[c] - core;
        const float sProj = D.Dot(axis);
        if (forwardOnly_ && sProj < 0.f)
          continue;  // a shower develops in depth
        const float zrel = std::abs(core.z()) - longitudinalZRef_;
        const float maxLong = maxLongitudinalDistance_ + maxLongitudinalSlope_ * (zrel > 0.f ? zrel : 0.f);
        if (std::abs(sProj) > maxLong)
          continue;
        const float dT2 = std::max(0.f, static_cast<float>(D.Mag2()) - sProj * sProj);
        const float rT = transverseRadius0_ + transverseSlope_ * std::abs(sProj);
        if (dT2 > rT * rT)
          continue;
        if (tracksters[s].timeError() > 0.f && tracksters[c].timeError() > 0.f) {
          const float sigma2 = tracksters[s].timeError() * tracksters[s].timeError() +
                               tracksters[c].timeError() * tracksters[c].timeError();
          const float dt = tracksters[s].time() - tracksters[c].time();
          if (dt * dt > timeCompatibilityNSigma_ * timeCompatibilityNSigma_ * sigma2)
            continue;
        }
        // Axis stability: a satellite that would swing the core direction belongs to a
        // neighbouring shower, not this one.
        const double newE = coreE[s] + energy[c];
        const math::XYZVectorF newAxis = unit((coreSum[s] + bary[c] * energy[c]) / newE);
        if (newAxis.Dot(axis) < axisToleranceCos_)
          continue;

        best = static_cast<int>(s);
        bestE = coreE[s];
      }
      if (best != kUnowned) {
        claim[c] = best;
        anyClaim = true;
      }
    }
    if (!anyClaim)
      break;

    // Commit the round, then refresh the core states for the next one.
    for (unsigned int c = 0; c < n; ++c)
      if (claim[c] != kUnowned) {
        owner[c] = claim[c];
        coreE[claim[c]] += energy[c];
        coreSum[claim[c]] += bary[c] * energy[c];
      }
  }

  std::vector<std::vector<unsigned int>> members(n);
  for (unsigned int i = 0; i < n; ++i)
    if (owner[i] != kUnowned)
      members[owner[i]].push_back(i);
  for (unsigned int i = 0; i < n; ++i)
    if (!members[i].empty())
      groups.push_back(members[i]);
  // Whatever no core claimed stays a trackster of its own: no energy is dropped here,
  // the emit-side threshold in linkTracksters is the only place anything is discarded.
  for (unsigned int i = 0; i < n; ++i)
    if (owner[i] == kUnowned)
      groups.push_back({i});
}

void TracksterLinkingByCornetto::linkByUnionFind(const Inputs& input,
                                                 std::vector<std::vector<unsigned int>>& groups) const {
  const auto& tracksters = input.tracksters;
  const unsigned int n = tracksters.size();

  // Flat per-trackster snapshot (SoA shape, mirroring the future device layout).
  std::vector<float> eta(n), phi(n), energy(n), time(n), timeErr(n);
  std::vector<math::XYZVectorF> bary(n), axis(n);
  for (unsigned int i = 0; i < n; ++i) {
    const auto& ts = tracksters[i];
    bary[i] = ts.barycenter();
    axis[i] = ts.eigenvectors(0);
    eta[i] = ts.barycenter().eta();
    phi[i] = ts.barycenter().phi();
    energy[i] = ts.raw_energy();
    time[i] = ts.time();
    timeErr[i] = ts.timeError();
  }

  // (eta,phi) tile grid bounds the pair search (the same binning the GPU port
  // uses as a histogram-fill kernel): only neighboring phi tiles are scanned, so
  // the per-trackster candidate set stays local at PU200 densities.
  const int nPhiTiles = std::max(4, static_cast<int>(2. * M_PI / etaWindow_));
  const float phiTileWidth = 2.f * static_cast<float>(M_PI) / nPhiTiles;
  auto phiTile = [&](float p) {
    int t = static_cast<int>((p + static_cast<float>(M_PI)) / phiTileWidth);
    return std::min(std::max(t, 0), nPhiTiles - 1);
  };
  std::vector<std::vector<unsigned int>> tiles(nPhiTiles);
  std::vector<unsigned int> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&eta](unsigned int a, unsigned int b) { return eta[a] < eta[b]; });
  std::vector<unsigned int> rankOf(n);
  for (unsigned int oi = 0; oi < n; ++oi)
    rankOf[order[oi]] = oi;
  for (unsigned int oi = 0; oi < n; ++oi)
    tiles[phiTile(phi[order[oi]])].push_back(oi);

  std::vector<unsigned int> parent(n);
  std::iota(parent.begin(), parent.end(), 0u);

  for (unsigned int oi = 0; oi < n; ++oi) {
    const unsigned int i = order[oi];
    const int ti = phiTile(phi[i]);
    for (int dt = -1; dt <= 1; ++dt) {
      const int tj = (ti + dt + nPhiTiles) % nPhiTiles;
      for (unsigned int oj : tiles[tj]) {
        if (oj <= oi)
          continue;
        const unsigned int j = order[oj];
        if (eta[j] - eta[i] > etaWindow_)
          continue;
        if (bary[i].z() * bary[j].z() < 0.f)
          continue;  // same endcap only
        if (std::abs(reco::deltaPhi(phi[i], phi[j])) > etaWindow_)
          continue;

        // Anchor = higher-energy trackster; its axis defines the cone. Ties are
        // broken by the smaller INPUT index, which makes the pair test a function
        // of the unordered pair alone. Without the tie-break the anchor of an
        // equal-energy pair is whichever trackster the eta sort happened to put
        // first, and std::sort is not stable for equal eta, so the linking is not
        // reproducible; equal raw_energy is not exotic, single layer cluster
        // tracksters carrying one identical cell energy hit it. It also makes the
        // test order-free for a device port, where the two elements examining the
        // same pair have no common ordering to agree on.
        const bool iIsAnchor = (energy[i] > energy[j]) or (energy[i] == energy[j] and i < j);
        const unsigned int a = iIsAnchor ? i : j;
        const unsigned int o = iIsAnchor ? j : i;
        const auto D = bary[o] - bary[a];
        const float s = D.Dot(axis[a]);
        // Longitudinal window grows with the anchor's calorimeter depth: hadronic
        // showers reach deeper into CE-H and string their fragments out along the
        // axis (measured req_s median 15 cm in the CE-E front to 38 cm in the CE-H
        // back at PU200), while the transverse width stays roughly fixed.
        // maxLongitudinalSlope = 0 recovers a flat window.
        const float zrel = std::abs(bary[a].z()) - longitudinalZRef_;
        const float maxLong = maxLongitudinalDistance_ + maxLongitudinalSlope_ * (zrel > 0.f ? zrel : 0.f);
        if (std::abs(s) > maxLong)
          continue;
        const float dT2 = std::max(0.f, D.Mag2() - s * s);
        const float rT = transverseRadius0_ + transverseSlope_ * std::abs(s);
        if (dT2 > rT * rT)
          continue;

        // Timing gate (pileup rejection); applies only when both times are valid.
        if (timeErr[i] > 0.f && timeErr[j] > 0.f) {
          const float sigma2 = timeErr[i] * timeErr[i] + timeErr[j] * timeErr[j];
          const float dt = time[i] - time[j];
          if (dt * dt > timeCompatibilityNSigma_ * timeCompatibilityNSigma_ * sigma2)
            continue;
        }

        unite(parent, i, j);
      }
    }
  }

  // Emit EVERY component, singletons included: spectrum continuity by
  // construction (nothing is dropped, nothing is gated on energy).
  std::vector<std::vector<unsigned int>> components(n);
  for (unsigned int i = 0; i < n; ++i)
    components[findRoot(parent, i)].push_back(i);

  for (unsigned int r = 0; r < n; ++r)
    if (!components[r].empty())
      groups.push_back(std::move(components[r]));
}
