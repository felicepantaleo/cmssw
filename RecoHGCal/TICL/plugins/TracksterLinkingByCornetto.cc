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
      timeCompatibilityNSigma_(conf.getParameter<double>("timeCompatibilityNSigma")) {}

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

  // Eta-ordered sweep bounds the pair search (the GPU version replaces this with
  // an (eta,phi) tile grid; the tested pair set only grows, never shrinks).
  std::vector<unsigned int> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(), [&eta](unsigned int a, unsigned int b) { return eta[a] < eta[b]; });

  std::vector<unsigned int> parent(n);
  std::iota(parent.begin(), parent.end(), 0u);

  for (unsigned int oi = 0; oi < n; ++oi) {
    const unsigned int i = order[oi];
    for (unsigned int oj = oi + 1; oj < n; ++oj) {
      const unsigned int j = order[oj];
      if (eta[j] - eta[i] > etaWindow_)
        break;  // sorted in eta: no candidate farther on
      if (bary[i].z() * bary[j].z() < 0.f)
        continue;  // same endcap only
      if (std::abs(reco::deltaPhi(phi[i], phi[j])) > etaWindow_)
        continue;

      // Anchor = higher-energy trackster; its axis defines the cone.
      const unsigned int a = (energy[i] >= energy[j]) ? i : j;
      const unsigned int o = (a == i) ? j : i;
      const auto D = bary[o] - bary[a];
      const float s = D.Dot(axis[a]);
      if (std::abs(s) > maxLongitudinalDistance_)
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

  // Emit EVERY component, singletons included: spectrum continuity by
  // construction (nothing is dropped, nothing is gated on energy).
  std::vector<std::vector<unsigned int>> components(n);
  for (unsigned int i = 0; i < n; ++i)
    components[findRoot(parent, i)].push_back(i);

  for (unsigned int r = 0; r < n; ++r) {
    if (components[r].empty())
      continue;
    Trackster merged;
    merged.mergeTracksters(input.tracksters, components[r]);
    linkedResultTracksters.push_back({static_cast<unsigned int>(resultTracksters.size())});
    resultTracksters.push_back(merged);
    linkedTracksterIdToInputTracksterId.push_back(components[r]);
  }
}
