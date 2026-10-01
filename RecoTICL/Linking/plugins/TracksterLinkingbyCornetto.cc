#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>

#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TracksterLinkingbyCornetto.h"

using namespace ticl;

TracksterLinkingbyCornetto::TracksterLinkingbyCornetto(const edm::ParameterSet& conf,
                                                       edm::ConsumesCollector iC,
                                                       cms::Ort::ONNXRuntime const* onnxRuntime)
    : TracksterLinkingAlgoBase(conf, iC, onnxRuntime),
      etaWindow_(conf.getParameter<float>("etaWindow")),
      cone_{conf.getParameter<float>("maxBackwardDistance"),
            conf.getParameter<float>("maxLongitudinalDistance"),
            conf.getParameter<float>("transverseRadius0"),
            conf.getParameter<float>("transverseSlope")},
      useTimeGate_(conf.getParameter<float>("timeCompatibilityNSigma") > 0.f),
      timeNSigma2_(conf.getParameter<float>("timeCompatibilityNSigma") *
                   conf.getParameter<float>("timeCompatibilityNSigma")),
      timeFloor2_(conf.getParameter<float>("timeResolutionFloor") * conf.getParameter<float>("timeResolutionFloor")),
      typeVetoProbability_(conf.getParameter<float>("typeVetoProbability")),
      minEmittedEnergy_(conf.getParameter<float>("minEmittedEnergy")),
      minEmittedPt_(conf.getParameter<float>("minEmittedPt")) {
  const bool vetoValid = typeVetoProbability_ == 0.f || (typeVetoProbability_ >= 0.5f && typeVetoProbability_ <= 1.f);
  if (!(etaWindow_ > 0.f && etaWindow_ < std::numbers::pi_v<float>) || !(cone_.maxLongitudinal > 0.f) ||
      !(cone_.maxBackward >= 0.f) || !(cone_.radius0 >= 0.f) || !(cone_.slope >= 0.f) ||
      !(conf.getParameter<float>("timeResolutionFloor") >= 0.f) || !vetoValid)
    throw cms::Exception("Configuration")
        << "TracksterLinkingbyCornetto: invalid parameters, see the parameter descriptions. etaWindow " << etaWindow_
        << ", maxLongitudinalDistance " << cone_.maxLongitudinal << ", maxBackwardDistance " << cone_.maxBackward
        << ", transverseRadius0 " << cone_.radius0 << ", transverseSlope " << cone_.slope << ", timeResolutionFloor "
        << conf.getParameter<float>("timeResolutionFloor") << ", typeVetoProbability " << typeVetoProbability_;
}

bool TracksterLinkingbyCornetto::timeCompatible(
    float tAnchor, float errAnchor, float tOther, float errOther, float path) const {
  if (!useTimeGate_ || !(errAnchor > 0.f) || !(errOther > 0.f))
    return true;
  constexpr float kInvSpeedOfLight = 1.f / 29.9792458f;  // ns/cm
  const float dt = tOther - tAnchor - path * kInvSpeedOfLight;
  return dt * dt <= timeNSigma2_ * (errAnchor * errAnchor + errOther * errOther + timeFloor2_);
}

TracksterLinkingbyCornetto::Features TracksterLinkingbyCornetto::computeFeatures(const Inputs& input) const {
  const auto& tracksters = input.tracksters;
  const std::size_t n = tracksters.size();
  Features f;
  f.bary.resize(n);
  f.energy.resize(n);
  f.eta.resize(n);
  f.phi.resize(n);
  f.rad.resize(n);
  f.em.resize(n);
  f.valid.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    auto const& ts = tracksters[i];
    const auto& b = ts.barycenter();
    f.bary[i] = b;
    f.energy[i] = ts.raw_energy();
    f.valid[i] = std::isfinite(b.x()) && std::isfinite(b.y()) && std::isfinite(b.z()) && std::isfinite(f.energy[i]);
    f.eta[i] = b.eta();
    f.phi[i] = b.phi();
    f.rad[i] = std::sqrt(b.Mag2());
    float sum = 0.f;
    for (float v : ts.id_probabilities())
      sum += v;
    f.em[i] = sum > 0.f ? (ts.id_probability(Trackster::ParticleType::photon) +
                           ts.id_probability(Trackster::ParticleType::electron)) /
                              sum
                        : -1.f;
  }
  return f;
}

// A NaN position fails the cone and window tests. A NaN time error or EM probability
// disables the time gate or the type veto for that pair.
bool TracksterLinkingbyCornetto::compatible(const Inputs& input,
                                            const Features& f,
                                            unsigned int a,
                                            unsigned int b) const {
  if (!(f.bary[a].z() * f.bary[b].z() > 0.f))
    return false;  // same endcap only
  if (!(std::abs(f.eta[b] - f.eta[a]) <= etaWindow_) || !(std::abs(reco::deltaPhi(f.phi[b], f.phi[a])) <= etaWindow_))
    return false;
  const auto s = cornetto::separationInCone(cone_, f.bary[a], f.rad[a], f.bary[b]);
  if (!s)
    return false;
  auto const& ta = input.tracksters[a];
  auto const& tb = input.tracksters[b];
  if (!timeCompatible(ta.time(), ta.timeError(), tb.time(), tb.timeError(), *s))
    return false;
  if (typeVetoProbability_ > 0.f && f.em[a] >= 0.f && f.em[b] >= 0.f) {
    const float hi = typeVetoProbability_;
    const float lo = 1.f - typeVetoProbability_;
    if ((f.em[a] > hi && f.em[b] < lo) || (f.em[a] < lo && f.em[b] > hi))
      return false;
  }
  return true;
}

// Each trackster searches the tile of its endcap within the window of cornetto::searchWindow, which
// contains every anchor that can pass the cone.
std::vector<unsigned int> TracksterLinkingbyCornetto::findParents(const Inputs& input,
                                                                  const Features& f,
                                                                  const Tiles& tiles) const {
  const auto n = static_cast<unsigned int>(f.energy.size());
  // a ranks above b when it has more energy, or equal energy and a smaller index.
  const auto ranksAbove = [&f](unsigned int a, unsigned int b) {
    return f.energy[a] > f.energy[b] || (f.energy[a] == f.energy[b] && a < b);
  };
  std::vector<unsigned int> parents(n);
  for (unsigned int i = 0; i < n; ++i) {
    unsigned int best = i;
    if (f.valid[i] && f.rad[i] > 0.f) {
      const float absEta = std::abs(f.eta[i]);
      const auto window = cornetto::searchWindow(cone_, absEta, f.rad[i], etaWindow_);
      const float etaMin = std::clamp(absEta - window.dEta, TileConstants::minEta, TileConstants::maxEta);
      const float etaMax = std::clamp(absEta + window.dEta, TileConstants::minEta, TileConstants::maxEta);
      auto const& tile = tiles[f.bary[i].z() > 0.f ? 1 : 0];
      const auto box = tile.searchBoxEtaPhi(etaMin, etaMax, f.phi[i] - window.dPhi, f.phi[i] + window.dPhi);
      for (int etaBin = box[0]; etaBin <= box[1]; ++etaBin) {
        for (int phiBin = box[2]; phiBin <= box[3]; ++phiBin) {
          for (unsigned int j : tile[tile.globalBin(etaBin, phiBin % TileConstants::nPhiBins)]) {
            const bool candidate = j != i && ranksAbove(j, i) && (best == i || ranksAbove(j, best));
            if (candidate && compatible(input, f, j, i))
              best = j;
          }
        }
      }
    }
    parents[i] = best;
  }
  return parents;
}

std::vector<std::vector<unsigned int>> TracksterLinkingbyCornetto::buildGroups(
    const Inputs& input, const Features& f, const std::vector<unsigned int>& parents) const {
  const auto n = static_cast<unsigned int>(parents.size());

  // Pointer jumping to the root. Each round reads the result of the previous round only.
  std::vector<unsigned int> root(parents);
  std::vector<unsigned int> next(n);
  for (bool changed = true; changed;) {
    changed = false;
    for (unsigned int i = 0; i < n; ++i) {
      next[i] = root[root[i]];
      changed |= next[i] != root[i];
    }
    root.swap(next);
  }

  // A trackster that is not compatible with its root leaves it.
  std::vector<std::vector<unsigned int>> members(n);
  for (unsigned int i = 0; i < n; ++i)
    members[(root[i] == i || compatible(input, f, root[i], i)) ? root[i] : i].push_back(i);

  std::vector<std::vector<unsigned int>> byMin(n);
  for (auto& m : members)
    if (!m.empty())
      byMin[m.front()] = std::move(m);  // filled ascending, front() is the min
  std::vector<std::vector<unsigned int>> groups;
  for (auto& g : byMin)
    if (!g.empty())
      groups.push_back(std::move(g));
  return groups;
}

void TracksterLinkingbyCornetto::linkTracksters(
    const Inputs& input,
    std::vector<Trackster>& resultTracksters,
    std::vector<std::vector<unsigned int>>& linkedResultTracksters,
    std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) {
  const Features f = computeFeatures(input);

  // About 200 kB, so on the heap rather than on the stack.
  const auto tiles = std::make_unique<Tiles>();
  for (unsigned int i = 0; i < f.energy.size(); ++i)
    if (f.valid[i])
      (*tiles)[f.bary[i].z() > 0.f ? 1 : 0].fill(f.eta[i], f.phi[i], i);

  auto groups = buildGroups(input, f, findParents(input, f, *tiles));

  resultTracksters.reserve(resultTracksters.size() + groups.size());
  linkedResultTracksters.reserve(linkedResultTracksters.size() + groups.size());
  linkedTracksterIdToInputTracksterId.reserve(linkedTracksterIdToInputTracksterId.size() + groups.size());
  for (auto& group : groups) {
    Trackster merged;
    merged.mergeTracksters(input.tracksters, group);
    // The merged trackster has no barycenter yet, so the pt uses the energy-weighted member barycenter.
    math::XYZVectorF weighted;
    for (unsigned int i : group)
      weighted += input.tracksters[i].barycenter() * input.tracksters[i].raw_energy();
    const float pt = merged.raw_energy() / std::cosh(weighted.eta());
    if (!(merged.raw_energy() >= minEmittedEnergy_) || !(pt >= minEmittedPt_))
      continue;
    linkedResultTracksters.push_back({static_cast<unsigned int>(resultTracksters.size())});
    resultTracksters.push_back(std::move(merged));
    linkedTracksterIdToInputTracksterId.push_back(std::move(group));
  }
}
