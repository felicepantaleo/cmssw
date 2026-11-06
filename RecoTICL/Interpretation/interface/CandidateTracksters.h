#ifndef RecoTICL_Interpretation_CandidateTracksters_h
#define RecoTICL_Interpretation_CandidateTracksters_h

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "RecoTICL/Interpretation/interface/AssignmentMaps.h"
#include "RecoTICL/Interpretation/interface/ExpectedDeposit.h"
#include "RecoTICL/Interpretation/interface/TrackImpact.h"

namespace ticl {

  // How the layer clusters of a trackster that several candidates hold are given to the candidates.
  struct CandidateTrackstersParameters {
    // true: each layer cluster goes whole to one candidate; false: shared in proportion to the candidate energy.
    bool nearest = true;
    // Track radius [cm] at the HGCAL front and its growth per cm of depth.
    float radius0 = 3.f;
    float radiusSlope = 0.05f;
    // A charged hadron takes layer clusters while its raw energy stays within budgetScale x its expected raw deposit;
    // 0 sets no limit.
    float budgetScale = 0.f;
    ExpectedDeposit deposit{0.67f, 0.04f, 0.2f, 1.5f};
  };

  // The fraction of each layer cluster that each candidate takes. A candidate holds its tracksters of the collection
  // tracksterId and, for a jet-member track, the jet trackster of the track. A layer cluster that one candidate holds
  // goes to it with its fraction in the trackster. A layer cluster that several candidates hold: with nearest, the
  // charged candidates take it nearest track first, within the track radius and their energy budget; the rest goes
  // to the most energetic neutral candidate that holds it, else to no candidate when a budget is set, else to the
  // nearest track. impactOf(c) is the HGCAL impact of the track of candidate c.
  template <typename ImpactOf>
  std::vector<std::map<unsigned int, float>> candidateLayerClusterFractions(
      const std::vector<TICLCandidate> &candidates,
      edm::ProductID tracksterId,
      const std::vector<Trackster> &tracksters,
      const std::vector<int> &trackToTrackster,
      const std::vector<int> &trackMode,
      const std::vector<reco::CaloCluster> &layerClusters,
      ImpactOf &&impactOf,
      const CandidateTrackstersParameters &par) {
    const unsigned int nC = candidates.size();
    std::vector<std::vector<unsigned int>> holders(tracksters.size());
    std::vector<double> energySum(tracksters.size(), 0.);
    for (unsigned int c = 0; c < nC; ++c) {
      std::vector<unsigned int> mine;
      for (auto const &ptr : candidates[c].tracksters())
        if (ptr.isNonnull() && ptr.id() == tracksterId)
          mine.push_back(ptr.key());
      auto const track = candidates[c].trackPtr();
      if (track.isNonnull() && track.key() < trackMode.size() &&
          trackMode[track.key()] == static_cast<int>(TrackMode::kJetMember) && trackToTrackster[track.key()] >= 0)
        mine.push_back(trackToTrackster[track.key()]);
      std::sort(mine.begin(), mine.end());
      mine.erase(std::unique(mine.begin(), mine.end()), mine.end());
      for (auto t : mine) {
        holders[t].push_back(c);
        energySum[t] += candidates[c].energy();
      }
    }

    std::vector<std::map<unsigned int, float>> fractions(nC);
    std::vector<float> held(nC, 0.f);
    // Contested layer clusters: (trackster, vertex).
    std::vector<std::pair<unsigned int, unsigned int>> contested;
    for (unsigned int t = 0; t < tracksters.size(); ++t) {
      auto const &owners = holders[t];
      if (owners.empty() || energySum[t] <= 0.)
        continue;
      auto const &ts = tracksters[t];
      for (unsigned int k = 0; k < ts.vertices().size(); ++k) {
        const unsigned int lc = ts.vertices()[k];
        const float f = 1.f / ts.vertex_multiplicity()[k];
        if (owners.size() == 1) {
          fractions[owners[0]][lc] += f;
          held[owners[0]] += f * layerClusters[lc].energy();
        } else if (!par.nearest) {
          for (auto c : owners)
            fractions[c][lc] += f * candidates[c].energy() / energySum[t];
        } else {
          contested.emplace_back(t, k);
        }
      }
    }
    if (contested.empty())
      return fractions;

    auto capped = [&](unsigned int c) {
      const int pdg = std::abs(candidates[c].pdgId());
      return par.budgetScale > 0.f && pdg != 11 && pdg != 13;
    };
    // Proposals (distance, contested index, candidate) within the track radius, nearest first; and the nearest track.
    std::vector<std::tuple<float, unsigned int, unsigned int>> proposals;
    std::vector<int> nearestTrack(contested.size(), -1);
    for (unsigned int i = 0; i < contested.size(); ++i) {
      auto const &[t, k] = contested[i];
      auto const &cluster = layerClusters[tracksters[t].vertices()[k]];
      float best = std::numeric_limits<float>::infinity();
      for (auto c : holders[t]) {
        if (candidates[c].trackPtr().isNull())
          continue;
        auto const &impact = impactOf(c);
        const float d = impactTransverseDistance(impact, cluster.position());
        if (d < best) {
          best = d;
          nearestTrack[i] = c;
        }
        const float depth = std::max(0.f, std::abs(static_cast<float>(cluster.z())) - std::abs(impact.position.z()));
        if (d < par.radius0 + par.radiusSlope * depth)
          proposals.emplace_back(d, i, c);
      }
    }
    std::sort(proposals.begin(), proposals.end());
    std::vector<bool> assigned(contested.size(), false);
    for (auto const &[d, i, c] : proposals) {
      if (assigned[i])
        continue;
      auto const &[t, k] = contested[i];
      const unsigned int lc = tracksters[t].vertices()[k];
      const float f = 1.f / tracksters[t].vertex_multiplicity()[k];
      const float e = f * layerClusters[lc].energy();
      if (capped(c) && held[c] + e > par.budgetScale * par.deposit(candidates[c].trackPtr()->p()))
        continue;
      fractions[c][lc] += f;
      held[c] += e;
      assigned[i] = true;
    }
    for (unsigned int i = 0; i < contested.size(); ++i) {
      if (assigned[i])
        continue;
      auto const &[t, k] = contested[i];
      const unsigned int lc = tracksters[t].vertices()[k];
      const float f = 1.f / tracksters[t].vertex_multiplicity()[k];
      int neutral = -1;
      for (auto c : holders[t])
        if (candidates[c].trackPtr().isNull() && (neutral < 0 || candidates[c].energy() > candidates[neutral].energy()))
          neutral = c;
      if (neutral >= 0)
        fractions[neutral][lc] += f;
      else if (par.budgetScale <= 0.f && nearestTrack[i] >= 0)
        fractions[nearestTrack[i]][lc] += f;
    }
    return fractions;
  }

  // One trackster per candidate with layer clusters, from the fractions, and the candidate index of each. A candidate
  // with no layer cluster gets no trackster here and keeps its source tracksters: every candidate holds a track or a
  // trackster.
  inline std::vector<Trackster> tracksterOfFractions(const std::vector<std::map<unsigned int, float>> &fractions,
                                                     std::vector<int> &candidateIndex) {
    std::vector<Trackster> result;
    candidateIndex.clear();
    for (unsigned int c = 0; c < fractions.size(); ++c) {
      Trackster merged;
      for (auto const &[lc, f] : fractions[c])
        if (f > 0.f) {
          merged.vertices().push_back(lc);
          merged.vertex_multiplicity().push_back(1.f / f);
        }
      if (merged.vertices().empty())
        continue;
      result.push_back(std::move(merged));
      candidateIndex.push_back(c);
    }
    return result;
  }

}  // namespace ticl

#endif
