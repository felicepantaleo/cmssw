#ifndef CornettoStudy_Probe_DeepTrackMatch_h
#define CornettoStudy_Probe_DeepTrackMatch_h
// Study helper, not for integration. The particle that a track follows (deep match): of the reco-to-truth matches of
// the track with purity >= 0.5 to an assignable root, the ones that are not an ancestor of another such match; of
// these the highest purity, then the lowest energy. -1 when there is none.

#include <algorithm>
#include <cstddef>
#include <vector>

#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

namespace cornettostudy {
  class DeepTrackMatch {
  public:
    using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;

    DeepTrackMatch(const truth::Graph& graph,
                   const MapType::Type& tkR2t,
                   const std::vector<unsigned int>& assignableRoots)
        : graph_(graph), tkR2t_(tkR2t), assignable_(graph.particles().size(), 0) {
      for (const unsigned int id : assignableRoots)
        if (id < assignable_.size())
          assignable_[id] = 1;
    }

    // The deep match of the track with index key, and its purity (-1 without a match).
    int operator()(std::size_t key, float& purity) const {
      purity = -1.f;
      pool_.clear();
      if (key >= tkR2t_.size())
        return -1;
      const std::size_t nP = assignable_.size();
      for (auto const& m : tkR2t_[key])
        if (m.index() < nP && assignable_[m.index()] && 1.f - m.score() >= 0.5f)
          pool_.push_back(m.index());
      int best = -1;
      float bestPurity = -1.f;
      double bestE = 0.;
      for (auto const& m : tkR2t_[key]) {
        const auto p = m.index();
        if (p >= nP || !assignable_[p] || 1.f - m.score() < 0.5f)
          continue;
        bool ancestor = false;
        for (auto const& d : graph_.particle(p).descendants())
          if (d.id() != p && std::find(pool_.begin(), pool_.end(), d.id()) != pool_.end()) {
            ancestor = true;
            break;
          }
        if (ancestor)
          continue;
        const float pur = 1.f - m.score();
        const double e = graph_.particles()[p].momentum.energy();
        if (best < 0 || pur > bestPurity || (pur == bestPurity && e < bestE)) {
          best = static_cast<int>(p);
          bestPurity = pur;
          bestE = e;
        }
      }
      purity = bestPurity;
      return best;
    }

  private:
    const truth::Graph& graph_;
    const MapType::Type& tkR2t_;
    std::vector<char> assignable_;
    mutable std::vector<unsigned int> pool_;
  };
}  // namespace cornettostudy

#endif
