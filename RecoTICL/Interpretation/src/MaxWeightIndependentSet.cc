#include "RecoTICL/Interpretation/interface/MaxWeightIndependentSet.h"

#include <algorithm>
#include <bit>
#include <cstdint>

namespace ticl {

  std::vector<bool> maxWeightIndependentSet(const std::vector<float>& weights,
                                            const std::vector<std::vector<unsigned int>>& adjacency,
                                            unsigned int maxExactSize,
                                            unsigned long searchBudget,
                                            MaxWeightIndependentSetStats& stats) {
    const unsigned int n = weights.size();
    std::vector<bool> selected(n, false);
    std::vector<bool> seen(n, false);
    for (unsigned int s0 = 0; s0 < n; ++s0) {
      if (seen[s0] || !(weights[s0] > 0.f))
        continue;
      std::vector<unsigned int> nodes, stack{s0};
      seen[s0] = true;
      while (!stack.empty()) {
        const unsigned int v = stack.back();
        stack.pop_back();
        nodes.push_back(v);
        for (unsigned int u : adjacency[v])
          if (!seen[u] && weights[u] > 0.f) {
            seen[u] = true;
            stack.push_back(u);
          }
      }
      std::sort(nodes.begin(), nodes.end(), [&weights](unsigned int a, unsigned int b) {
        return weights[a] > weights[b] || (weights[a] == weights[b] && a < b);
      });
      if (nodes.size() > maxExactSize) {
        ++stats.greedyComponents;
        std::vector<bool> blocked(n, false);
        for (unsigned int v : nodes) {
          if (blocked[v])
            continue;
          selected[v] = true;
          for (unsigned int u : adjacency[v])
            blocked[u] = true;
        }
        continue;
      }
      // Bit k of a mask is nodes[k]: the heaviest remaining node is the lowest set bit.
      const unsigned int k = nodes.size();
      std::vector<uint64_t> nbr(k, 0);
      for (unsigned int a = 0; a < k; ++a)
        for (unsigned int b = 0; b < k; ++b)
          if (a != b && std::binary_search(adjacency[nodes[a]].begin(), adjacency[nodes[a]].end(), nodes[b]))
            nbr[a] |= uint64_t(1) << b;
      const uint64_t all = (k == 64) ? ~uint64_t(0) : ((uint64_t(1) << k) - 1);
      // Incumbent: the greedy set by weight.
      float bestW = 0.f;
      uint64_t bestSet = 0;
      for (uint64_t c = all; c;) {
        const int v = std::countr_zero(c);
        bestSet |= uint64_t(1) << v;
        bestW += weights[nodes[v]];
        c &= ~(uint64_t(1) << v) & ~nbr[v];
      }
      // Upper bound: a cover of the candidates by cliques, heaviest node first. An independent set takes at most one
      // node of each clique.
      auto bound = [&](uint64_t cand) {
        float s = 0.f;
        while (cand) {
          const int v = std::countr_zero(cand);
          uint64_t clique = uint64_t(1) << v;
          uint64_t common = nbr[v] & cand;
          s += weights[nodes[v]];
          while (common) {
            const int u = std::countr_zero(common);
            clique |= uint64_t(1) << u;
            common &= nbr[u];
          }
          cand &= ~clique;
        }
        return s;
      };
      unsigned long calls = 0;
      auto search = [&](auto&& self, uint64_t cand, uint64_t chosen, float w) -> void {
        if (++calls > searchBudget || w + bound(cand) <= bestW)
          return;
        if (!cand) {
          bestW = w;
          bestSet = chosen;
          return;
        }
        const int v = std::countr_zero(cand);
        const uint64_t bit = uint64_t(1) << v;
        self(self, cand & ~bit & ~nbr[v], chosen | bit, w + weights[nodes[v]]);
        self(self, cand & ~bit, chosen, w);
      };
      search(search, all, 0, 0.f);
      ++stats.exactComponents;
      if (calls > searchBudget)
        ++stats.budgetExhausted;
      // A search that reached the budget can leave a set that is not maximal: add the free nodes, heaviest first.
      uint64_t free = all;
      for (uint64_t c = bestSet; c; c &= c - 1)
        free &= ~(uint64_t(1) << std::countr_zero(c)) & ~nbr[std::countr_zero(c)];
      while (free) {
        const int v = std::countr_zero(free);
        bestSet |= uint64_t(1) << v;
        free &= ~(uint64_t(1) << v) & ~nbr[v];
      }
      for (uint64_t c = bestSet; c; c &= c - 1)
        selected[nodes[std::countr_zero(c)]] = true;
    }
    return selected;
  }

}  // namespace ticl
