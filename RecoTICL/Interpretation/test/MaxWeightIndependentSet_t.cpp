#include <algorithm>
#include <random>
#include <vector>

#include <catch2/catch_all.hpp>

#include "RecoTICL/Interpretation/interface/MaxWeightIndependentSet.h"

namespace {
  using Graph = std::vector<std::vector<unsigned int>>;

  Graph randomGraph(unsigned int n, float density, std::mt19937& gen) {
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    Graph adj(n);
    for (unsigned int a = 0; a < n; ++a)
      for (unsigned int b = a + 1; b < n; ++b)
        if (u01(gen) < density) {
          adj[a].push_back(b);
          adj[b].push_back(a);
        }
    for (auto& nb : adj)
      std::sort(nb.begin(), nb.end());
    return adj;
  }

  bool independent(const std::vector<bool>& selected, const Graph& adj) {
    for (unsigned int a = 0; a < adj.size(); ++a)
      for (unsigned int b : adj[a])
        if (selected[a] && selected[b])
          return false;
    return true;
  }

  float total(const std::vector<bool>& selected, const std::vector<float>& w) {
    float s = 0.f;
    for (unsigned int i = 0; i < w.size(); ++i)
      if (selected[i])
        s += w[i];
    return s;
  }

  // Best weight over all subsets of the positive-weight nodes.
  float bruteForce(const std::vector<float>& w, const Graph& adj) {
    const unsigned int n = w.size();
    float best = 0.f;
    for (unsigned long mask = 0; mask < (1UL << n); ++mask) {
      std::vector<bool> sel(n);
      for (unsigned int i = 0; i < n; ++i)
        sel[i] = (mask >> i) & 1UL;
      if (independent(sel, adj))
        best = std::max(best, total(sel, w));
    }
    return best;
  }
}  // namespace

TEST_CASE("The exact solver gives the maximum-weight independent set", "[MaxWeightIndependentSet]") {
  std::mt19937 gen(4242);
  std::uniform_real_distribution<float> uW(-2.f, 10.f);
  for (int trial = 0; trial < 200; ++trial) {
    const unsigned int n = 4 + trial % 13;
    const auto adj = randomGraph(n, 0.1f + 0.05f * (trial % 10), gen);
    std::vector<float> w(n);
    for (auto& x : w)
      x = uW(gen);
    ticl::MaxWeightIndependentSetStats stats;
    const auto sel = ticl::maxWeightIndependentSet(w, adj, 64, 1000000, stats);
    REQUIRE(independent(sel, adj));
    REQUIRE(total(sel, w) == Catch::Approx(bruteForce(w, adj)).margin(1e-4));
    REQUIRE(stats.budgetExhausted == 0);
    REQUIRE(stats.greedyComponents == 0);
    for (unsigned int i = 0; i < n; ++i)
      if (!(w[i] > 0.f))
        REQUIRE(!sel[i]);
  }
}

TEST_CASE("A component above the exact size is solved greedily", "[MaxWeightIndependentSet]") {
  // A path 0 - 1 - 2: the exact set is {0, 2} (weight 8), the greedy set is {1} (weight 5).
  const Graph adj{{1}, {0, 2}, {1}};
  const std::vector<float> w{4.f, 5.f, 4.f};
  ticl::MaxWeightIndependentSetStats stats;
  REQUIRE(ticl::maxWeightIndependentSet(w, adj, 64, 1000000, stats) == std::vector<bool>{true, false, true});
  REQUIRE(ticl::maxWeightIndependentSet(w, adj, 2, 1000000, stats) == std::vector<bool>{false, true, false});
  REQUIRE(stats.exactComponents == 1);
  REQUIRE(stats.greedyComponents == 1);
}

TEST_CASE("Equal weights go to the lower index", "[MaxWeightIndependentSet]") {
  const Graph adj{{1}, {0}};
  ticl::MaxWeightIndependentSetStats stats;
  REQUIRE(ticl::maxWeightIndependentSet({1.f, 1.f}, adj, 64, 1000000, stats) == std::vector<bool>{true, false});
}

TEST_CASE("The set is maximal at every search budget", "[MaxWeightIndependentSet]") {
  std::mt19937 gen(777);
  std::uniform_real_distribution<float> uW(-2.f, 10.f);
  for (unsigned long budget : {1UL, 3UL, 10UL, 100UL}) {
    for (int trial = 0; trial < 100; ++trial) {
      const unsigned int n = 20 + trial % 30;
      const auto adj = randomGraph(n, 0.2f, gen);
      std::vector<float> w(n);
      for (auto& x : w)
        x = uW(gen);
      ticl::MaxWeightIndependentSetStats stats;
      const auto sel = ticl::maxWeightIndependentSet(w, adj, 64, budget, stats);
      REQUIRE(independent(sel, adj));
      for (unsigned int a = 0; a < n; ++a) {
        if (!(w[a] > 0.f) || sel[a])
          continue;
        const bool covered = std::any_of(adj[a].begin(), adj[a].end(), [&](unsigned int b) { return sel[b]; });
        REQUIRE(covered);
      }
    }
  }
}
