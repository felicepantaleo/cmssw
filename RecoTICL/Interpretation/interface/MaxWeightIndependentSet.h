#ifndef RecoTICL_Interpretation_MaxWeightIndependentSet_h
#define RecoTICL_Interpretation_MaxWeightIndependentSet_h

#include <vector>

namespace ticl {

  struct MaxWeightIndependentSetStats {
    unsigned long exactComponents = 0;
    // Exact components that reached the search budget: the best set found is kept.
    unsigned long budgetExhausted = 0;
    unsigned long greedyComponents = 0;
  };

  // Maximum-weight independent set of the nodes with a positive weight. adjacency[i] lists the neighbours of node i,
  // sorted and without repetition. Each connected component is solved exactly by branch and bound, up to
  // maxExactSize (at most 64) nodes and searchBudget calls; a larger component is solved greedily by weight. Ties go
  // to the lower index. The set is maximal: every positive-weight node is selected or has a selected neighbour.
  // Returns the selection flag of each node.
  std::vector<bool> maxWeightIndependentSet(const std::vector<float>& weights,
                                            const std::vector<std::vector<unsigned int>>& adjacency,
                                            unsigned int maxExactSize,
                                            unsigned long searchBudget,
                                            MaxWeightIndependentSetStats& stats);

}  // namespace ticl

#endif
