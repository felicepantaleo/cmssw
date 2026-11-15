#ifndef _toffset_fit_helper_crate_2_h_
#define _toffset_fit_helper_crate_2_h_

#include <vector>
#include <algorithm>
#include <iostream>
#include "TF1.h"

// adapted from Alicia Triana De Las Heras
// original on GitHub: https://gitlab.cern.ch/atriana/timing_jumps_precision/-/blob/f1f037460614e24019cfe3170e7b358781bc9107/TOffsetFitter_crate_2.h

class TOffsetFitter_crate_2 {
public:
  explicit TOffsetFitter_crate_2(size_t ncrates) : ncrates_(ncrates) {}

  /**
        @short sets the new values for the train start edges
    */
  void setTrainStartEdges(std::vector<int> v) {
    train_start_edges_.resize(v.size());
    for (size_t i = 0; i < v.size(); i++)
      train_start_edges_[i] = v[i];
  }

  /**
        @short implements the prediction of the time offset based on the bunch in which the event has been registered
        Parameters:
            par[0 ... nedges-1] -> bunch offsets
            par[nedges ... nedges+11]         -> global offsets from crates 0...11
            par[nedges + 12]     -> slope
        The prediction is y = crate_offset + train_offset + slope * bx
    */
  double operator()(double* x, double* par) {
    const int x_value = static_cast<int>(x[0]);
    const int bx = x_value % 3565;
    const int crate = x_value / 3565;
    const size_t nedges = train_start_edges_.size();
    const double global_offset = par[nedges + crate];
    const double slope = par[nedges + ncrates_];

    // Find the first edge strictly greater than bx.
    auto it = std::upper_bound(train_start_edges_.begin(), train_start_edges_.end(), bx);
    double offset = global_offset;
    if (it != train_start_edges_.begin()) {
      const size_t idx = std::distance(train_start_edges_.begin(), it) - 1;
      offset += par[idx];
    }

    return bx * slope + offset;
  }

  /**
        @short debug the train edges
    */
  void print() {
    std::cout << "[TOffsetFitter_crate][print] currently with " << train_start_edges_.size() << " train edges in memory"
              << std::endl;
    for (auto x : train_start_edges_)
      std::cout << x << " ";
    std::cout << std::endl;
  }

  size_t npars() { return ncrates_ + 1 + train_start_edges_.size(); }

private:
  size_t ncrates_;
  std::vector<int> train_start_edges_;
};

#endif
