// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 12/2024

#ifndef RecoHGCal_TICL_TracksterFilterByAlgo_H__
#define RecoHGCal_TICL_TracksterFilterByAlgo_H__

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "TracksterFilterBase.h"

#include <memory>
#include <utility>

// Filter clusters that belong to a specific algorithm
namespace ticl {
  class TracksterFilterByAlgo final : public TracksterFilterBase {
  public:
    TracksterFilterByAlgo(const edm::ParameterSet& ps)
        : TracksterFilterBase(ps), iteration_number_(ps.getParameter<std::vector<int>>("iteration_number")) {}
    ~TracksterFilterByAlgo() override {}

    void filter(const std::vector<ticl::Trackster>& tracksters,
                std::vector<float>& trackstersMask) const override {
      for (size_t i = 0; i < tracksters.size(); i++) {
        if (find(iteration_number_.begin(), iteration_number_.end(), tracksters[i].ticlIteration()) == iteration_number_.end()) {
          trackstersMask[i] = 0.;
        }
      }
    }

  private:
    std::vector<int> iteration_number_;
  };
}  // namespace ticl

#endif
