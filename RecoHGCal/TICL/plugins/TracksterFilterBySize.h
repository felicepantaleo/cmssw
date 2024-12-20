// Author: Marco Rovere - marco.rovere@cern.ch
// Date: 11/2018

#ifndef RecoHGCal_TICL_TracksterFilterBySize_H__
#define RecoHGCal_TICL_TracksterFilterBySize_H__

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "TracksterFilterBase.h"

#include <memory>
#include <utility>

// Filter clusters that belong to a specific algorithm
namespace ticl {
  class TracksterFilterBySize final : public TracksterFilterBase {
  public:
    TracksterFilterBySize(const edm::ParameterSet& ps)
        : TracksterFilterBase(ps), max_num_layerclusters_(ps.getParameter<int>("max_num_layerclusters")), min_num_layerclusters_(ps.getParameter<int>("min_num_layerclusters")) {}
    ~TracksterFilterBySize() override {}

    void filter(const std::vector<ticl::Trackster>& tracksters,
                std::vector<float>& trackstersMask) const override {
      for (size_t i = 0; i < tracksters.size(); i++) {
        if (tracksters[i].vertices().size() > max_num_layerclusters_ and tracksters[i].vertices().size() < min_num_layerclusters_) {
          trackstersMask[i] = 0.f;
        }
      }
    }

  private:
    unsigned int max_num_layerclusters_;
    unsigned int min_num_layerclusters_;

  };
}  // namespace ticl

#endif
