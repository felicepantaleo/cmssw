// Author: Felice Pantaleo - felice.pantaleo@cern.ch

#ifndef RecoHGCal_TICL_TracksterFilterBase_H__
#define RecoHGCal_TICL_TracksterFilterBase_H__

#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"


#include <memory>
#include <vector>

namespace edm {
  class ParameterSet;
}

namespace ticl {
  class Trackster;
}

namespace ticl {
  class TracksterFilterBase {
  public:
    explicit TracksterFilterBase(const edm::ParameterSet&) {}
    virtual ~TracksterFilterBase() {}

    virtual void filter(const std::vector<ticl::Trackster>& tracksters,
                        std::vector<float>& trackstersMask) const = 0;
  };
}  // namespace ticl

#endif
