#include "RecoTICL/Inference/interface/TracksterPointCloudInputs.h"

namespace ticl {

  TracksterPointCloud::PrimaryVertex leadingPrimaryVertex(const std::vector<reco::Vertex>& vertices) {
    TracksterPointCloud::PrimaryVertex pv;
    for (auto const& v : vertices) {
      if (!v.isValid() || v.isFake())
        continue;
      pv.x = v.x();
      pv.y = v.y();
      pv.z = v.z();
      pv.t = v.t();
      pv.tError = v.tError() > 0. ? v.tError() : -1.f;
      pv.valid = true;
      break;
    }
    return pv;
  }

  int countValidVertices(const std::vector<reco::Vertex>& vertices) {
    int n = 0;
    for (auto const& v : vertices)
      if (v.isValid() && !v.isFake())
        ++n;
    return n;
  }

  std::pair<float, float> mtdTrackTime(const MtdHostCollection* timing, unsigned int track) {
    if (timing == nullptr)
      return {0.f, -1.f};
    auto const& view = timing->const_view();
    return {view.time()[track], view.timeErr()[track]};
  }

}  // namespace ticl
