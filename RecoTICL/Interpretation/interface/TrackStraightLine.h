#ifndef RecoTICL_Interpretation_TrackStraightLine_h
#define RecoTICL_Interpretation_TrackStraightLine_h

#include <cmath>
#include <limits>

#include "DataFormats/TrackReco/interface/Track.h"

namespace ticl {

  // Transverse distance [cm] at the z of point between point and the straight-line extension of the track from its
  // outermost state (from its reference point if the outer state is missing). Infinite when point is on the other
  // side of z = 0 from the track.
  template <typename Point>
  float straightLineTransverseDistance(const reco::Track &track, const Point &point) {
    const auto dir = track.outerOk() ? track.outerMomentum() : track.momentum();
    const auto origin = track.outerOk() ? track.outerPosition() : track.referencePoint();
    if (!(point.z() * dir.z() > 0.f))
      return std::numeric_limits<float>::infinity();
    const float s = (point.z() - origin.z()) / dir.z();
    return static_cast<float>(std::hypot(origin.x() + s * dir.x() - point.x(), origin.y() + s * dir.y() - point.y()));
  }

}  // namespace ticl

#endif
