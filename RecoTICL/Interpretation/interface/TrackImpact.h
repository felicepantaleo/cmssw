#ifndef RecoTICL_Interpretation_TrackImpact_h
#define RecoTICL_Interpretation_TrackImpact_h

#include <array>
#include <cmath>
#include <limits>
#include <memory>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

namespace ticl {

  // Where a track enters HGCAL: the crossing of the front disk of its endcap and the momentum direction there.
  struct TrackImpact {
    GlobalPoint position;
    GlobalVector direction;
    bool valid = false;
  };

  // Propagates the track through the field from its outermost state (from its reference point if the outer state is
  // missing) to the HGCAL front disk of its endcap. The impact is not valid when the propagation fails.
  inline TrackImpact propagateToHGCalFront(const reco::Track &track,
                                           const MagneticField *field,
                                           const Propagator &propagator,
                                           const std::array<std::unique_ptr<GeomDet>, 2> &frontDisks) {
    TrackImpact impact;
    const auto fts = track.outerOk() ? trajectoryStateTransform::outerFreeState(track, field)
                                     : trajectoryStateTransform::initialFreeState(track, field);
    const auto tsos = propagator.propagate(fts, frontDisks[track.eta() > 0 ? 1 : 0]->surface());
    if (tsos.isValid()) {
      impact.position = tsos.globalPosition();
      impact.direction = tsos.globalMomentum();
      impact.valid = impact.direction.z() != 0.f;
    }
    return impact;
  }

  // Transverse distance [cm] at the z of point between point and the line from the impact along its direction.
  // Infinite when the impact is not valid or point is on the other side of z = 0.
  template <typename Point>
  float impactTransverseDistance(const TrackImpact &impact, const Point &point) {
    if (!impact.valid || !(point.z() * impact.position.z() > 0.f))
      return std::numeric_limits<float>::infinity();
    const float s = (point.z() - impact.position.z()) / impact.direction.z();
    return std::hypot(impact.position.x() + s * impact.direction.x() - point.x(),
                      impact.position.y() + s * impact.direction.y() - point.y());
  }

  // True when point is closer than r [cm] to the line, with the distance of impactTransverseDistance in dist. Most
  // points fail the cheap test: the distance is at least max(|dx|, |dy|).
  template <typename Point>
  bool impactTransverseDistanceBelow(const TrackImpact &impact, const Point &point, float r, float &dist) {
    if (!impact.valid || !(point.z() * impact.position.z() > 0.f))
      return false;
    const float s = (point.z() - impact.position.z()) / impact.direction.z();
    const auto dx = impact.position.x() + s * impact.direction.x() - point.x();
    const auto dy = impact.position.y() + s * impact.direction.y() - point.y();
    if (!(std::abs(dx) < r) || !(std::abs(dy) < r))
      return false;
    dist = std::hypot(dx, dy);
    return dist < r;
  }

}  // namespace ticl

#endif
