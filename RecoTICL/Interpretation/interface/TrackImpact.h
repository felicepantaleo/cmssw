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

  // Transverse offset (dx, dy) [cm] at the z of point from point to the line through origin along dir. False when
  // point is on the other side of z = 0 from sideZ.
  template <typename Origin, typename Direction, typename Point, typename T>
  bool lineOffset(const Origin &origin, const Direction &dir, float sideZ, const Point &point, T &dx, T &dy) {
    if (!(point.z() * sideZ > 0.f))
      return false;
    const float s = (point.z() - origin.z()) / dir.z();
    dx = origin.x() + s * dir.x() - point.x();
    dy = origin.y() + s * dir.y() - point.y();
    return true;
  }

  // Transverse distance [cm] at the z of point between point and the line from the impact along its direction.
  // Infinite when the impact is not valid or point is on the other side of z = 0.
  template <typename Point>
  float impactTransverseDistance(const TrackImpact &impact, const Point &point) {
    decltype(impact.position.x() + point.x()) dx, dy;
    if (!impact.valid || !lineOffset(impact.position, impact.direction, impact.position.z(), point, dx, dy))
      return std::numeric_limits<float>::infinity();
    return std::hypot(dx, dy);
  }

  // True when point is closer than r [cm] to the line, with the distance of impactTransverseDistance in dist. Most
  // points fail the cheap test: the distance is at least max(|dx|, |dy|).
  template <typename Point>
  bool impactTransverseDistanceBelow(const TrackImpact &impact, const Point &point, float r, float &dist) {
    decltype(impact.position.x() + point.x()) dx, dy;
    if (!impact.valid || !lineOffset(impact.position, impact.direction, impact.position.z(), point, dx, dy))
      return false;
    if (!(std::abs(dx) < r) || !(std::abs(dy) < r))
      return false;
    dist = std::hypot(dx, dy);
    return dist < r;
  }

  // Transverse distance [cm] at the z of point between point and the track: through the HGCAL impact when it is
  // valid, else along the straight line from the outermost state (from the reference point if the outer state is
  // missing). Infinite when point is on the other side of z = 0.
  template <typename Point>
  float trackTransverseDistance(const TrackImpact &impact, const reco::Track &track, const Point &point) {
    if (impact.valid)
      return impactTransverseDistance(impact, point);
    double dx, dy;
    const auto dir = track.outerOk() ? track.outerMomentum() : track.momentum();
    const auto origin = track.outerOk() ? track.outerPosition() : track.referencePoint();
    if (!lineOffset(origin, dir, dir.z(), point, dx, dy))
      return std::numeric_limits<float>::infinity();
    return static_cast<float>(std::hypot(dx, dy));
  }

}  // namespace ticl

#endif
