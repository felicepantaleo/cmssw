#include "RecoTICL/Interpretation/interface/CandidateTime.h"

#include <cmath>
#include <memory>

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/GlobalTrackingGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

namespace ticl {

  float trackPathLengthToHGCal(const reco::Track &track,
                               float zAbs,
                               const MagneticField *field,
                               const Propagator &propagator,
                               const GlobalTrackingGeometry &trackingGeometry,
                               const HGCalDDDConstants &hgcons) {
    if (!track.innerOk() || !track.outerOk())
      return 0.f;
    const auto &fts_inn = trajectoryStateTransform::innerFreeState(track, field);
    const auto &fts_out = trajectoryStateTransform::outerFreeState(track, field);
    const auto &surf_inn = trajectoryStateTransform::innerStateOnSurface(track, trackingGeometry, field);
    const auto &surf_out = trajectoryStateTransform::outerStateOnSurface(track, trackingGeometry, field);

    Basic3DVector<float> pos(track.referencePoint());
    Basic3DVector<float> mom(track.momentum());
    FreeTrajectoryState stateAtBeamspot{GlobalPoint(pos), GlobalVector(mom), track.charge(), field};

    float pathlength = propagator.propagateWithPath(stateAtBeamspot, surf_inn.surface()).second;
    if (!pathlength)
      return 0.f;
    const auto &t_inn_out = propagator.propagateWithPath(fts_inn, surf_out.surface());
    if (!t_inn_out.first.isValid())
      return 0.f;
    pathlength += t_inn_out.second;
    std::pair<float, float> rMinMax = hgcons.rangeR(zAbs, true);
    const float zSide = (track.eta() > 0) ? zAbs : -zAbs;
    const auto &disk = std::make_unique<GeomDet>(
        Disk::build(Disk::PositionType(0, 0, zSide),
                    Disk::RotationType(),
                    SimpleDiskBounds(rMinMax.first, rMinMax.second, zSide - 0.5f, zSide + 0.5f))
            .get());
    const auto &tsos = propagator.propagateWithPath(fts_out, disk->surface());
    if (!tsos.first.isValid())
      return 0.f;
    return pathlength + tsos.second;
  }

  void assignTimeToCandidates(std::vector<TICLCandidate> &candidates,
                              const MtdHostCollection::ConstView &timing,
                              const CandidateTimeParameters &parameters,
                              const MagneticField *field,
                              const Propagator &propagator,
                              const GlobalTrackingGeometry &trackingGeometry,
                              const HGCalDDDConstants &hgcons) {
    constexpr float c_light = CLHEP::c_light * CLHEP::ns / CLHEP::cm;
    // Lower limit [ns] of the time error of the tracksters.
    constexpr float timeRes = 0.02f;
    for (auto &cand : candidates) {
      float beta = 1;
      float time = 0.f;
      float invTimeErr = 0.f;
      float timeErr = -1.f;

      const int trackIndex = cand.trackPtr().isNonnull() ? static_cast<int>(cand.trackPtr().key()) : -1;
      for (const auto &tr : cand.tracksters()) {
        if (tr->timeError() > 0) {
          const auto invTimeESq = pow(tr->timeError(), -2);
          const auto x = tr->barycenter().X();
          const auto y = tr->barycenter().Y();
          const auto z = tr->barycenter().Z();
          auto path = std::sqrt(x * x + y * y + z * z);
          if (trackIndex != -1) {
            if (parameters.useMTDTiming and timing.timeErr()[trackIndex] > 0) {
              const auto xMtd = timing.posInMTD_x()[trackIndex];
              const auto yMtd = timing.posInMTD_y()[trackIndex];
              const auto zMtd = timing.posInMTD_z()[trackIndex];
              beta = timing.beta()[trackIndex];
              path = std::sqrt((x - xMtd) * (x - xMtd) + (y - yMtd) * (y - yMtd) + (z - zMtd) * (z - zMtd)) +
                     timing.pathLength()[trackIndex];
            } else {
              const float pathLength =
                  trackPathLengthToHGCal(*cand.trackPtr(), std::abs(z), field, propagator, trackingGeometry, hgcons);
              if (pathLength) {
                path = pathLength;
              }
            }
          }
          time += (tr->time() - path / (beta * c_light)) * invTimeESq;
          invTimeErr += invTimeESq;
        }
      }
      if (invTimeErr > 0) {
        time = time / invTimeErr;
        timeErr = sqrt(1.f / invTimeErr);
        if (timeErr < timeRes)
          timeErr = timeRes;
        cand.setTime(time, timeErr);
      }

      if (parameters.useMTDTiming and cand.charge() and trackIndex != -1) {
        const bool assocQuality = timing.MVAquality()[trackIndex] > parameters.timingQualityThreshold;
        if (assocQuality) {
          const auto timeHGC = cand.time();
          const auto timeEHGC = cand.timeError();
          const auto timeMTD = timing.time0()[trackIndex];
          const auto timeEMTD = timing.time0Err()[trackIndex];

          if (parameters.useTimingAverage && (timeEMTD > 0 && timeEHGC > 0)) {
            const auto invTimeESqHGC = pow(timeEHGC, -2);
            const auto invTimeESqMTD = pow(timeEMTD, -2);
            timeErr = 1.f / (invTimeESqHGC + invTimeESqMTD);
            time = (timeHGC * invTimeESqHGC + timeMTD * invTimeESqMTD) * timeErr;
            timeErr = sqrt(timeErr);
          } else if (timeEMTD > 0) {
            time = timeMTD;
            timeErr = timeEMTD;
          }
        }
        cand.setTime(time, timeErr);
        cand.setMTDTime(timing.time()[trackIndex], timing.timeErr()[trackIndex]);
      }
    }
  }

}  // namespace ticl
