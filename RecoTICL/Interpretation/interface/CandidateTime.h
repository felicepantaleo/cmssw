#ifndef RecoTICL_Interpretation_CandidateTime_h
#define RecoTICL_Interpretation_CandidateTime_h

#include <vector>

#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"

class GlobalTrackingGeometry;
class HGCalDDDConstants;
class MagneticField;
class Propagator;

namespace ticl {

  // Path length [cm] of the track from its reference point to the HGCAL disk at |z| = zAbs on its side, through its
  // inner and outer states. 0 when a state is missing or a propagation fails.
  float trackPathLengthToHGCal(const reco::Track &track,
                               float zAbs,
                               const MagneticField *field,
                               const Propagator &propagator,
                               const GlobalTrackingGeometry &trackingGeometry,
                               const HGCalDDDConstants &hgcons);

  struct CandidateTimeParameters {
    bool useMTDTiming;
    bool useTimingAverage;
    float timingQualityThreshold;
  };

  // Time of each candidate: the error-weighted mean of the times of its tracksters, corrected for the flight path (the
  // MTD path of the track when it has an MTD time, else the track path, else the straight line from the origin), then
  // combined with the MTD time of the track when the association quality passes. The track of a candidate is a Ptr
  // into the collection that the timing view covers.
  void assignTimeToCandidates(std::vector<TICLCandidate> &candidates,
                              const MtdHostCollection::ConstView &timing,
                              const CandidateTimeParameters &parameters,
                              const MagneticField *field,
                              const Propagator &propagator,
                              const GlobalTrackingGeometry &trackingGeometry,
                              const HGCalDDDConstants &hgcons);

}  // namespace ticl

#endif
