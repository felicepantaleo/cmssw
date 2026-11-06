#ifndef DataFormats_HGCalReco_interface_TracksterPidSoA_h
#define DataFormats_HGCalReco_interface_TracksterPidSoA_h

#include <Eigen/Core>

#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  // The PID probabilities of each trackster of one collection, in the order of the collection. They replace the
  // probabilities of the trackster SoA, which the device does not fill.
  GENERATE_SOA_LAYOUT(TracksterPidLayout, SOA_EIGEN_COLUMN(TracksterIdProbabilities, idProbabilities))

  using TracksterPidSoA = TracksterPidLayout<>;

}  // namespace ticl

#endif  // DataFormats_HGCalReco_interface_TracksterPidSoA_h
