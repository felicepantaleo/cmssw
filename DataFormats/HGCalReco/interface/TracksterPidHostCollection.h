#ifndef DataFormats_HGCalReco_interface_TracksterPidHostCollection_h
#define DataFormats_HGCalReco_interface_TracksterPidHostCollection_h

#include <Eigen/Core>

#include "DataFormats/HGCalReco/interface/TracksterPidSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace ticl {

  using TracksterPidHostCollection = PortableHostCollection<TracksterPidSoA>;

}  // namespace ticl

#endif  // DataFormats_HGCalReco_interface_TracksterPidHostCollection_h
