#ifndef DataFormats_HGCalReco_interface_alpaka_TracksterPidDeviceCollection_h
#define DataFormats_HGCalReco_interface_alpaka_TracksterPidDeviceCollection_h

#include "DataFormats/HGCalReco/interface/TracksterPidHostCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterPidSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using TracksterPidDeviceCollection = PortableCollection<::ticl::TracksterPidSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_HGCalReco_interface_alpaka_TracksterPidDeviceCollection_h
