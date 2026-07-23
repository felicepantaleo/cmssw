#ifndef DataFormats_HGCalReco_interface_alpaka_TracksterComponentsDeviceCollection_h
#define DataFormats_HGCalReco_interface_alpaka_TracksterComponentsDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterComponentsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using TracksterComponentsDeviceCollection = PortableCollection<TracksterComponentsSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_HGCalReco_interface_alpaka_TracksterComponentsDeviceCollection_h
