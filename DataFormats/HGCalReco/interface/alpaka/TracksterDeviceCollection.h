#ifndef DataFormats_HGCalReco_interface_alpaka_TracksterDeviceCollection_h
#define DataFormats_HGCalReco_interface_alpaka_TracksterDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using TracksterDeviceCollection = PortableCollection<TracksterSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_HGCalReco_interface_alpaka_TracksterDeviceCollection_h
