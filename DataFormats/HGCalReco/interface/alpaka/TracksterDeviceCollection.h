#ifndef DataFormats_HGCalReco_interface_alpaka_TracksterDeviceCollection_h
#define DataFormats_HGCalReco_interface_alpaka_TracksterDeviceCollection_h

#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using TracksterDeviceCollection = PortableCollection<::ticl::TracksterSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_HGCalReco_interface_alpaka_TracksterDeviceCollection_h
