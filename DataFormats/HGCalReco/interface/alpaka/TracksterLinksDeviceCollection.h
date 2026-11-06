#ifndef DataFormats_HGCalReco_interface_alpaka_TracksterLinksDeviceCollection_h
#define DataFormats_HGCalReco_interface_alpaka_TracksterLinksDeviceCollection_h

#include "DataFormats/HGCalReco/interface/TracksterLinksHostCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterLinksSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using TracksterLinksDeviceCollection = PortableCollection<::ticl::TracksterLinksSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_HGCalReco_interface_alpaka_TracksterLinksDeviceCollection_h
