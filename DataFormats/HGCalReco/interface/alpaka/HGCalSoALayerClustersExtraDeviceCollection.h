#ifndef DataFormats_HGCalReco_interface_alpaka_HGCalSoALayerClustersExtraDeviceCollection_h
#define DataFormats_HGCalReco_interface_alpaka_HGCalSoALayerClustersExtraDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoALayerClustersExtra.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using HGCalSoALayerClustersExtraDeviceCollection = PortableCollection<HGCalSoALayerClustersExtra>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_HGCalReco_interface_alpaka_HGCalSoALayerClustersExtraDeviceCollection_h
