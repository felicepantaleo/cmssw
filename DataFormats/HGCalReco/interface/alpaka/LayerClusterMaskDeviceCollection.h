#ifndef DataFormats_HGCalReco_interface_alpaka_LayerClusterMaskDeviceCollection_h
#define DataFormats_HGCalReco_interface_alpaka_LayerClusterMaskDeviceCollection_h

#include "DataFormats/HGCalReco/interface/LayerClusterMaskHostCollection.h"
#include "DataFormats/HGCalReco/interface/LayerClusterMaskSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using LayerClusterMaskDeviceCollection = PortableCollection<::ticl::LayerClusterMaskSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_HGCalReco_interface_alpaka_LayerClusterMaskDeviceCollection_h
