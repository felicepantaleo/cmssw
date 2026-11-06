#ifndef DataFormats_HGCalReco_interface_LayerClusterMaskHostCollection_h
#define DataFormats_HGCalReco_interface_LayerClusterMaskHostCollection_h

#include "DataFormats/HGCalReco/interface/LayerClusterMaskSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace ticl {

  using LayerClusterMaskHostCollection = PortableHostCollection<LayerClusterMaskSoA>;

}  // namespace ticl

#endif  // DataFormats_HGCalReco_interface_LayerClusterMaskHostCollection_h
