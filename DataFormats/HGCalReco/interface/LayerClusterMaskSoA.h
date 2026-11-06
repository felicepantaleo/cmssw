#ifndef DataFormats_HGCalReco_interface_LayerClusterMaskSoA_h
#define DataFormats_HGCalReco_interface_LayerClusterMaskSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace ticl {

  // The mask of each layer cluster, in the order of the merged layer clusters: 1 when an iteration can use it, 0 when
  // it cannot. The values are those of the std::vector<float> masks of TICL.
  GENERATE_SOA_LAYOUT(LayerClusterMaskLayout, SOA_COLUMN(float, mask))

  using LayerClusterMaskSoA = LayerClusterMaskLayout<>;

}  // namespace ticl

#endif  // DataFormats_HGCalReco_interface_LayerClusterMaskSoA_h
