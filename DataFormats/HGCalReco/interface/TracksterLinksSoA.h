#ifndef DataFormats_HGCalReco_interface_TracksterLinksSoA_h
#define DataFormats_HGCalReco_interface_TracksterLinksSoA_h

#include <cstdint>

#include "DataFormats/TICL/interface/AssociationMap.h"

namespace ticl {

  // The input tracksters of each linked trackster, as global indices in the ordered list of the input collections.
  // The nested template is named directly: through the alias ticl::AssociationMap the dictionary gets another name.
  using TracksterLinksSoA = AssociationMapLayout<uint32_t, uint32_t>::Layout<>;

}  // namespace ticl

#endif  // DataFormats_HGCalReco_interface_TracksterLinksSoA_h
