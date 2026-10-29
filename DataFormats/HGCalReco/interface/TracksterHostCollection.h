#ifndef DataFormats_HGCalReco_interface_TracksterHostCollection_h
#define DataFormats_HGCalReco_interface_TracksterHostCollection_h

#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace ticl {

  using TracksterHostCollection = PortableHostCollection<TracksterSoA>;

}  // namespace ticl

#endif  // DataFormats_HGCalReco_interface_TracksterHostCollection_h
