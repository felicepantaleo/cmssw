#ifndef DataFormats_HGCalReco_interface_TracksterLinksHostCollection_h
#define DataFormats_HGCalReco_interface_TracksterLinksHostCollection_h

#include "DataFormats/HGCalReco/interface/TracksterLinksSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace ticl {

  using TracksterLinksHostCollection = PortableHostCollection<TracksterLinksSoA>;

}  // namespace ticl

#endif  // DataFormats_HGCalReco_interface_TracksterLinksHostCollection_h
