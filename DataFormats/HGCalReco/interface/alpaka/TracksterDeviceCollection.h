#ifndef DataFormats_HGCalReco_interface_alpaka_TracksterDeviceCollection_h
#define DataFormats_HGCalReco_interface_alpaka_TracksterDeviceCollection_h

#include "DataFormats/Common/interface/RefProdVector.h"
#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  using TracksterDeviceCollection = PortableCollection<::ticl::TracksterSoA>;

  // An ordered list of trackster collections. Its order defines the global index of a trackster: the index in its
  // collection plus the sizes of the collections before it.
  using TracksterCollectionRefs = edm::RefProdVector<TracksterDeviceCollection>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // DataFormats_HGCalReco_interface_alpaka_TracksterDeviceCollection_h
