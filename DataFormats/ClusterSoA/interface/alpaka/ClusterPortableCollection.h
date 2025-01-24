#ifndef DataFormats_Portable_interface_alpaka_ClusterPortableCollection_h
#define DataFormats_Portable_interface_alpaka_ClusterPortableCollection_h

#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// Generic headers for portable collections
#include "DataFormats/Portable/interface/PortableCollection.h"

// The SoAs we want to combine
#include "DataFormats/ClusterSoA/interface/CaloClusterSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // However, if you have multiple SoAs to unify, use PortableMultiCollection:
  using CaloClusterSoACollection = PortableMultiCollection<Device,
                                                           Position4D_Energy_SoA,
                                                           Position4D_Energy_Errors_SoA,
                                                           CaloClusterExtra_SoA
                                                           >;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_Portable_interface_alpaka_ClusterPortableCollection_h
