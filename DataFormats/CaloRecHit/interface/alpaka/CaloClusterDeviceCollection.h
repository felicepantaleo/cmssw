#ifndef DataFormats_CaloRecHit_interface_alpaka_CaloClusterDeviceCollection_h
#define DataFormats_CaloRecHit_interface_alpaka_CaloClusterDeviceCollection_h

#include <Eigen/Core>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterSoA.h"
#include <alpaka/alpaka.hpp>

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using CaloClusterDeviceCollection = PortableCollection<::reco::CaloClusterSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif
