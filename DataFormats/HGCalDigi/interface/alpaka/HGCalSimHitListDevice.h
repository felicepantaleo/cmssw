#ifndef DataFormats_HGCalDigi_interface_alpaka_HGCalSimHitListDevice_h
#define DataFormats_HGCalDigi_interface_alpaka_HGCalSimHitListDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalSimHitListSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::hgcaldigi {

  using HGCalSimHitListDevice = PortableCollection<::hgcaldigi::HGCalSimHitListSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::hgcaldigi

#endif  // DataFormats_HGCalDigi_interface_alpaka_HGCalSimHitListDevice_h
