#ifndef DataFormats_HGCalDigi_interface_HGCalSimHitListHost_h
#define DataFormats_HGCalDigi_interface_HGCalSimHitListHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalSimHitListSoA.h"

namespace hgcaldigi {

  using HGCalSimHitListHost = PortableHostCollection<HGCalSimHitListSoA>;

}  // namespace hgcaldigi

#endif  // DataFormats_HGCalDigi_interface_HGCalSimHitListHost_h
