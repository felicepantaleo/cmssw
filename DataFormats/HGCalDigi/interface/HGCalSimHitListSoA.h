#ifndef DataFormats_HGCalDigi_interface_HGCalSimHitListSoA_h
#define DataFormats_HGCalDigi_interface_HGCalSimHitListSoA_h

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace hgcaldigi {

  // One Geant4 hit in a readout cell and in the time window of the digitizer, in input order.
  // energy is in keV. energyXtof is energy * tof, computed on the host, so that the accumulation only adds.
  GENERATE_SOA_LAYOUT(HGCalSimHitListSoALayout,
                      SOA_COLUMN(int32_t, denseIdx),
                      SOA_COLUMN(int32_t, timeBin),
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, energyXtof))
  using HGCalSimHitListSoA = HGCalSimHitListSoALayout<>;

}  // namespace hgcaldigi

#endif  // DataFormats_HGCalDigi_interface_HGCalSimHitListSoA_h
