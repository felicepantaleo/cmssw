#ifndef SimCalorimetry_HGCalSimProducers_interface_alpaka_HGCalSimHitListAlgorithms_h
#define SimCalorimetry_HGCalSimProducers_interface_alpaka_HGCalSimHitListAlgorithms_h

#include <cstdint>

#include "DataFormats/HGCalDigi/interface/HGCalSimHitListHost.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalSimHitAccumDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class HGCalSimHitListAlgorithms {
  public:
    explicit HGCalSimHitListAlgorithms(int nThreads) : nThreads_(nThreads) {}

    // Accumulates a hit list into one row per dense index on the device: sumQ holds the sum of the
    // energies and avgTOA the sum of energy * tof, per time bin. The sums are equal, bit for bit,
    // to a sequential loop over the list in input order.
    hgcaldigi::HGCalRealisticSimHitDevice accumulate(Queue&,
                                                     ::hgcaldigi::HGCalSimHitListHost const&,
                                                     int32_t nCells) const;

  private:
    int nThreads_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // SimCalorimetry_HGCalSimProducers_interface_alpaka_HGCalSimHitListAlgorithms_h
