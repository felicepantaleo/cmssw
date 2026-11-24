// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "SimDataFormats/TruthInfo/interface/DeltaCoding.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace truth {

  void LogicalGraphHitIndex::Channel::pack() {
    delta::encode(directOffsets, storedDirectCounts);
    delta::encode(dfsOffsets, storedDfsCounts);
    const std::size_t nHits = directHits.size();
    storedDetIds.resize(nHits);
    storedCells.resize(nHits);
    storedEnergies.resize(nHits);
    uint32_t previousDetId = 0;
    uint32_t previousCell = 0;
    for (std::size_t i = 0; i < nHits; ++i) {
      Hit const& hit = directHits[i];
      storedDetIds[i] = hit.detId - previousDetId;
      storedCells[i] = hit.detId == previousDetId ? hit.recHitIndex - previousCell : hit.recHitIndex;
      storedEnergies[i] = hit.energy;
      previousDetId = hit.detId;
      previousCell = hit.recHitIndex;
    }
  }

  void LogicalGraphHitIndex::Channel::unpack(std::vector<uint32_t> const& directCounts,
                                             std::vector<uint32_t> const& dfsCounts,
                                             std::vector<uint32_t> const& detIds,
                                             std::vector<uint32_t> const& cells,
                                             std::vector<float> const& energies,
                                             std::vector<uint32_t>& directOffsets,
                                             std::vector<uint32_t>& dfsOffsets,
                                             std::vector<Hit>& directHits) {
    delta::decode(directCounts, directOffsets);
    delta::decode(dfsCounts, dfsOffsets);
    const std::size_t nHits = detIds.size();
    directHits.resize(nHits);
    if (cells.size() != nHits || energies.size() != nHits) {
      directHits.clear();
      return;
    }
    uint32_t previousDetId = 0;
    uint32_t previousCell = 0;
    for (std::size_t i = 0; i < nHits; ++i) {
      Hit& hit = directHits[i];
      hit.detId = previousDetId + detIds[i];
      hit.recHitIndex = hit.detId == previousDetId ? previousCell + cells[i] : cells[i];
      hit.energy = energies[i];
      previousDetId = hit.detId;
      previousCell = hit.recHitIndex;
    }
  }

  void LogicalGraphHitIndex::packRanges(std::vector<uint32_t> const& dfsPos,
                                        std::vector<uint32_t> const& rangeOffsets,
                                        std::vector<SlotRange> const& ranges,
                                        std::vector<uint32_t>& storedDfsPos,
                                        std::vector<uint32_t>& storedRangeCounts,
                                        std::vector<uint32_t>& storedRangeFirstSlots,
                                        std::vector<uint32_t>& storedRangeSlotCounts) {
    delta::encode(dfsPos, storedDfsPos);
    delta::encode(rangeOffsets, storedRangeCounts);
    storedRangeFirstSlots.resize(ranges.size());
    storedRangeSlotCounts.resize(ranges.size());
    uint32_t previousFirstSlot = 0;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      storedRangeFirstSlots[i] = ranges[i].firstSlot - previousFirstSlot;
      storedRangeSlotCounts[i] = ranges[i].slotCount;
      previousFirstSlot = ranges[i].firstSlot;
    }
  }

  void LogicalGraphHitIndex::unpackRanges(std::vector<uint32_t> const& storedDfsPos,
                                          std::vector<uint32_t> const& storedRangeCounts,
                                          std::vector<uint32_t> const& storedRangeFirstSlots,
                                          std::vector<uint32_t> const& storedRangeSlotCounts,
                                          std::vector<uint32_t>& dfsPos,
                                          std::vector<uint32_t>& rangeOffsets,
                                          std::vector<SlotRange>& ranges) {
    delta::decode(storedDfsPos, dfsPos);
    delta::decode(storedRangeCounts, rangeOffsets);
    ranges.resize(storedRangeFirstSlots.size());
    if (storedRangeSlotCounts.size() != ranges.size()) {
      ranges.clear();
      return;
    }
    uint32_t previousFirstSlot = 0;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      previousFirstSlot += storedRangeFirstSlots[i];
      ranges[i].firstSlot = previousFirstSlot;
      ranges[i].slotCount = storedRangeSlotCounts[i];
    }
  }

  void LogicalGraphHitIndex::pack() {
    for (auto& channel : channels_)
      channel.pack();
    packRanges(dfsPos_,
               rangeOffsets_,
               ranges_,
               storedDfsPos_,
               storedRangeCounts_,
               storedRangeFirstSlots_,
               storedRangeSlotCounts_);
  }

}  // namespace truth
