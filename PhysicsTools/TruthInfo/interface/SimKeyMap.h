// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_SimKeyMap_h
#define PhysicsTools_TruthInfo_SimKeyMap_h

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace truth {

  // A map from a 64-bit key (truth::simObjectKey) to a 32-bit index, for a lookup made
  // once per sim-hit. Open addressing with linear probing in one contiguous table, at
  // most half full, so a lookup reads one cache line in most cases. A value must be
  // smaller than kNotFound.
  class SimKeyMap {
  public:
    static constexpr uint32_t kNotFound = std::numeric_limits<uint32_t>::max();

    SimKeyMap() = default;
    explicit SimKeyMap(std::size_t expectedSize) { reserve(expectedSize); }

    // Makes room for expectedSize keys with no further allocation.
    void reserve(std::size_t expectedSize) {
      std::size_t capacity = kMinCapacity;
      while (capacity < 2 * expectedSize)
        capacity *= 2;
      if (capacity > slots_.size())
        rehash(capacity);
    }

    [[nodiscard]] std::size_t size() const { return size_; }

    // Adds the key, or replaces the value of a key that is present.
    void insertOrAssign(uint64_t key, uint32_t value) {
      if (2 * (size_ + 1) > slots_.size())
        rehash(slots_.empty() ? kMinCapacity : 2 * slots_.size());
      Slot& slot = slots_[slotOf(key)];
      if (slot.value == kNotFound)
        ++size_;
      slot = Slot{key, value};
    }

    // The value of the key, or kNotFound.
    [[nodiscard]] uint32_t find(uint64_t key) const { return slots_.empty() ? kNotFound : slots_[slotOf(key)].value; }

  private:
    struct Slot {
      uint64_t key = 0;
      uint32_t value = kNotFound;
    };

    static constexpr std::size_t kMinCapacity = 16;

    // The slot that holds the key, or the free slot where it goes. The table is never
    // full, so the probe ends.
    [[nodiscard]] std::size_t slotOf(uint64_t key) const {
      // Fibonacci hashing: the high bits of the product depend on every bit of the key.
      std::size_t i = static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ull) >> shift_);
      while (slots_[i].value != kNotFound && slots_[i].key != key)
        i = (i + 1) & mask_;
      return i;
    }

    void rehash(std::size_t capacity) {
      std::vector<Slot> old(capacity);
      old.swap(slots_);
      mask_ = capacity - 1;
      shift_ = 64;
      for (std::size_t c = capacity; c > 1; c /= 2)
        --shift_;
      for (Slot const& slot : old) {
        if (slot.value != kNotFound)
          slots_[slotOf(slot.key)] = slot;
      }
    }

    std::vector<Slot> slots_;
    std::size_t mask_ = 0;
    unsigned shift_ = 63;
    std::size_t size_ = 0;
  };

}  // namespace truth

#endif
