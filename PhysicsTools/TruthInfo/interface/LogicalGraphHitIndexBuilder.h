// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_LogicalGraphHitIndexBuilder_h
#define PhysicsTools_TruthInfo_LogicalGraphHitIndexBuilder_h

#include <array>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "SimDataFormats/TruthInfo/interface/InteractionId.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace truth {

  class LogicalGraphHitIndexBuilder {
  public:
    // sharedSubgraphStore selects the shared layout described in LogicalGraphHitIndex:
    // each hit is stored once, in an order that makes every subtree a contiguous range.
    // The producer writes this layout by default. False builds the materialised layout.
    explicit LogicalGraphHitIndexBuilder(uint32_t nParticles, bool sharedSubgraphStore = true);

    // trackId is event-local (each mixing sub-event reuses 1,2,3,...); it MUST be
    // namespaced by the packed EncodedEventId or signal and pileup collide.
    void setSimTrackForParticle(uint32_t particleId, uint64_t eventId, uint32_t trackId);
    void addParticleChild(uint32_t parentParticleId, uint32_t childParticleId);

    // Add a hit on `trackId`'s SimTrack to `channel`. recHitIndex defaults to "no
    // recHit" for channels without a DetId->RecHit link (muon); calo passes the mapped
    // global recHit index, a cell-keyed channel the cell. Returns false when the hit is
    // dropped: no energy, or no particle carries that SimTrack.
    bool addHit(HitChannel channel,
                uint64_t eventId,
                uint32_t trackId,
                uint32_t detId,
                float energy,
                uint32_t recHitIndex = LogicalGraphHitIndex::Hit::kInvalidRecHitIndex);

    // The same, for a channel that carries a time per hit, in ns. A channel takes either
    // this call or addHit, never both.
    bool addTimedHit(HitChannel channel,
                     uint64_t eventId,
                     uint32_t trackId,
                     uint32_t detId,
                     float energy,
                     uint32_t recHitIndex,
                     float time);

    // Whether a channel's hits are keyed by (detId, cell) rather than by detId alone.
    // The producer sets it for the Tracker, where the cell is the digi channel, and for
    // the MTD, where the cell is the category, row and col; the Calo and Muon channels
    // name a cell with their DetId.
    void setCellKeyed(HitChannel channel, bool value) { cellKeyed_[static_cast<std::size_t>(channel)] = value; }

    static uint64_t simKey(uint64_t eventId, uint32_t trackId) { return simObjectKey(eventId, trackId); }

    [[nodiscard]] LogicalGraphHitIndex finish();

    // Whether finish() actually wrote the shared layout. False when it was not asked
    // for, and also when it was asked for but the hit-carrying particles did not form
    // a forest, in which case finish() falls back to the materialised layout.
    [[nodiscard]] bool usedSharedStore() const { return usedSharedStore_; }

  private:
    using Hit = LogicalGraphHitIndex::Hit;
    using SlotRange = LogicalGraphHitIndex::SlotRange;

    // Per-particle hits are an append-only list, coalesced in finish(), so an insertion
    // is a single push_back.
    using HitList = std::vector<Hit>;

    static void appendHit(HitList& hits, uint32_t detId, uint32_t recHitIndex, float energy);

    // Sort by detId and merge entries that share a detId: energies are summed and the
    // merged entry keeps the valid recHitIndex, if any. Entries with non-positive energy
    // are dropped. Idempotent on already-coalesced lists.
    // cellKeyed groups by (detId, cell) instead of by detId, so two cells of one module
    // stay separate entries.
    static void coalesce(HitList& hits, bool cellKeyed);

    // The same merge for a channel with a time per hit: times[i] belongs to hits[i]
    // before and after, and a merged entry keeps the earliest time.
    static void coalesce(HitList& hits, std::vector<float>& times, bool cellKeyed);

    // Whether every particle has one time per hit.
    [[nodiscard]] static bool timesInStep(std::vector<HitList> const& hits,
                                          std::vector<std::vector<float>> const& times);

    // Collect the particle and every distinct descendant (cycle-safe) into `order`, each
    // once. `visited`, `touched` and `stack` are reusable scratch: `touched` lists the ids
    // set in `visited`, so the caller clears them in O(subgraph size).
    void collectSubgraphParticles(uint32_t particleId,
                                  std::vector<uint8_t>& visited,
                                  std::vector<uint32_t>& touched,
                                  std::vector<uint32_t>& stack,
                                  std::vector<uint32_t>& order) const;

    // Concatenate the (already coalesced) per-particle lists into CSR storage.
    static void buildHitCSR(std::vector<HitList> const& lists,
                            std::vector<uint32_t>& offsets,
                            std::vector<Hit>& storage);

    // Per particle, whether its descendant closure contains anything with a SimTrack.
    [[nodiscard]] std::vector<uint8_t> closureReachesSimTrack() const;

    // Order the particles so that every subtree occupies consecutive slots, so a
    // subgraph is a range of the single hit store. The tree is the SIM parentage: a
    // hit-carrying particle hangs under its nearest hit-carrying ancestor, through any
    // GEN-only particles between them. Fills the DFS slot of every particle and the
    // number of particles in its subtree.
    // False when a hit-carrying particle has two hit-carrying parents, directly or
    // through GEN-only particles; the outputs are then meaningless.
    [[nodiscard]] bool buildDfsOrder(std::vector<uint32_t>& slotToParticle,
                                     std::vector<uint32_t>& dfsPos,
                                     std::vector<uint32_t>& subtreeCount) const;

    // Slot ranges covering each particle's subgraph: one run for a hit-carrying
    // particle, the merged union of the runs below it for a GEN-only particle.
    void buildSubgraphRanges(std::vector<uint32_t> const& dfsPos,
                             std::vector<uint32_t> const& subtreeCount,
                             std::vector<uint32_t>& rangeOffsets,
                             std::vector<SlotRange>& ranges) const;

    [[nodiscard]] LogicalGraphHitIndex finishShared();
    [[nodiscard]] LogicalGraphHitIndex finishMaterialised();

    static constexpr uint32_t kNoParent = std::numeric_limits<uint32_t>::max();

    uint32_t nParticles_ = 0;
    std::array<bool, kNumHitChannels> cellKeyed_{};
    bool sharedSubgraphStore_ = false;
    bool usedSharedStore_ = false;

    std::unordered_map<uint64_t, uint32_t> trackIdToParticle_;
    std::vector<std::vector<uint32_t>> children_;

    // Whether a particle has a SimTrack, so it can carry hits and take part in the
    // SIM parentage tree.
    std::vector<uint8_t> hasSimTrack_;

    // [channel index][particle] -> direct hit list. Subgraph hits are aggregated
    // in finish().
    std::array<std::vector<HitList>, kNumHitChannels> directHits_;

    // A channel that never received a hit (not selected, or its detector absent)
    // is left empty by finish() without the per-particle subgraph aggregation.
    std::array<bool, kNumHitChannels> channelTouched_{};

    // [channel index][particle] -> the times of the direct hits, in step with
    // directHits_. Filled only for a channel that addTimedHit fills.
    std::array<std::vector<std::vector<float>>, kNumHitChannels> directTimes_;
    std::array<bool, kNumHitChannels> timed_{};
  };

}  // namespace truth

#endif
