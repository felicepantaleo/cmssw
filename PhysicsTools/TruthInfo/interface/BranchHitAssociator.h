// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_interface_BranchHitAssociator_h
#define PhysicsTools_TruthInfo_interface_BranchHitAssociator_h

#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <vector>

#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace truth {

  // The hit format the graph matches against. Any reco object can be matched by
  // exposing its hits as a range of RecoHit.
  struct RecoHit {
    uint32_t detId = 0;
    // The cell (rec)hit energy, for callers that need a per-object weight. The
    // SharedEnergy metric does not read it: the per-cell weight comes from the
    // associator's CellEnergyTable when it has one, else from the truth hit index.
    float energy = 0.f;
    float fraction = 1.f;  // fraction of the cell assigned to this reco object
    // The cell inside the module, on the tracker channel, where a DetId names a module
    // rather than a cell. A tracker hit matches only the same cell, so a hit left at
    // kNoCell matches nothing there.
    uint32_t cell = LogicalGraphHitIndex::Hit::kNoCell;
  };

  // Reconstructed energy per cell, detId ascending after finalize(). With it, the
  // associator uses the TICL arithmetic: each cell is weighted by its rechit energy, the
  // reco object owns fraction * energy and the branch owns its sim fraction * energy. A
  // cell absent from the table weighs nothing.
  class CellEnergyTable {
  public:
    void reserve(std::size_t n) {
      keys_.reserve(n);
      values_.reserve(n);
    }
    void add(uint32_t detId, float energy) {
      keys_.push_back(detId);
      values_.push_back(energy);
    }
    // Sorts by detId and sums the energies of a repeated detId. Entries added in
    // ascending detId order are not sorted again.
    void finalize();
    // The rechit energy on a cell, 0 if the cell has none.
    [[nodiscard]] float energy(uint32_t detId) const;
    [[nodiscard]] bool empty() const { return keys_.empty(); }
    [[nodiscard]] std::size_t size() const { return keys_.size(); }

  private:
    std::vector<uint32_t> keys_;
    std::vector<float> values_;
  };

  // Customization point: a reco object R is matchable if R::truthHits() returns a range
  // of RecoHit-like elements.
  template <class R>
  concept HasTruthHits = requires(const R& r) {
    { r.truthHits() } -> std::ranges::range;
  };

  struct BranchMatch {
    static constexpr uint32_t kInvalidRoot = std::numeric_limits<uint32_t>::max();
    uint32_t rootParticleId = 0;
    float sharedEnergy = 0.f;  // (SharedHits metric: number of shared reco hits)
    // Reco-normalized score: how much of the reco object the branch fails to cover
    // (denominator = the reco self-energy, or the reco hit count for SharedHits). Use
    // for the reco->branch direction. Lower is better.
    float score = 0.f;
    // Branch-normalized score: how much of the branch the reco object fails to cover
    // (denominator = the branch subgraph self-energy, or its cell count for
    // SharedHits). Use for the branch->reco direction. Lower is better.
    float reverseScore = 0.f;
    // Sim-normalized shared quantity: sharedEnergy over the branch energy in the
    // denominator detectors (its cell count for SharedHits). The HGCalValidator
    // efficiency axis. It is linear, so it is not 1 - reverseScore, which is squared.
    float sharedEnergyFraction = 0.f;
  };

  // Ascending score, then ascending index, so [0] is the best match.
  inline constexpr auto byAscendingScore = [](const auto& a, const auto& b) {
    if (a.score() != b.score())
      return a.score() < b.score();
    return a.index() < b.index();
  };

  // Associates reco objects to truth branches (subtrees) by shared detector hits.
  // Built once per event over a set of candidate branch roots. Holds the inverted
  // detId -> roots index and the per-cell total sim energy as sorted flat arrays.
  // bestBranches() merge-joins the sorted hits of the object with the sorted subgraph
  // hits of each candidate.
  class BranchHitAssociator {
  public:
    // SharedEnergy is the arithmetic of AllTracksterToSimTracksterAssociatorsByHitsProducer
    // in both directions: the score is the squared uncovered energy over the squared self
    // energy, and the shared energy per cell is the minimum of the two sides.
    // SharedHits counts objects, not energy: one rechit on the reco side, so a pixel
    // cluster counts once, and one cell on the branch side.
    enum class Metric { SharedEnergy, SharedHits };

    // Detectors that the sharedEnergyFraction denominator covers, one bit per
    // DetId::det() value. HitChannel::Calo holds barrel ECAL and HCAL PCaloHits next to
    // the HGCAL ones, with sampling fractions that differ by orders of magnitude. The
    // caller passes the detectors that its reco collection reconstructs. kAllDetectors
    // keeps the whole channel.
    static constexpr uint32_t kAllDetectors = 0xFFFFu;
    [[nodiscard]] static uint32_t detectorBit(uint32_t detId);

    // candidateRoots restricts the branch roots. An empty list means every particle,
    // or no candidate when emptyRootsMeansAll is false. Use false when the list is a
    // restriction that can select no particle in an event.
    // recHitEnergies, when given, weights every cell of the SharedEnergy metric by its
    // reconstructed energy instead of its total sim energy; it must outlive the
    // associator.
    // generations, when given, is particleGenerations of the graph and must outlive the
    // associator. It orders two candidates that tie on both scores, which happens when
    // a particle and its ancestor own the same cells: the particle, with the larger
    // generation, comes first. Without it, the lower particle id comes first.
    explicit BranchHitAssociator(LogicalGraphHitIndex const& hitIndex,
                                 std::vector<uint32_t> candidateRoots = {},
                                 Metric metric = Metric::SharedEnergy,
                                 HitChannel channel = HitChannel::Calo,
                                 bool emptyRootsMeansAll = true,
                                 uint32_t denominatorDetectors = kAllDetectors,
                                 CellEnergyTable const* recHitEnergies = nullptr,
                                 std::span<const uint32_t> generations = {});

    // Best branches for the hits of a reco object, by ascending score, then ascending
    // reverse score. If maxResults > 0, only the best maxResults are returned.
    [[nodiscard]] std::vector<BranchMatch> bestBranches(std::span<const RecoHit> recoHits,
                                                        std::size_t maxResults = 0) const;

    template <HasTruthHits R>
    [[nodiscard]] std::vector<BranchMatch> bestBranches(R const& reco, std::size_t maxResults = 0) const {
      std::vector<RecoHit> hits;
      for (auto const& h : reco.truthHits())
        hits.push_back(RecoHit{h.detId, h.energy, h.fraction, h.cell});
      return bestBranches(std::span<const RecoHit>(hits), maxResults);
    }

    // Adaptive-level match: the candidate that minimises
    //     score + reverseWeight * reverseScore
    // As a branch climbs, score falls and reverseScore rises. Candidates with reverseScore
    // above maxReverseScore are rejected. If no candidate is left, the ceiling is ignored.
    // rootParticleId is BranchMatch::kInvalidRoot when no root shares a hit.
    [[nodiscard]] BranchMatch bestAdaptiveBranch(std::span<const RecoHit> recoHits,
                                                 float reverseWeight = 1.f,
                                                 float maxReverseScore = 1.f) const;

    // The same argmin over a bestBranches() list, so several working points share one
    // merge-join.
    [[nodiscard]] static BranchMatch bestAdaptiveBranch(std::span<const BranchMatch> matches,
                                                        float reverseWeight,
                                                        float maxReverseScore);

  private:
    // Fill the coalesced per-root hit store used by the shared layout. A no-op for a
    // materialised index, which already persists the coalesced spans.
    void buildRootHits();

    [[nodiscard]] std::span<const LogicalGraphHitIndex::Hit> rootHits(uint32_t rootId) const;

    // Candidate roots whose subgraph touches a cell, by binary search; empty span
    // if the cell is untouched.
    [[nodiscard]] std::span<const uint32_t> rootsForCell(uint32_t detId) const;
    // Total sim energy on a cell (denominator for branch fractions), 0 if none.
    [[nodiscard]] float cellTotalEnergy(uint32_t detId) const;
    // The weight of a cell in the SharedEnergy metric: its rechit energy with a
    // CellEnergyTable, its total sim energy without one.
    [[nodiscard]] float cellWeight(uint32_t detId) const;
    // A branch hit's energy in the SharedEnergy metric: its sim fraction of the cell
    // times the cell weight.
    [[nodiscard]] float branchHitEnergy(LogicalGraphHitIndex::Hit const& hit) const;

    LogicalGraphHitIndex const* hitIndex_;
    CellEnergyTable const* recHitEnergies_ = nullptr;
    Metric metric_;
    HitChannel channel_;
    // Whether a DetId of this channel names a module, so that two hits match only on
    // the same cell. True for a cell-keyed channel, the tracker and the MTD; on the
    // other channels a DetId already names a cell and the field holds a recHit index.
    bool cellAware_ = false;
    uint32_t denominatorDetectors_;
    std::vector<uint32_t> roots_;
    // particleGenerations of the graph, or empty.
    std::span<const uint32_t> generations_;

    // Inverted index detId -> candidate roots, stored CSR-style: cellRootsKeys_
    // holds the distinct cell detIds (ascending); cellRootsOffsets_ indexes
    // cellRoots_, which holds the root ids (ascending within each cell).
    std::vector<uint32_t> cellRootsKeys_;
    std::vector<uint32_t> cellRootsOffsets_;
    std::vector<uint32_t> cellRoots_;

    // Per-cell total sim energy as parallel sorted arrays (cellEnergyKeys_ ascending).
    std::vector<uint32_t> cellEnergyKeys_;
    std::vector<float> cellEnergyValues_;

    // Per-root branch self-energy (sum of subgraph-hit energy^2 on channel_), indexed by
    // particle id. The denominator of the reverse score.
    std::vector<double> rootSelfEnergySq_;
    // Per-root branch total energy (LINEAR sum of the same hits, restricted to
    // denominatorDetectors_), the denominator of sharedEnergyFraction.
    std::vector<double> rootEnergy_;

    // Shared layout only: the subgraph hits of the candidate roots, coalesced to one
    // ascending entry per cell for the merge-join. CSR over roots_. Empty for a
    // materialised index, which persists coalesced spans.
    std::vector<uint32_t> rootHitOffsets_;
    std::vector<LogicalGraphHitIndex::Hit> rootHitStorage_;
    // particle id -> position in rootHitOffsets_, or kNoRoot when the particle is not a
    // candidate, or kPersistedSpan when the persisted span is already coalesced and no
    // private copy was made.
    std::vector<uint32_t> rootHitSlotOfRoot_;
    static constexpr uint32_t kNoRoot = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kPersistedSpan = kNoRoot - 1u;
  };

}  // namespace truth

#endif
