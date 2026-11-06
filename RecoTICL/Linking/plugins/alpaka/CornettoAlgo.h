#ifndef RecoTICL_Linking_plugins_alpaka_CornettoAlgo_h
#define RecoTICL_Linking_plugins_alpaka_CornettoAlgo_h

// TracksterDeviceCollection.h includes Eigen, which must precede the other SoA headers
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"

#include <cstdint>
#include <optional>

#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterPidSoA.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterLinksDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl::cornetto {

  constexpr uint32_t kMaxCollections = 4;
  constexpr uint32_t kMaxEmCategories = 8;  // the number of ticl::Trackster::ParticleType values

  struct Cone {
    float maxBackward;      // max separation [cm] upstream of the anchor along its axis
    float maxLongitudinal;  // max separation [cm] downstream of the anchor along its axis
    float radius0;          // max transverse distance [cm] at zero separation
    float slope;            // growth of the max transverse distance per cm of separation
  };

  // The parameters of the pair test and of the emission rule, with the meaning of the parameters of
  // ticl::TracksterLinkingbyCornetto.
  struct Parameters {
    float etaWindow;
    Cone cone;
    Cone emCone;  // cone of an EM anchor, inside cone
    bool useTimeGate;
    float timeNSigma2;
    float timeFloor2;
    float typeVetoProbability;
    // the ParticleType values that count to the EM probability, summed in this order
    int emCategories[kMaxEmCategories];
    uint32_t nEmCategories;
    float minEmittedEnergy;
    float minEmittedPt;
    float minEmittedEnergyEM;
    float minEmittedPtEM;
    float minEmittedEnergyAnyPt;
  };

  // The input trackster collections, in the order of the ordered list. The global index of a trackster is its index
  // in its collection plus the offset of the collection. A collection with hasPid takes its PID probabilities from
  // pid, else from the trackster SoA.
  struct Inputs {
    uint32_t nCollections = 0;
    uint32_t offsets[kMaxCollections + 1] =
        {};  // the global index of the first trackster of each collection, and the total
    ::ticl::TracksterSoA::ConstView tracksters[kMaxCollections];
    ::ticl::TracksterPidSoA::ConstView pid[kMaxCollections];
    bool hasPid[kMaxCollections] = {};

    ALPAKA_FN_HOST_ACC uint32_t size() const { return offsets[nCollections]; }

    ALPAKA_FN_HOST_ACC uint32_t collectionOf(uint32_t global) const {
      uint32_t c = 0;
      while (c + 1 < nCollections and global >= offsets[c + 1]) {
        ++c;
      }
      return c;
    }
  };

  // The buffers that the grouping leaves for the output, and the counts the host needs to allocate the output
  struct Scratch {
    uint32_t nTracksters = 0;
    // per input trackster
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> label;    // the group label of each trackster
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> members;  // the tracksters, grouped, ascending
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> headEnd;  // its last entry is the group count
    // per group, in the order of the smallest member; entries after the last group are 0
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> groupSize;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> groupEnd;     // inclusive scan of groupSize
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> emitted;      // 1 when the group is emitted
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> outputEnd;    // inclusive scan of emitted
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> verticesEnd;  // of the emitted groups
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> membersEnd;   // of the emitted groups
    // the number of emitted groups, of their layer clusters and of their tracksters; pinned, read in produce
    std::optional<cms::alpakatools::host_buffer<uint32_t[]>> counts;

    uint32_t nLinked() const { return (*counts)[0]; }
    uint32_t nVertices() const { return (*counts)[1]; }
    uint32_t nMembers() const { return (*counts)[2]; }
  };

  // Cornetto linking on the device: the groups and the emission of ticl::TracksterLinkingbyCornetto without its
  // second pass, and the linked tracksters with their properties from the layer clusters.
  class Algo {
  public:
    // Groups the tracksters: the features, the tiles, the parent of each trackster, its root, its group, and the
    // emission of each group. Leaves in scratch what fill needs, and enqueues the copy of the counts to the host.
    void group(Queue& queue, const Parameters& parameters, const Inputs& inputs, Scratch& scratch) const;

    // Fills the emitted groups: the input tracksters of each linked trackster, its layer clusters (the layer clusters
    // of each input trackster in ascending order, in the order of the input tracksters), and its
    // properties from the layer clusters (energy, barycenter, time, principal components over all its layer clusters)
    // with the PID set to unknown. The collections must have scratch.nLinked() tracksters, scratch.nVertices() layer
    // clusters and scratch.nMembers() links.
    void fill(Queue& queue,
              const Inputs& inputs,
              const Scratch& scratch,
              float zLimitEm,
              reco::CaloClusterDeviceCollection::ConstView layerClusters,
              TracksterDeviceCollection::View tracksters,
              TracksterLinksDeviceCollection::View links) const;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl::cornetto

#endif  // RecoTICL_Linking_plugins_alpaka_CornettoAlgo_h
