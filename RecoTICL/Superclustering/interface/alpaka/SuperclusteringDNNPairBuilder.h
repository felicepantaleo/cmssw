#ifndef RecoTICL_Superclustering_alpaka_SuperclusteringDNNPairBuilder_h
#define RecoTICL_Superclustering_alpaka_SuperclusteringDNNPairBuilder_h

#include <cstdint>
#include <optional>

#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNFeatures.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNDeviceCollections.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The preselection of TracksterLinkingbySuperClusteringDNN. Plain data: it is passed by value to the kernels.
  struct SuperclusteringDNNPairSelection {
    static constexpr unsigned int kMaxPIDCategories = 8;

    unsigned int featureCount;
    float deltaEtaWindow;
    float deltaPhiWindow;
    float seedPtThreshold;
    float candidateEnergyThreshold;
    float explVarRatioCut_energyBoundary;
    float explVarRatioMinimum_lowEnergy;
    float explVarRatioMinimum_highEnergy;
    bool filterByTracksterPID;
    // the PID categories summed into the e/gamma probability, in the configured order (the order of the float sum)
    int pidCategories[kMaxPIDCategories];
    unsigned int nPIDCategories;
    float PIDThreshold;
  };

  // A trackster as the pairing reads it: computed once per trackster, then read by every pair it is part of.
  struct SuperclusteringDNNPairingTrackster {
    ::ticl::superclustering::TracksterKinematics kinematics;
    bool seed;       // can seed a supercluster: pT, explained variance ratio and PID
    bool candidate;  // can join a supercluster: energy and explained variance ratio
  };

  // Builds the (seed, candidate) pairs of the superclustering DNN and their features on the device.
  //
  // The number of pairs is only known on the device, and the framework needs it on the host to allocate the
  // products. So prepare(), called in acquire(), counts them and copies the count to the host; fill(), called in
  // produce() after the framework has waited for that copy, writes them.
  //
  // The pairs are those of TracksterLinkingbySuperClusteringDNN::linkTracksters: each candidate with every seed of
  // higher pT in the same endcap and in the eta-phi window. The legacy code finds them with an eta-phi tile; here
  // each candidate checks the list of the tracksters that can seed, which finds the same pairs. With tens of seeds
  // at high pileup, that is tens of checks per candidate on any backend. They are written grouped by candidate,
  // in increasing candidate index, and by increasing seed index within a candidate: the same order on every backend.
  class SuperclusteringDNNPairBuilder {
  public:
    explicit SuperclusteringDNNPairBuilder(SuperclusteringDNNPairSelection const& selection);

    // Enqueues the per-trackster kinematics and preselection, the list of the seeds, the counting of the pairs of
    // each candidate, their offsets, and the copy of the number of pairs to the host.
    void prepare(Queue& queue, ::ticl::TracksterSoA::ConstView tracksters);

    // The number of pairs: only after the framework has synchronised the work of prepare().
    uint32_t pairCount() const;

    // Enqueues the filling of the pairs and of their features, into collections of pairCount() rows.
    void fill(Queue& queue,
              ticl::SuperclusteringDNNPairsDevice& pairs,
              ticl::SuperclusteringDNNPairFeaturesDevice& features) const;

  private:
    SuperclusteringDNNPairSelection const selection_;

    // per event, from prepare() to fill()
    uint32_t nTracksters_ = 0;
    std::optional<cms::alpakatools::device_buffer<Device, SuperclusteringDNNPairingTrackster[]>> tracksters_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> seeds_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t>> nSeedsDevice_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> pairOffsets_;
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t>> pairCountDevice_;
    std::optional<cms::alpakatools::host_buffer<uint32_t>> pairCountHost_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Superclustering_alpaka_SuperclusteringDNNPairBuilder_h
