#ifndef RecoTICL_Superclustering_alpaka_SuperclusteringDNNPairBuilder_h
#define RecoTICL_Superclustering_alpaka_SuperclusteringDNNPairBuilder_h

#include <cstdint>
#include <optional>
#include <vector>

#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Superclustering/interface/alpaka/SuperclusteringDNNDeviceCollections.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The preselection of TracksterLinkingbySuperClusteringDNN, read from the ParameterSet by the producer so that
  // the device code does not depend on FWCore.
  struct SuperclusteringDNNPairSelection {
    unsigned int featureCount;
    float deltaEtaWindow;
    float deltaPhiWindow;
    float seedPtThreshold;
    float candidateEnergyThreshold;
    float explVarRatioCut_energyBoundary;
    float explVarRatioMinimum_lowEnergy;
    float explVarRatioMinimum_highEnergy;
    bool filterByTracksterPID;
    std::vector<int> tracksterPIDCategoriesToFilter;
    float PIDThreshold;
  };

  // Builds the (seed, candidate) pairs and their DNN features on the device.
  // Same role as TracksterFeatureExtractionRunner in the CNN PID port: the number of pairs is only known on the
  // device, so prepare() counts them and copies the count to the host in acquire(), and fill() allocates and fills
  // the collections in produce().
  //
  // The host association reproduces the legacy algorithm only if the pairs are ordered by candidate pT rank, as
  // the legacy loop emits them: fill() writes each candidate's pairs at its prefix-sum offset.
  class SuperclusteringDNNPairBuilder {
  public:
    explicit SuperclusteringDNNPairBuilder(SuperclusteringDNNPairSelection selection);

    // Enqueues the pT ordering, the trackster preselection, the per-candidate pair counting and the copy of the
    // pair count to the host.
    void prepare(Queue& queue, ::ticl::TracksterSoA::ConstView tracksters);

    // Only call after the framework has synchronized the prepare() work.
    uint32_t pairCount() const;

    // Enqueues the filling of the pair indices and of the features.
    void fill(Queue& queue,
              ::ticl::TracksterSoA::ConstView tracksters,
              ticl::SuperclusteringDNNPairsDevice& pairs,
              ticl::SuperclusteringDNNPairFeaturesDevice& features);

  private:
    SuperclusteringDNNPairSelection const selection_;
    // TODO: per-event buffers, e.g. pT order, preselection mask, per-candidate pair offsets
    std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> candidatePairOffsetsDevice_;
    std::optional<cms::alpakatools::host_buffer<uint32_t>> pairCountHost_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTICL_Superclustering_alpaka_SuperclusteringDNNPairBuilder_h
