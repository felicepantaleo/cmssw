#ifndef RecoTICL_Superclustering_interface_alpaka_SuperclusteringDNN_h
#define RecoTICL_Superclustering_interface_alpaka_SuperclusteringDNN_h

#include <cstdint>

#include <Eigen/Core>

#include "DataFormats/HGCalReco/interface/TracksterSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTICL/Superclustering/interface/SuperclusteringDNNSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl {

  // The preselection of TracksterLinkingbySuperClusteringDNN. Plain data: it is passed by value to the kernels.
  struct SuperclusteringDNNPairSelection {
    static constexpr unsigned int kMaxPIDCategories = 8;

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

  // What the features and the preselection read from a trackster.
  struct TracksterKinematics {
    Eigen::Vector3f barycenter;
    float eta, phi;
    float rawEnergy;
    float rawPt;
    float time;
    float eigenvalue0;
    float explVarRatio;   // 0 when the eigenvalues sum to zero
    bool hasEigenvalues;  // the eigenvalues do not sum to zero
    Eigen::Vector3f pca;  // principal axis, pointing away from the interaction point
  };

  // A trackster as the pairing reads it: computed once per trackster, then read by every pair it is part of.
  struct SuperclusteringDNNPairingTrackster {
    TracksterKinematics kinematics;
    bool seed;       // can seed a supercluster: pT, explained variance ratio and PID
    bool candidate;  // can join a supercluster: energy and explained variance ratio
  };

  // The device buffers of the pairs of an event, from findPairs() to fillPairs()
  struct SuperclusteringDNNPairSearch {
    SuperclusteringDNNPairSearch(Queue& queue, uint32_t n)
        : nTracksters{n},
          tracksters{cms::alpakatools::make_device_buffer<SuperclusteringDNNPairingTrackster[]>(queue, n)},
          seeds{cms::alpakatools::make_device_buffer<uint32_t[]>(queue, n)},
          nSeeds{cms::alpakatools::make_device_buffer<uint32_t>(queue)},
          pairOffsets{cms::alpakatools::make_device_buffer<uint32_t[]>(queue, n)},
          nPairs{cms::alpakatools::make_device_buffer<uint32_t>(queue)} {}

    uint32_t nTracksters;
    cms::alpakatools::device_buffer<Device, SuperclusteringDNNPairingTrackster[]> tracksters;
    cms::alpakatools::device_buffer<Device, uint32_t[]> seeds;
    cms::alpakatools::device_buffer<Device, uint32_t> nSeeds;
    cms::alpakatools::device_buffer<Device, uint32_t[]> pairOffsets;
    cms::alpakatools::device_buffer<Device, uint32_t> nPairs;
  };

  // The device work of the superclustering DNN.
  //
  // The pairs are those of TracksterLinkingbySuperClusteringDNN::linkTracksters: each candidate with every seed of
  // higher pT in the same endcap and in the eta-phi window. The legacy code finds them with an eta-phi tile; here
  // each candidate checks the list of the tracksters that can seed, which finds the same pairs. With tens of seeds
  // at high pileup, that is tens of checks per candidate on any backend. They are written grouped by candidate,
  // in increasing candidate index, and by increasing seed index within a candidate: the same order on every backend.
  struct SuperclusteringDNNAlgo {
    // The kinematics and the preselection of the tracksters, the list of the seeds, and the offset of the pairs of each
    // candidate. search.nPairs holds their number.
    static void findPairs(Queue& queue,
                          ::ticl::TracksterSoA::ConstView tracksters,
                          const SuperclusteringDNNPairSelection& selection,
                          SuperclusteringDNNPairSearch& search);

    // The pairs and their features, into collections of search.nPairs rows.
    static void fillPairs(Queue& queue,
                          const SuperclusteringDNNPairSelection& selection,
                          const SuperclusteringDNNPairSearch& search,
                          ::ticl::SuperclusteringDNNPairsSoA::View pairs,
                          ::ticl::SuperclusteringDNNPairFeaturesSoA::View features);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ticl

#endif  // RecoTICL_Superclustering_interface_alpaka_SuperclusteringDNN_h
