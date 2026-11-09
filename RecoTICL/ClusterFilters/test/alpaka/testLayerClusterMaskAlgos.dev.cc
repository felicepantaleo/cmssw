// The layer cluster masks on the device: the mask of ones, the copy, the mask of the layer clusters of tracksters,
// and the algorithm and size filter.

#include <Eigen/Core>

#include <cstdint>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/LayerClusterMaskHostCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/LayerClusterMaskDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// The algorithms are a plugin source, so the test compiles them for its backend
#include "RecoTICL/ClusterFilters/plugins/alpaka/LayerClusterMaskAlgos.dev.cc"

using ALPAKA_ACCELERATOR_NAMESPACE::Platform;
using ALPAKA_ACCELERATOR_NAMESPACE::Queue;
using ALPAKA_ACCELERATOR_NAMESPACE::reco::CaloClusterDeviceCollection;
using ALPAKA_ACCELERATOR_NAMESPACE::ticl::LayerClusterMaskDeviceCollection;
using ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterDeviceCollection;
namespace algos = ALPAKA_ACCELERATOR_NAMESPACE::ticl;

namespace {

  struct LayerCluster {
    ::reco::CaloCluster::AlgoId algo;
    int cells;
    DetId::Detector detector;
    float inputMask;
  };

  // 0: silicon, below the min size; 1: kept; 2: scintillator, below the min size and kept; 3: algorithm not in the
  // list; 4: masked at the input; 5: above the max size
  const std::vector<LayerCluster> kLayerClusters = {
      {::reco::CaloCluster::hgcal_em, 1, DetId::HGCalEE, 1.f},
      {::reco::CaloCluster::hgcal_em, 3, DetId::HGCalEE, 1.f},
      {::reco::CaloCluster::hgcal_scintillator, 1, DetId::HGCalHSc, 1.f},
      {::reco::CaloCluster::hfnose, 3, DetId::Forward, 1.f},
      {::reco::CaloCluster::hgcal_had, 3, DetId::HGCalHSi, 0.f},
      {::reco::CaloCluster::hgcal_had, 200, DetId::HGCalHSi, 1.f},
  };
  const std::vector<float> kFiltered = {0.f, 1.f, 1.f, 0.f, 0.f, 0.f};
  // one trackster with the layer clusters 1 and 2
  const std::vector<uint32_t> kTracksterVertices = {1, 2};
  const std::vector<float> kMaskedByTracksters = {1.f, 0.f, 0.f, 1.f, 1.f, 1.f};

  std::vector<float> toVector(Queue& queue, const LayerClusterMaskDeviceCollection& mask) {
    const int32_t n = mask->metadata().size();
    ::ticl::LayerClusterMaskHostCollection host(queue, n);
    alpaka::memcpy(queue, host.buffer(), mask.buffer());
    alpaka::wait(queue);
    std::vector<float> out(n);
    for (int32_t i = 0; i < n; ++i) {
      out[i] = host.view()[i].mask();
    }
    return out;
  }

}  // namespace

TEST_CASE("Layer cluster masks on the device") {
  const auto& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    SKIP("no device of the backend");
  }

  for (const auto& device : devices) {
    Queue queue(device);
    const int32_t n = kLayerClusters.size();

    ::reco::CaloClusterHostCollection layerClustersHost(queue, n, n, n, n);
    ::ticl::LayerClusterMaskHostCollection inputHost(queue, n);
    auto lcs = layerClustersHost.view();
    for (int32_t i = 0; i < n; ++i) {
      const auto& lc = kLayerClusters[i];
      lcs.position()[i].x() = 50.f;
      lcs.position()[i].y() = 0.f;
      lcs.position()[i].z() = 350.f;
      lcs.position()[i].layer() = 10;
      lcs.position()[i].cells() = lc.cells;
      lcs.energy()[i].energy() = 1.f;
      lcs.energy()[i].correctedEnergy() = 1.f;
      lcs.energy()[i].correctedEnergyUncertainty() = 0.f;
      lcs.indexes()[i].algoID() = lc.algo;
      lcs.indexes()[i].seedID() = DetId(lc.detector, 0);
      lcs.indexes()[i].flags() = 0;
      lcs.timing()[i].time() = 0.f;
      lcs.timing()[i].timeError() = -1.f;
      inputHost.view()[i].mask() = lc.inputMask;
    }
    CaloClusterDeviceCollection layerClusters(queue, n, n, n, n);
    alpaka::memcpy(queue, layerClusters.buffer(), layerClustersHost.buffer());
    LayerClusterMaskDeviceCollection input(queue, n);
    alpaka::memcpy(queue, input.buffer(), inputHost.buffer());

    SECTION("The mask of ones") {
      LayerClusterMaskDeviceCollection ones(queue, n);
      algos::fillMaskWithOnes(queue, ones.view());
      CHECK(toVector(queue, ones) == std::vector<float>(n, 1.f));
    }

    SECTION("The copy") {
      LayerClusterMaskDeviceCollection copy(queue, n);
      algos::copyMask(queue, input.const_view(), copy.view());
      CHECK(toVector(queue, copy) == toVector(queue, input));
    }

    SECTION("The layer clusters of the tracksters are masked") {
      const int32_t nVertices = kTracksterVertices.size();
      ::ticl::TracksterHostCollection trackstersHost(queue, 1, 1, 1, 1, 1, 1, 1, nVertices);
      auto vertices = trackstersHost.view().vertices();
      vertices.offsets()[0].keys_offsets() = 0;
      vertices.offsets()[1].keys_offsets() = nVertices;
      for (int32_t k = 0; k < nVertices; ++k) {
        vertices.content()[k].values() = kTracksterVertices[k];
      }
      TracksterDeviceCollection tracksters(queue, 1, 1, 1, 1, 1, 1, 1, nVertices);
      alpaka::memcpy(queue, tracksters.buffer(), trackstersHost.buffer());
      LayerClusterMaskDeviceCollection mask(queue, n);
      algos::fillMaskWithOnes(queue, mask.view());
      algos::maskLayerClustersOfTracksters(queue, tracksters.const_view(), mask.view());
      CHECK(toVector(queue, mask) == kMaskedByTracksters);
    }

    SECTION("The algorithm and size filter") {
      algos::AlgoAndSizeFilter filter;
      filter.algos = {
          ::reco::CaloCluster::hgcal_em, ::reco::CaloCluster::hgcal_had, ::reco::CaloCluster::hgcal_scintillator};
      filter.nAlgos = 3;
      filter.minSize = 2;
      filter.maxSize = 100;
      LayerClusterMaskDeviceCollection output(queue, n);
      algos::filterByAlgoAndSize(queue, filter, layerClusters.const_view(), input.const_view(), output.view());
      CHECK(toVector(queue, output) == kFiltered);
    }
  }
}
