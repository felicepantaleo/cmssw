// The Recovery tracksters on the device: one trackster for each layer cluster with a non-zero mask and an energy of at
// least the minimum energy, in the order of the layer clusters.

#include <Eigen/Core>

#include <cmath>
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

// The wrapper is a plugin source, so the test compiles it for its backend
#include "RecoTICL/PatternRecognition/plugins/alpaka/HGCalRecoveryTrackstersSoAAlgoWrapper.dev.cc"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  struct LayerCluster {
    float z;
    float energy;
    float mask;
  };

  // 0: below the min energy; 1: kept; 2: masked; 3: at the min energy, kept; 4: kept, beyond the EM limit
  const std::vector<LayerCluster> kLayerClusters = {
      {330.f, 0.5f, 1.f},
      {340.f, 2.f, 1.f},
      {350.f, 3.f, 0.f},
      {360.f, 1.f, 1.f},
      {420.f, 4.f, 1.f},
  };
  const std::vector<uint32_t> kSelected = {1, 3, 4};
  constexpr double kMinEnergy = 1.;
  constexpr float kZLimitEm = 400.f;

}  // namespace

TEST_CASE("Recovery tracksters on the device") {
  const auto& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    SKIP("no device of the backend");
  }

  for (const auto& device : devices) {
    Queue queue(device);
    const int32_t n = kLayerClusters.size();

    ::reco::CaloClusterHostCollection layerClustersHost(queue, n, n, n, n);
    ::ticl::LayerClusterMaskHostCollection maskHost(queue, n);
    auto lcs = layerClustersHost.view();
    for (int32_t i = 0; i < n; ++i) {
      const auto& lc = kLayerClusters[i];
      lcs.position()[i].x() = 40.f + i;
      lcs.position()[i].y() = 10.f;
      lcs.position()[i].z() = lc.z;
      lcs.position()[i].layer() = 10 + i;
      lcs.position()[i].cells() = 3;
      lcs.energy()[i].energy() = lc.energy;
      lcs.energy()[i].correctedEnergy() = lc.energy;
      lcs.energy()[i].correctedEnergyUncertainty() = 0.f;
      lcs.indexes()[i].algoID() = ::reco::CaloCluster::hgcal_em;
      lcs.indexes()[i].seedID() = DetId(DetId::HGCalEE, 0);
      lcs.indexes()[i].flags() = 0;
      lcs.timing()[i].time() = 0.1f * i;
      lcs.timing()[i].timeError() = 0.02f;
      maskHost.view()[i].mask() = lc.mask;
    }
    ALPAKA_ACCELERATOR_NAMESPACE::reco::CaloClusterDeviceCollection layerClusters(queue, n, n, n, n);
    alpaka::memcpy(queue, layerClusters.buffer(), layerClustersHost.buffer());
    ALPAKA_ACCELERATOR_NAMESPACE::ticl::LayerClusterMaskDeviceCollection mask(queue, n);
    alpaka::memcpy(queue, mask.buffer(), maskHost.buffer());

    const HGCalRecoveryTrackstersSoAAlgoWrapper algo;
    auto selected = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, n);
    auto positions = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, n);
    algo.selectLayerClusters(
        queue, kMinEnergy, layerClusters.const_view(), mask.const_view(), selected.data(), positions.data());
    auto positionsHost = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, n);
    alpaka::memcpy(queue, positionsHost, positions);
    alpaka::wait(queue);
    const uint32_t nTracksters = positionsHost[n - 1];
    REQUIRE(nTracksters == kSelected.size());

    ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterDeviceCollection tracksters(
        queue, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters);
    algo.fillTracksters(
        queue, kZLimitEm, layerClusters.const_view(), selected.data(), positions.data(), tracksters.view());
    ::ticl::TracksterHostCollection trackstersHost(
        queue, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters, nTracksters);
    alpaka::memcpy(queue, trackstersHost.buffer(), tracksters.buffer());
    alpaka::wait(queue);
    const auto view = trackstersHost.const_view();

    for (uint32_t t = 0; t < nTracksters; ++t) {
      INFO("trackster " << t);
      const uint32_t lc = kSelected[t];
      const auto vertices = view.vertices()[t];
      REQUIRE(vertices.size() == 1);
      CHECK(vertices[0] == lc);
      CHECK(view.energy()[t].rawEnergy() == kLayerClusters[lc].energy);
      CHECK(view.energy()[t].rawEmEnergy() == (kLayerClusters[lc].z <= kZLimitEm ? kLayerClusters[lc].energy : 0.f));
      CHECK(view.energy()[t].regressedEnergy() == 0.f);
      CHECK(view.barycenter()[t].x() == lcs.position()[lc].x());
      CHECK(view.barycenter()[t].y() == lcs.position()[lc].y());
      CHECK(view.barycenter()[t].z() == lcs.position()[lc].z());
      CHECK(view.timing()[t].time() == lcs.timing()[lc].time());
      CHECK(view.timing()[t].timeError() == lcs.timing()[lc].timeError());
      for (int k = 0; k < 3; ++k) {
        CHECK(view.pca()[t].eigenvalues()(k) == 0.f);
      }
      for (int k = 0; k < 8; ++k) {
        CHECK(view.id()[t].idProbabilities()(k) == 0.f);
      }
    }
  }
}
