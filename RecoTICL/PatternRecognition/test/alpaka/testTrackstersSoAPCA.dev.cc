// The principal components of a trackster whose cleaned set (the most energetic layer cluster of each layer) has one
// layer cluster are all zeros, and the ones of a regular trackster are finite.

#include <cmath>
#include <cstdint>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "DataFormats/CaloRecHit/interface/alpaka/CaloClusterDeviceCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/HGCalSoALayerClustersExtraHostCollection.h"
#include "DataFormats/HGCalReco/interface/TracksterHostCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoALayerClustersExtraDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

// The wrapper is a plugin source, so the test compiles it for its backend
#include "RecoTICL/PatternRecognition/plugins/alpaka/HGCalTrackstersSoAAlgoWrapper.dev.cc"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  struct LayerCluster {
    float x;
    float y;
    float z;
    int layer;
    float energy;
    int trackster;
  };

  // Trackster 0: three layer clusters on one layer, so its cleaned set has one layer cluster.
  // Trackster 1: three layer clusters on three layers along a line.
  const std::vector<LayerCluster> kLayerClusters = {
      {50.f, 10.f, 330.f, 5, 3.f, 0},
      {51.f, 10.f, 330.f, 5, 2.f, 0},
      {50.f, 11.f, 330.f, 5, 1.f, 0},
      {60.f, 20.f, 340.f, 10, 1.f, 1},
      {60.5f, 20.2f, 341.f, 11, 1.f, 1},
      {61.f, 20.4f, 342.f, 12, 1.f, 1},
  };
  constexpr uint32_t kTracksters = 2;
  constexpr float kZLimitEm = 400.f;

}  // namespace

TEST_CASE("Trackster properties from the layer clusters") {
  const auto& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    SKIP("no device of the backend");
  }

  for (const auto& device : devices) {
    Queue queue(device);
    const uint32_t n = kLayerClusters.size();

    ::reco::CaloClusterHostCollection layerClustersHost(queue, n, n, n, n);
    HGCalSoALayerClustersExtraHostCollection clusteringHost(queue, n);
    auto lcs = layerClustersHost.view();
    auto clustering = clusteringHost.view();
    for (uint32_t i = 0; i < n; ++i) {
      const auto& lc = kLayerClusters[i];
      lcs.position()[i].x() = lc.x;
      lcs.position()[i].y() = lc.y;
      lcs.position()[i].z() = lc.z;
      lcs.position()[i].layer() = lc.layer;
      lcs.position()[i].cells() = 3;
      lcs.energy()[i].energy() = lc.energy;
      lcs.energy()[i].correctedEnergy() = lc.energy;
      lcs.energy()[i].correctedEnergyUncertainty() = 0.f;
      lcs.indexes()[i].algoID() = ::reco::CaloCluster::hgcal_em;
      lcs.indexes()[i].seedID() = DetId(DetId::HGCalEE, 0);
      lcs.indexes()[i].flags() = 0;
      lcs.timing()[i].time() = -99.f;
      lcs.timing()[i].timeError() = -1.f;
      clustering[i].tracksterIndex() = lc.trackster;
    }
    clustering.numberOfTrackstersScalar() = kTracksters;
    clustering.numberOfVerticesScalar() = n;

    ALPAKA_ACCELERATOR_NAMESPACE::reco::CaloClusterDeviceCollection layerClusters(queue, n, n, n, n);
    HGCalSoALayerClustersExtraDeviceCollection layerClustersClustering(queue, n);
    alpaka::memcpy(queue, layerClusters.buffer(), layerClustersHost.buffer());
    alpaka::memcpy(queue, layerClustersClustering.buffer(), clusteringHost.buffer());

    ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterDeviceCollection tracksters(
        queue, kTracksters, kTracksters, kTracksters, kTracksters, kTracksters, kTracksters, n);
    HGCalTrackstersSoAAlgoWrapper{}.run(
        queue, kZLimitEm, layerClusters.const_view(), layerClustersClustering.const_view(), tracksters.view());

    ::ticl::TracksterHostCollection trackstersHost(
        queue, kTracksters, kTracksters, kTracksters, kTracksters, kTracksters, kTracksters, n);
    alpaka::memcpy(queue, trackstersHost.buffer(), tracksters.buffer());
    alpaka::wait(queue);
    const auto view = trackstersHost.const_view();

    SECTION("Energy and layer clusters") {
      CHECK(view.energy()[0].rawEnergy() == Catch::Approx(6.f));
      CHECK(view.energy()[1].rawEnergy() == Catch::Approx(3.f));
      CHECK(view.vertices().offsets()[0].keys_offsets() == 0);
      CHECK(view.vertices().offsets()[1].keys_offsets() == 3);
      CHECK(view.vertices().offsets()[2].keys_offsets() == 6);
    }

    SECTION("A cleaned set of one layer cluster gives a PCA of zeros") {
      const auto pca = view.pca()[0];
      for (int k = 0; k < 3; ++k) {
        CHECK(pca.eigenvalues()(k) == 0.f);
        CHECK(pca.sigmas()(k) == 0.f);
        CHECK(pca.sigmasPCA()(k) == 0.f);
        for (int j = 0; j < 3; ++j) {
          CHECK(pca.eigenvectors()(j, k) == 0.f);
        }
      }
      CHECK(std::isfinite(view.barycenter()[0].x()));
      CHECK(std::isfinite(view.barycenter()[0].y()));
      CHECK(std::isfinite(view.barycenter()[0].z()));
    }

    SECTION("Three layers give a finite PCA with the principal axis along the line") {
      const auto pca = view.pca()[1];
      for (int k = 0; k < 3; ++k) {
        CHECK(std::isfinite(pca.eigenvalues()(k)));
        CHECK(std::isfinite(pca.sigmas()(k)));
        CHECK(std::isfinite(pca.sigmasPCA()(k)));
      }
      CHECK(pca.eigenvalues()(0) > 0.f);
      CHECK(pca.eigenvalues()(0) >= pca.eigenvalues()(1));
      CHECK(pca.eigenvalues()(1) >= pca.eigenvalues()(2));
      // the line has direction (0.5, 0.2, 1) per layer; the principal axis points away from the interaction point
      const float norm = std::sqrt(0.5f * 0.5f + 0.2f * 0.2f + 1.f);
      CHECK(pca.eigenvectors()(0, 0) == Catch::Approx(0.5f / norm).margin(1e-3));
      CHECK(pca.eigenvectors()(1, 0) == Catch::Approx(0.2f / norm).margin(1e-3));
      CHECK(pca.eigenvectors()(2, 0) == Catch::Approx(1.f / norm).margin(1e-3));
    }
  }
}
