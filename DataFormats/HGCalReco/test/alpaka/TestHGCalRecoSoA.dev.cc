#include <Eigen/Core>

#include <cstdint>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoALayerClustersExtraDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/LayerClusterMaskDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterLinksDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/TracksterPidDeviceCollection.h"
#include "DataFormats/HGCalReco/test/HGCalRecoSoATestValues.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"

using ALPAKA_ACCELERATOR_NAMESPACE::Platform;
using ALPAKA_ACCELERATOR_NAMESPACE::Queue;

namespace {

  // Copies a filled host collection to the device and back to a new host collection of the same sizes.
  template <typename TDevice, typename THost, typename... Sizes>
  THost roundTrip(Queue& queue, THost const& host, Sizes... sizes) {
    TDevice device(queue, sizes...);
    alpaka::memcpy(queue, device.buffer(), host.const_buffer());
    THost back(queue, sizes...);
    alpaka::memcpy(queue, back.buffer(), device.const_buffer());
    alpaka::wait(queue);
    return back;
  }

}  // namespace

TEST_CASE("HGCalReco SoA collections: sizes and copy to the device and back", "[HGCalRecoSoA]") {
  const auto& devices = cms::alpakatools::devices<Platform>();
  REQUIRE(not devices.empty());
  Queue queue(devices.front());

  const int32_t n = GENERATE(0, 1, 2, 5);
  const int32_t nVertices = hgcalrecotest::numberOfVertices(n);
  SECTION("tracksters") {
    auto host = hgcalrecotest::makeTracksters(n);
    REQUIRE(host.const_view().vertices().offsets().metadata().size() == n + 1);
    hgcalrecotest::fill(host.view());
    REQUIRE_NOTHROW(hgcalrecotest::check(host.const_view(), n));
    const auto back = roundTrip<ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterDeviceCollection>(
        queue, host, n, n, n, n, n, n, n, nVertices);
    REQUIRE_NOTHROW(hgcalrecotest::check(back.const_view(), n));
  }
  SECTION("layer clusters extra") {
    HGCalSoALayerClustersExtraHostCollection host(queue, n);
    hgcalrecotest::fill(host.view());
    const auto back =
        roundTrip<ALPAKA_ACCELERATOR_NAMESPACE::HGCalSoALayerClustersExtraDeviceCollection>(queue, host, n);
    REQUIRE_NOTHROW(hgcalrecotest::check(back.const_view(), n));
  }
  SECTION("layer cluster masks") {
    ticl::LayerClusterMaskHostCollection host(queue, n);
    hgcalrecotest::fill(host.view());
    const auto back = roundTrip<ALPAKA_ACCELERATOR_NAMESPACE::ticl::LayerClusterMaskDeviceCollection>(queue, host, n);
    REQUIRE_NOTHROW(hgcalrecotest::check(back.const_view(), n));
  }
  SECTION("trackster PID") {
    ticl::TracksterPidHostCollection host(queue, n);
    hgcalrecotest::fill(host.view());
    const auto back = roundTrip<ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterPidDeviceCollection>(queue, host, n);
    REQUIRE_NOTHROW(hgcalrecotest::check(back.const_view(), n));
  }
  SECTION("trackster links") {
    auto host = hgcalrecotest::makeLinks(n);
    REQUIRE(host.const_view().offsets().metadata().size() == n + 1);
    hgcalrecotest::fill(host.view());
    const auto back =
        roundTrip<ALPAKA_ACCELERATOR_NAMESPACE::ticl::TracksterLinksDeviceCollection>(queue, host, n, nVertices);
    REQUIRE_NOTHROW(hgcalrecotest::check(back.const_view(), n));
  }
}
