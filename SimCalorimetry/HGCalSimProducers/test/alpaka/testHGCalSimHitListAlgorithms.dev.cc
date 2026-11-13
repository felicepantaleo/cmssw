#include <bit>
#include <cstdint>
#include <random>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <alpaka/alpaka.hpp>

#include "DataFormats/HGCalDigi/interface/HGCalSimHitAccumHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "DataFormats/HGCalDigi/interface/alpaka/HGCalSimHitAccumDevice.h"
#include "SimCalorimetry/HGCalSimProducers/interface/alpaka/HGCalSimHitListAlgorithms.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  struct Sums {
    std::vector<float> e, eXt;
  };

  // sequential sums in the given hit order
  Sums sequentialSums(::hgcaldigi::HGCalSimHitListHost const& list, int32_t nCells, bool reverse) {
    Sums sums{std::vector<float>(nCells * ::hgcaldigi::kTimeBins, 0.f),
              std::vector<float>(nCells * ::hgcaldigi::kTimeBins, 0.f)};
    auto const view = list.const_view();
    int32_t const n = view.metadata().size();
    for (int32_t k = 0; k < n; ++k) {
      auto const hit = view[reverse ? n - 1 - k : k];
      auto const idx = hit.denseIdx() * ::hgcaldigi::kTimeBins + hit.timeBin();
      sums.e[idx] += hit.energy();
      sums.eXt[idx] += hit.energyXtof();
    }
    return sums;
  }

  ::hgcaldigi::HGCalSimHitListHost makeList(int32_t nHits, int32_t nCells, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> cell(0, nCells - 1);
    std::uniform_int_distribution<int32_t> bin(0, ::hgcaldigi::kTimeBins - 1);
    std::uniform_real_distribution<float> energy(1e-6f, 1e-3f);
    std::uniform_real_distribution<float> time(-150.f, 150.f);
    ::hgcaldigi::HGCalSimHitListHost list(nHits);
    auto view = list.view();
    for (int32_t i = 0; i < nHits; ++i) {
      auto row = view[i];
      row.denseIdx() = cell(rng);
      row.timeBin() = bin(rng);
      row.energy() = energy(rng);
      row.energyXtof() = row.energy() * time(rng);
    }
    return list;
  }

  // number of (cell, bin) entries that differ bit for bit from the reference
  int32_t countDifferences(Queue& queue,
                           ALPAKA_ACCELERATOR_NAMESPACE::hgcaldigi::HGCalRealisticSimHitDevice const& device,
                           Sums const& reference) {
    auto const nCells = device.const_view().metadata().size();
    ::hgcaldigi::HGCalRealisticSimHitHost host(queue, nCells);
    alpaka::memcpy(queue, host.buffer(), device.const_buffer());
    alpaka::wait(queue);
    int32_t differences = 0;
    for (int32_t c = 0; c < nCells; ++c) {
      for (int32_t b = 0; b < ::hgcaldigi::kTimeBins; ++b) {
        auto const idx = c * ::hgcaldigi::kTimeBins + b;
        differences +=
            std::bit_cast<uint32_t>(host.const_view()[c].sumQ()(b, 0)) != std::bit_cast<uint32_t>(reference.e[idx]);
        differences +=
            std::bit_cast<uint32_t>(host.const_view()[c].avgTOA()(b, 0)) != std::bit_cast<uint32_t>(reference.eXt[idx]);
      }
    }
    return differences;
  }

}  // namespace

TEST_CASE("HGCalSimHitListAlgorithms " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE), "[HGCalSimHitListAlgorithms]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    SKIP("No devices for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend");
  }
  Queue queue(devices.front());
  HGCalSimHitListAlgorithms const algo(256);

  SECTION("The sums are equal, bit for bit, to a sequential loop in input order") {
    constexpr int32_t nCells = 5000;
    constexpr int32_t nHits = 400000;
    auto const list = makeList(nHits, nCells, 12345);
    auto const forward = sequentialSums(list, nCells, false);

    // the test can see a change of order: the reversed order gives other sums
    auto const backward = sequentialSums(list, nCells, true);
    REQUIRE(forward.eXt != backward.eXt);

    auto const cells = algo.accumulate(queue, list, nCells);
    REQUIRE(countDifferences(queue, cells, forward) == 0);
  }

  SECTION("Cells without hits are zero") {
    constexpr int32_t nCells = 1000;
    ::hgcaldigi::HGCalSimHitListHost const list(0);
    auto const cells = algo.accumulate(queue, list, nCells);
    REQUIRE(countDifferences(queue, cells, sequentialSums(list, nCells, false)) == 0);
  }
}
