#include "FWCore/Concurrency/interface/ModulePoolPolicy.h"

#include <catch2/catch_all.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {
  // Drives the policy with invocations whose holding time is a fixed multiple of
  // the wall time consumed, so the resulting occupancy estimate is that multiple.
  unsigned int driveAtOccupancy(edm::ModulePoolPolicy& policy,
                                double occupancy,
                                unsigned int invocations,
                                std::chrono::microseconds step) {
    unsigned int target = policy.targetSize();
    for (unsigned int i = 0; i < invocations; ++i) {
      std::this_thread::sleep_for(step);
      const auto hold = std::chrono::nanoseconds(
          static_cast<std::int64_t>(occupancy * static_cast<double>(std::chrono::nanoseconds(step).count())));
      target = policy.recordCompletion(hold, 0ns);
    }
    return target;
  }
}  // namespace

TEST_CASE("ModulePoolPolicy sizing arithmetic", "[ModulePoolPolicy]") {
  constexpr unsigned int kMax = 16;

  SECTION("an idle module needs one instance") {
    REQUIRE(edm::ModulePoolPolicy::instancesFor(0., 1.5, kMax) == 1u);
    REQUIRE(edm::ModulePoolPolicy::instancesFor(0.001, 1.5, kMax) == 1u);
  }

  SECTION("headroom is applied and rounded up") {
    // The measured Phase-2 HLT hot module sits near occupancy 2.1 at 16 streams.
    REQUIRE(edm::ModulePoolPolicy::instancesFor(2.09, 1.5, kMax) == 4u);
    REQUIRE(edm::ModulePoolPolicy::instancesFor(2.09, 1.0, kMax) == 3u);
    REQUIRE(edm::ModulePoolPolicy::instancesFor(0.52, 1.5, kMax) == 1u);
  }

  SECTION("never exceeds the stream count") {
    REQUIRE(edm::ModulePoolPolicy::instancesFor(1000., 1.5, kMax) == kMax);
    REQUIRE(edm::ModulePoolPolicy::instancesFor(12., 2.0, kMax) == kMax);
  }

  SECTION("degenerate inputs still give a usable pool") {
    REQUIRE(edm::ModulePoolPolicy::instancesFor(-1., 1.5, kMax) == 1u);
    REQUIRE(edm::ModulePoolPolicy::instancesFor(2., 0., kMax) == 1u);
    REQUIRE(edm::ModulePoolPolicy::instancesFor(2., 1.5, 0u) == 1u);
  }
}

TEST_CASE("ModulePoolPolicy growth", "[ModulePoolPolicy]") {
  constexpr unsigned int kMax = 8;

  SECTION("starts at a single slot") {
    edm::ModulePoolPolicy policy{kMax};
    REQUIRE(policy.targetSize() == 1u);
    REQUIRE(policy.maxInstances() == kMax);
  }

  SECTION("stays at one slot for a module that is never busy") {
    edm::ModulePoolPolicy policy{kMax};
    driveAtOccupancy(policy, 0.01, 200, 200us);
    REQUIRE(policy.targetSize() == 1u);
  }

  SECTION("gives slots back when the load falls away") {
    edm::ModulePoolPolicy policy{kMax};
    const unsigned int busy = driveAtOccupancy(policy, 3.0, 300, 200us);
    REQUIRE(busy > 1u);
    const unsigned int idle = driveAtOccupancy(policy, 0.005, 400, 200us);
    REQUIRE(idle < busy);
  }

  SECTION("keeps several slots for a module busy on several events at once") {
    edm::ModulePoolPolicy policy{kMax};
    const unsigned int target = driveAtOccupancy(policy, 3.0, 200, 200us);
    REQUIRE(target > 1u);
    REQUIRE(target <= kMax);
  }

  SECTION("an overloaded window cannot open the pool to its cap in one step") {
    // The failure seen on the Phase-2 HLT menu: a window in which everything
    // queues must add a single slot, not jump to the stream count.
    edm::ModulePoolPolicy policy{24};
    for (unsigned int i = 0; i < 20; ++i) {
      policy.recordCompletion(1ms, 100ms);
    }
    REQUIRE(policy.targetSize() == 2u);
  }

  SECTION("growth stops once the waiting is gone") {
    edm::ModulePoolPolicy policy{24};
    driveAtOccupancy(policy, 0.01, 100, 200us);
    for (unsigned int i = 0; i < 300; ++i) {
      policy.recordCompletion(1ms, 100ms);
    }
    const unsigned int afterQueueing = policy.targetSize();
    REQUIRE(afterQueueing > 1u);
    REQUIRE(afterQueueing < 24u);
    for (unsigned int i = 0; i < 300; ++i) {
      std::this_thread::sleep_for(200us);
      policy.recordCompletion(100us, 0ns);
    }
    // No waiting left to justify more slots, so the pool must not keep opening.
    REQUIRE(policy.targetSize() <= afterQueueing);
  }

  SECTION("the pool is never sized above the stream count") {
    edm::ModulePoolPolicy policy{kMax};
    for (unsigned int i = 0; i < 2000; ++i) {
      policy.recordCompletion(100ms, 100ms);
    }
    REQUIRE(policy.targetSize() <= kMax);
  }
}

TEST_CASE("ModulePoolPolicy sizes on wall time, not CPU time", "[ModulePoolPolicy]") {
  constexpr unsigned int kMax = 16;
  constexpr unsigned int kInternalThreads = 8;

  // Same work per invocation, done serially or spread over threads with a
  // parallel_for. The internally parallel module finishes in a fraction of the
  // wall time, so it copes with the same rate while holding one instance.
  const auto serialHold = 8ms;
  const auto parallelHold = serialHold / kInternalThreads;

  edm::ModulePoolPolicy serial{kMax};
  edm::ModulePoolPolicy parallel{kMax};
  for (unsigned int i = 0; i < 200; ++i) {
    std::this_thread::sleep_for(200us);
    serial.recordCompletion(serialHold, 0ns);
    parallel.recordCompletion(parallelHold, 0ns);
  }

  SECTION("the serial module needs several instances to keep up") { REQUIRE(serial.targetSize() > 1u); }

  SECTION("the internally parallel module keeps up with fewer") {
    REQUIRE(parallel.targetSize() < serial.targetSize());
  }

  SECTION("sizing CPU time instead would inflate the pool by the thread count") {
    // What the pool must NOT do: charge the module for the threads its
    // parallel_for consumed.
    const double wallOccupancy = parallel.demandEstimate();
    const unsigned int correct = edm::ModulePoolPolicy::instancesFor(wallOccupancy, 1.5, kMax);
    const unsigned int ifChargedCpu = edm::ModulePoolPolicy::instancesFor(wallOccupancy * kInternalThreads, 1.5, kMax);
    REQUIRE(ifChargedCpu > correct);
  }
}

TEST_CASE("ModulePoolPolicy is safe under concurrent completions", "[ModulePoolPolicy]") {
  constexpr unsigned int kMax = 16;
  constexpr unsigned int kThreads = 8;
  constexpr unsigned int kPerThread = 2000;

  edm::ModulePoolPolicy policy{kMax};
  std::atomic<bool> sawDecrease{false};

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (unsigned int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&policy, &sawDecrease]() {
      unsigned int previous = 0;
      for (unsigned int i = 0; i < kPerThread; ++i) {
        const unsigned int size = policy.recordCompletion(10us, 1us);
        if (size < previous) {
          sawDecrease = true;
        }
        previous = size;
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(not sawDecrease.load());
  REQUIRE(policy.invocations() == static_cast<std::uint64_t>(kThreads) * kPerThread);
  REQUIRE(policy.targetSize() >= 1u);
  REQUIRE(policy.targetSize() <= kMax);
}
