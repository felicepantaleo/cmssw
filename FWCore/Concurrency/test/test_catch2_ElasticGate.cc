#include "FWCore/Concurrency/interface/ElasticGate.h"

#include <catch2/catch_all.hpp>
#include <oneapi/tbb/task_group.h>

#include <atomic>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

namespace {
  struct Result {
    unsigned int limit;
    unsigned int peak;
    unsigned int done;
  };

  // Offers tasks at a steady rate, so the load the gate sees is set by how long
  // each task runs rather than by how fast the test can push.
  Result offer(unsigned int maxConcurrency,
               unsigned int tasks,
               std::chrono::microseconds work,
               std::chrono::microseconds gap) {
    edm::ElasticGate gate{maxConcurrency};
    std::atomic<unsigned int> running{0};
    std::atomic<unsigned int> peak{0};
    std::atomic<unsigned int> done{0};
    oneapi::tbb::task_group group;

    for (unsigned int i = 0; i < tasks; ++i) {
      gate.push(group, [&running, &peak, &done, work]() {
        const unsigned int now = ++running;
        unsigned int seen = peak.load();
        while (now > seen && !peak.compare_exchange_weak(seen, now)) {
        }
        std::this_thread::sleep_for(work);
        --running;
        ++done;
      });
      std::this_thread::sleep_for(gap);
    }
    group.wait();
    return {gate.concurrencyLimit(), peak.load(), done.load()};
  }
}  // namespace

TEST_CASE("ElasticGate starts open so its first measurement is undistorted", "[ElasticGate]") {
  edm::ElasticGate gate{16};
  REQUIRE(gate.concurrencyLimit() == 16u);
  REQUIRE(gate.maxConcurrency() == 16u);
  REQUIRE(gate.policy().invocations() == 0u);
}

TEST_CASE("ElasticGate leaves a light load at one slot", "[ElasticGate]") {
  // Work far shorter than the interval between arrivals: one slot is plenty, so
  // the gate must not grow. This is the 528 of 575 case in the Phase-2 HLT menu.
  const auto result = offer(16, 200, 100us, 1000us);
  REQUIRE(result.done == 200u);
  // Scheduling jitter on a shared machine can add a slot; what matters is that the
  // gate narrows to near one rather than staying open at the stream count.
  REQUIRE(result.limit <= 2u);
  REQUIRE(result.peak <= 2u);
}

TEST_CASE("ElasticGate opens up for a load one slot cannot carry", "[ElasticGate]") {
  // Work much longer than the arrival interval, so invocations pile up and the
  // waiting time is what tells the gate to widen.
  const auto result = offer(16, 200, 2000us, 100us);
  REQUIRE(result.done == 200u);
  REQUIRE(result.limit > 1u);
  REQUIRE(result.limit <= 16u);
  REQUIRE(result.peak <= 16u);
}

TEST_CASE("ElasticGate never runs more at once than its own limit", "[ElasticGate]") {
  const auto result = offer(4, 200, 1000us, 50us);
  REQUIRE(result.done == 200u);
  REQUIRE(result.peak <= 4u);
  REQUIRE(result.limit <= 4u);
}

TEST_CASE("ElasticGate returns every reservation", "[ElasticGate]") {
  constexpr unsigned int kStreams = 4;

  SECTION("releasing a stream that never reserved is a no-op") {
    // Reached when the prefetch fails before acquire runs, so produce has to be
    // able to release unconditionally without inventing an invocation.
    edm::ElasticGate gate{kStreams};
    gate.releaseSlot(2);
    REQUIRE(gate.policy().invocations() == 0u);
    REQUIRE(gate.concurrencyLimit() >= 1u);
  }

  SECTION("a reservation survives its call and is given back once released") {
    edm::ElasticGate gate{kStreams};
    oneapi::tbb::task_group group;
    gate.pushAndHold(group, 1, []() {});
    group.wait();
    // Still held: the point of pushAndHold is that the claim outlives the call.
    REQUIRE(gate.policy().invocations() == 0u);
    gate.releaseSlot(1);
    REQUIRE(gate.policy().invocations() == 1u);
    // Releasing again must not charge the policy a second time.
    gate.releaseSlot(1);
    REQUIRE(gate.policy().invocations() == 1u);
  }

  SECTION("reservations on many streams are all returned") {
    edm::ElasticGate gate{kStreams};
    oneapi::tbb::task_group group;
    for (unsigned int s = 0; s < kStreams; ++s) {
      gate.pushAndHold(group, s, []() {});
    }
    group.wait();
    for (unsigned int s = 0; s < kStreams; ++s) {
      gate.releaseSlot(s);
    }
    REQUIRE(gate.policy().invocations() == kStreams);
    // Every slot back, so a fresh reservation still succeeds immediately.
    gate.pushAndHold(group, 0, []() {});
    group.wait();
    gate.releaseSlot(0);
    REQUIRE(gate.policy().invocations() == kStreams + 1);
  }
}
