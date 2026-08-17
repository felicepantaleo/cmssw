#include "FWCore/Concurrency/interface/ElasticGate.h"
#include "FWCore/Concurrency/interface/ElasticTaskQueue.h"

#include <catch2/catch_all.hpp>
#include <oneapi/tbb/task_group.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("ATTACK lowering the limit under a sustained backlog", "[attack]") {
  // The header claims the count drains below a lowered limit naturally. releaseOne
  // hands a freed slot straight to a waiting task without rechecking the limit, so
  // with a continuous backlog the running count may never fall to the new value.
  // Several pusher threads are needed: with one, the inline fast path runs each
  // action to completion before returning and no backlog ever forms.
  constexpr unsigned int kPushers = 8;
  edm::ElasticTaskQueue queue{8};
  queue.setConcurrencyLimit(8);
  oneapi::tbb::task_group group;

  std::atomic<unsigned int> running{0};
  std::atomic<unsigned int> peakAfterShrink{0};
  std::atomic<bool> shrunk{false};
  std::atomic<unsigned int> done{0};
  std::atomic<unsigned int> maxSeenAnyTime{0};

  auto body = [&]() {
    const unsigned int now = ++running;
    unsigned int seen = maxSeenAnyTime.load();
    while (now > seen && !maxSeenAnyTime.compare_exchange_weak(seen, now)) {
    }
    if (shrunk.load()) {
      seen = peakAfterShrink.load();
      while (now > seen && !peakAfterShrink.compare_exchange_weak(seen, now)) {
      }
    }
    std::this_thread::sleep_for(500us);
    --running;
    ++done;
  };

  std::vector<std::jthread> pushers;
  pushers.reserve(kPushers);
  for (unsigned int t = 0; t < kPushers; ++t) {
    pushers.emplace_back([&]() {
      for (unsigned int i = 0; i < 120; ++i) {
        queue.push(group, body);
      }
    });
  }
  std::this_thread::sleep_for(20ms);
  queue.setConcurrencyLimit(1);
  shrunk = true;
  pushers.clear();  // joins
  group.wait();

  REQUIRE(done.load() == kPushers * 120);
  REQUIRE(queue.running() == 0u);
  INFO("peak concurrency before the shrink: " << maxSeenAnyTime.load());
  INFO("peak concurrency after lowering the limit to 1: " << peakAfterShrink.load());
  REQUIRE(maxSeenAnyTime.load() > 1u);  // the test itself must have created contention
  CHECK(peakAfterShrink.load() <= 1u);
}

TEST_CASE("ATTACK degenerate limits", "[attack]") {
  SECTION("zero maximum still yields a working single slot") {
    edm::ElasticTaskQueue queue{0};
    oneapi::tbb::task_group group;
    std::atomic<unsigned int> done{0};
    for (unsigned int i = 0; i < 20; ++i) {
      queue.push(group, [&done]() { ++done; });
    }
    group.wait();
    REQUIRE(done.load() == 20u);
    REQUIRE(queue.running() == 0u);
  }

  SECTION("a floor above the stream count is clamped, not honoured") {
    edm::ModulePoolPolicy::Config cfg;
    cfg.minInstances = 99;
    edm::ModulePoolPolicy policy{4, cfg};
    REQUIRE(policy.targetSize() <= 4u);
    REQUIRE(policy.targetSize() >= 1u);
  }

  SECTION("evaluationInterval of zero does not wedge or divide by zero") {
    edm::ModulePoolPolicy::Config cfg;
    cfg.evaluationInterval = 0;
    edm::ModulePoolPolicy policy{4, cfg};
    for (unsigned int i = 0; i < 50; ++i) {
      policy.recordCompletion(10us, 0ns);
    }
    REQUIRE(policy.targetSize() >= 1u);
    REQUIRE(policy.targetSize() <= 4u);
  }

  SECTION("absurd and negative durations do not escape the clamp") {
    edm::ModulePoolPolicy policy{8};
    policy.recordCompletion(std::chrono::nanoseconds::max(), std::chrono::nanoseconds::max());
    policy.recordCompletion(std::chrono::nanoseconds{-5}, std::chrono::nanoseconds{-5});
    for (unsigned int i = 0; i < 40; ++i) {
      policy.recordCompletion(1us, 0ns);
    }
    REQUIRE(policy.targetSize() >= 1u);
    REQUIRE(policy.targetSize() <= 8u);
  }
}

TEST_CASE("ATTACK an action that throws", "[attack]") {
  SECTION("inline path returns the slot when the action throws") {
    edm::ElasticTaskQueue queue{4};
    oneapi::tbb::task_group group;
    for (unsigned int i = 0; i < 10; ++i) {
      try {
        queue.push(group, []() { throw std::runtime_error("boom"); });
      } catch (std::exception const&) {
      }
    }
    group.wait();
    REQUIRE(queue.running() == 0u);
  }

  SECTION("a held reservation is still returned when the holder throws") {
    edm::ElasticGate gate{4};
    oneapi::tbb::task_group group;
    for (unsigned int stream = 0; stream < 4; ++stream) {
      try {
        gate.pushAndHold(group, stream, []() { throw std::runtime_error("boom"); });
      } catch (std::exception const&) {
      }
    }
    group.wait();
    for (unsigned int stream = 0; stream < 4; ++stream) {
      gate.releaseSlot(stream);
    }
    // Every slot back, so the gate still admits work.
    std::atomic<bool> ran{false};
    gate.push(group, [&ran]() { ran = true; });
    group.wait();
    REQUIRE(ran.load());
  }
}

TEST_CASE("ATTACK reserving twice on one stream without releasing", "[attack]") {
  // The framework guarantees one event per stream in flight. If that were ever
  // violated the second reservation overwrites the first, and the question is
  // whether the first slot is lost or returned.
  edm::ElasticGate gate{4};
  oneapi::tbb::task_group group;
  gate.pushAndHold(group, 0, []() {});
  gate.pushAndHold(group, 0, []() {});
  group.wait();
  gate.releaseSlot(0);
  std::atomic<unsigned int> done{0};
  for (unsigned int i = 0; i < 8; ++i) {
    gate.push(group, [&done]() { ++done; });
  }
  group.wait();
  INFO("if the overwritten slot leaked, fewer than 4 slots remain usable");
  REQUIRE(done.load() == 8u);
}
