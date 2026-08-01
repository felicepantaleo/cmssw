#include "FWCore/Concurrency/interface/ElasticTaskQueue.h"

#include <catch2/catch_all.hpp>
#include <oneapi/tbb/task_group.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST_CASE("ElasticTaskQueue limit", "[ElasticTaskQueue]") {
  SECTION("runs inline when a slot is free, so nothing is queued") {
    edm::ElasticTaskQueue queue{4};
    oneapi::tbb::task_group group;
    std::thread::id ran;
    const bool inlineRan = queue.push(group, [&ran]() { ran = std::this_thread::get_id(); });
    group.wait();
    REQUIRE(inlineRan);
    REQUIRE(ran == std::this_thread::get_id());
    REQUIRE(queue.running() == 0u);
  }

  SECTION("starts open and clamps to the maximum") {
    edm::ElasticTaskQueue queue{8};
    REQUIRE(queue.concurrencyLimit() == 8u);
    REQUIRE(queue.maxConcurrency() == 8u);

    queue.setConcurrencyLimit(4);
    REQUIRE(queue.concurrencyLimit() == 4u);

    queue.setConcurrencyLimit(100);
    REQUIRE(queue.concurrencyLimit() == 8u);
  }

  SECTION("can be narrowed once the load is known") {
    edm::ElasticTaskQueue queue{8};
    queue.setConcurrencyLimit(2);
    REQUIRE(queue.concurrencyLimit() == 2u);
    queue.setConcurrencyLimit(0);
    REQUIRE(queue.concurrencyLimit() == 1u);
  }

  SECTION("a zero maximum still leaves a usable slot") {
    edm::ElasticTaskQueue queue{0};
    REQUIRE(queue.maxConcurrency() == 1u);
    REQUIRE(queue.concurrencyLimit() == 1u);
  }
}

TEST_CASE("ElasticTaskQueue runs no more tasks at once than the limit", "[ElasticTaskQueue]") {
  constexpr unsigned int kTasks = 200;

  auto runWithLimit = [](unsigned int limit) {
    edm::ElasticTaskQueue queue{8};
    queue.setConcurrencyLimit(limit);

    std::atomic<unsigned int> running{0};
    std::atomic<unsigned int> peak{0};
    std::atomic<unsigned int> done{0};
    oneapi::tbb::task_group group;

    for (unsigned int i = 0; i < kTasks; ++i) {
      queue.push(group, [&running, &peak, &done]() {
        const unsigned int now = ++running;
        unsigned int seen = peak.load();
        while (now > seen && !peak.compare_exchange_weak(seen, now)) {
        }
        std::this_thread::sleep_for(200us);
        --running;
        ++done;
      });
    }
    group.wait();
    REQUIRE(done.load() == kTasks);
    return peak.load();
  };

  SECTION("a single slot serializes") { REQUIRE(runWithLimit(1) == 1u); }

  SECTION("three slots allow three at a time at most") { REQUIRE(runWithLimit(3) <= 3u); }
}

TEST_CASE("ElasticTaskQueue gives every slot back", "[ElasticTaskQueue]") {
  edm::ElasticTaskQueue queue{4};
  queue.setConcurrencyLimit(2);
  oneapi::tbb::task_group group;
  std::atomic<unsigned int> done{0};
  for (unsigned int i = 0; i < 500; ++i) {
    queue.push(group, [&done]() {
      std::this_thread::sleep_for(50us);
      ++done;
    });
  }
  group.wait();
  REQUIRE(done.load() == 500u);
  REQUIRE(queue.running() == 0u);
}

TEST_CASE("ElasticTaskQueue reserves a slot across an asynchronous gap", "[ElasticTaskQueue]") {
  // The behaviour the module pool depends on: the claim must outlive the call that
  // made it, so asynchronous work started there still owns the slot.
  constexpr unsigned int kTasks = 60;
  constexpr unsigned int kLimit = 3;

  edm::ElasticTaskQueue queue{8};
  queue.setConcurrencyLimit(kLimit);

  std::atomic<unsigned int> inFlight{0};
  std::atomic<unsigned int> peak{0};
  std::atomic<unsigned int> done{0};
  oneapi::tbb::task_group group;

  for (unsigned int i = 0; i < kTasks; ++i) {
    queue.pushAndHold(group, [&](edm::ElasticTaskQueue::Slot&& slot) {
      const unsigned int now = ++inFlight;
      unsigned int seen = peak.load();
      while (now > seen && !peak.compare_exchange_weak(seen, now)) {
      }
      auto held = std::make_shared<edm::ElasticTaskQueue::Slot>(std::move(slot));
      group.run([&inFlight, &done, held]() {
        std::this_thread::sleep_for(200us);
        --inFlight;
        ++done;
        held->release();
      });
    });
  }
  group.wait();

  REQUIRE(done.load() == kTasks);
  REQUIRE(peak.load() <= kLimit);
  REQUIRE(queue.running() == 0u);
}

TEST_CASE("ElasticTaskQueue can be grown while in use", "[ElasticTaskQueue]") {
  constexpr unsigned int kMax = 6;
  edm::ElasticTaskQueue queue{kMax};

  std::atomic<unsigned int> running{0};
  std::atomic<unsigned int> peak{0};
  std::atomic<unsigned int> done{0};
  oneapi::tbb::task_group group;

  auto body = [&running, &peak, &done]() {
    const unsigned int now = ++running;
    unsigned int seen = peak.load();
    while (now > seen && !peak.compare_exchange_weak(seen, now)) {
    }
    std::this_thread::sleep_for(300us);
    --running;
    ++done;
  };

  queue.setConcurrencyLimit(1);
  for (unsigned int i = 0; i < 100; ++i) {
    queue.push(group, body);
  }
  queue.setConcurrencyLimit(kMax);
  for (unsigned int i = 0; i < 100; ++i) {
    queue.push(group, body);
  }
  group.wait();

  REQUIRE(done.load() == 200u);
  REQUIRE(queue.concurrencyLimit() == kMax);
  REQUIRE(peak.load() <= kMax);
  REQUIRE(queue.running() == 0u);
}
