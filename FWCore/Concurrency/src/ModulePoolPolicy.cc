// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     ModulePoolPolicy
//

#include "FWCore/Concurrency/interface/ModulePoolPolicy.h"

#include <algorithm>
#include <cmath>

namespace {
  std::uint64_t nowNanos() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }
}  // namespace

namespace edm {

  ModulePoolPolicy::ModulePoolPolicy(unsigned int maxInstances, Config const& config) noexcept
      : maxInstances_(std::max(1u, maxInstances)), config_(config) {
    targetSize_.store(config_.neverLimit ? maxInstances_ : std::clamp(config_.minInstances, 1u, maxInstances_),
                      std::memory_order_release);
  }

  ModulePoolPolicy::ModulePoolPolicy(unsigned int maxInstances) noexcept : ModulePoolPolicy(maxInstances, Config{}) {}

  unsigned int ModulePoolPolicy::recordCompletion(std::chrono::nanoseconds holdTime,
                                                  std::chrono::nanoseconds waitTime) noexcept {
    const auto hold = static_cast<std::uint64_t>(std::max<std::int64_t>(0, holdTime.count()));
    const auto wait = static_cast<std::uint64_t>(std::max<std::int64_t>(0, waitTime.count()));

    const std::uint64_t now = nowNanos();
    std::uint64_t expected = 0;
    if (firstCompletionNanos_.compare_exchange_strong(expected, now, std::memory_order_acq_rel)) {
      windowStartNanos_.store(now, std::memory_order_release);
    }

    totalHoldNanos_.fetch_add(hold, std::memory_order_relaxed);
    totalWaitNanos_.fetch_add(wait, std::memory_order_relaxed);
    windowHoldNanos_.fetch_add(hold, std::memory_order_relaxed);
    windowWaitNanos_.fetch_add(wait, std::memory_order_relaxed);
    const std::uint64_t seen = invocations_.fetch_add(1, std::memory_order_acq_rel) + 1;

    std::uint64_t windowStart = invocationsAtWindowStart_.load(std::memory_order_acquire);
    if (seen - windowStart >= config_.evaluationInterval) {
      // Whichever thread wins the exchange owns this window evaluation.
      if (invocationsAtWindowStart_.compare_exchange_strong(windowStart, seen, std::memory_order_acq_rel)) {
        evaluate();
      }
    }
    return targetSize();
  }

  void ModulePoolPolicy::evaluate() noexcept {
    const std::uint64_t now = nowNanos();
    const std::uint64_t windowStart = windowStartNanos_.exchange(now, std::memory_order_acq_rel);
    const std::uint64_t hold = windowHoldNanos_.exchange(0, std::memory_order_acq_rel);
    const std::uint64_t wait = windowWaitNanos_.exchange(0, std::memory_order_acq_rel);

    if (now <= windowStart) {
      return;
    }
    const double demand = static_cast<double>(hold + wait) / static_cast<double>(now - windowStart);
    recentDemand_.store(demand, std::memory_order_release);
    const unsigned int wanted = instancesFor(demand, config_.headroom, maxInstances_, config_.minInstances);
    const unsigned int current = targetSize_.load(std::memory_order_acquire);

    if (wanted > current) {
      // One slot at a time, however high the estimate: while the pool is too small
      // the queue it measures is partly its own doing.
      windowsBelow_.store(0, std::memory_order_release);
      targetSize_.store(std::min(current + 1, maxInstances_), std::memory_order_release);
      return;
    }

    if (wanted == current) {
      windowsBelow_.store(0, std::memory_order_release);
      return;
    }

    // Give a slot up only after several windows agree, so a lull does not close a
    // pool that is about to be busy again. Nothing is destroyed by this, the slot
    // simply stops being handed out, so being wrong here costs only throughput.
    const unsigned int below = windowsBelow_.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (below >= config_.shrinkPatience) {
      windowsBelow_.store(0, std::memory_order_release);
      targetSize_.store(std::max(current - 1, std::clamp(config_.minInstances, 1u, maxInstances_)),
                        std::memory_order_release);
    }
  }

  unsigned int ModulePoolPolicy::instancesFor(double occupancy,
                                              double headroom,
                                              unsigned int maxInstances,
                                              unsigned int minInstances) noexcept {
    const unsigned int cap = std::max(1u, maxInstances);
    const unsigned int floor = std::clamp(minInstances, 1u, cap);
    if (not(occupancy > 0.) or not(headroom > 0.)) {
      return floor;
    }
    const double scaled = std::ceil(headroom * occupancy);
    if (scaled >= static_cast<double>(cap)) {
      return cap;
    }
    return std::clamp(static_cast<unsigned int>(scaled), floor, cap);
  }

  double ModulePoolPolicy::demandEstimate() const noexcept {
    const std::uint64_t first = firstCompletionNanos_.load(std::memory_order_acquire);
    if (first == 0) {
      return 0.;
    }
    const std::uint64_t elapsed = nowNanos() - first;
    if (elapsed == 0) {
      return 0.;
    }
    // Holding plus waiting, so the estimate reflects demand rather than what the
    // current limit happened to allow through.
    const std::uint64_t inSystem =
        totalHoldNanos_.load(std::memory_order_relaxed) + totalWaitNanos_.load(std::memory_order_relaxed);
    return static_cast<double>(inSystem) / static_cast<double>(elapsed);
  }

}  // namespace edm
