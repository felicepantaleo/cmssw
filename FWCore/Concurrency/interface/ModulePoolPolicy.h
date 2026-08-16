#ifndef FWCore_Concurrency_ModulePoolPolicy_h
#define FWCore_Concurrency_ModulePoolPolicy_h
// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     ModulePoolPolicy
//
/**\class ModulePoolPolicy ModulePoolPolicy.h "FWCore/Concurrency/interface/ModulePoolPolicy.h"

 Description: Decides how many interchangeable instances of one module a workload needs

 Usage:
    A pool reports each completed invocation and is told how many instances to hold.
 The estimate is Little's law over the LAST window: demand is (holding + waiting) /
 elapsed, so it measures what was asked for rather than what the current limit let
 through. The size moves one slot per window in either direction, growing at once
 and shrinking only after shrinkPatience windows agree, and stays within
 [minInstances, maxInstances].

    Holding time is WALL CLOCK from checkout to release, never CPU. A module that
 parallelises internally finishes sooner and so needs fewer instances, which is
 correct; charging it CPU time would multiply its estimate by its own thread count.
 An ExternalWork module holds its instance across acquire, the device work and
 produce, and checkout to release spans exactly that.

 Thread safety: recordCompletion may be called concurrently. Exactly one thread
 performs each window evaluation.

*/

#include <atomic>
#include <chrono>
#include <cstdint>

namespace edm {
  class ModulePoolPolicy {
  public:
    struct Config {
      // Multiplier applied to the Little's law estimate.
      double headroom = 1.5;
      // Invocations between evaluations. Also the sample count before the first one.
      unsigned int evaluationInterval = 20;
      // Consecutive windows justifying fewer slots before one is given up.
      unsigned int shrinkPatience = 3;
      // Diagnostic: hold every pool at the maximum, so the gate sits on the
      // execution path but never restricts anything. Separates the cost of the
      // gate being there at all from the cost of what it decides.
      bool neverLimit = false;
      // Smallest pool any module is held at. A mean occupancy near zero still
      // collides when many streams pass through the same module, and an event
      // crosses hundreds of modules, so a floor of one stalls nearly every event
      // somewhere even though each module is individually idle.
      unsigned int minInstances = 1;
    };

    ModulePoolPolicy(unsigned int maxInstances, Config const& config) noexcept;
    explicit ModulePoolPolicy(unsigned int maxInstances) noexcept;

    ModulePoolPolicy(ModulePoolPolicy const&) = delete;
    ModulePoolPolicy& operator=(ModulePoolPolicy const&) = delete;

    // Records one completed invocation. holdTime is the WALL CLOCK span from
    // checkout to release, including any asynchronous device work and any
    // internal parallelism, and must never be a CPU time. waitTime is how long
    // the checkout blocked waiting for a free instance. Returns the number of
    // instances the pool should now hold.
    unsigned int recordCompletion(std::chrono::nanoseconds holdTime, std::chrono::nanoseconds waitTime) noexcept;

    unsigned int targetSize() const noexcept { return targetSize_.load(std::memory_order_acquire); }

    unsigned int maxInstances() const noexcept { return maxInstances_; }

    // Instances implied by an occupancy, clamped to [1, maxInstances]. Pure, so
    // that the sizing arithmetic can be checked independently of any clock.
    static unsigned int instancesFor(double occupancy,
                                     double headroom,
                                     unsigned int maxInstances,
                                     unsigned int minInstances = 1) noexcept;

    // Mean number of invocations in the system, waiting or running, over the whole
    // job. Reporting only: the control loop uses the last window instead, because
    // a lifetime average keeps the startup queue in the estimate forever.
    double demandEstimate() const noexcept;

    // The windowed estimate the last evaluation acted on.
    double recentDemand() const noexcept { return recentDemand_.load(std::memory_order_acquire); }

    std::uint64_t invocations() const noexcept { return invocations_.load(std::memory_order_relaxed); }

  private:
    void evaluate() noexcept;

    const unsigned int maxInstances_;
    const Config config_;

    std::atomic<unsigned int> targetSize_{1};
    std::atomic<std::uint64_t> invocations_{0};
    std::atomic<std::uint64_t> invocationsAtWindowStart_{0};
    std::atomic<std::uint64_t> totalHoldNanos_{0};
    std::atomic<std::uint64_t> totalWaitNanos_{0};
    // Reset at every evaluation: the control loop acts on the last window only.
    std::atomic<std::uint64_t> windowHoldNanos_{0};
    std::atomic<std::uint64_t> windowWaitNanos_{0};
    std::atomic<std::uint64_t> windowStartNanos_{0};
    std::atomic<double> recentDemand_{0.};
    // Consecutive windows that justified fewer slots than the pool holds.
    std::atomic<unsigned int> windowsBelow_{0};
    // Steady-clock reading at the first completion, so that time spent before the
    // module was ever invoked does not dilute the occupancy estimate.
    std::atomic<std::uint64_t> firstCompletionNanos_{0};
  };
}  // namespace edm

#endif
