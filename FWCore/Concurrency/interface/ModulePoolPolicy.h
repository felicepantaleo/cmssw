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
    A pool of interchangeable module instances reports each completed invocation to
 this policy, which returns the number of instances the pool should hold. The size
 never exceeds the maximum given at construction (the stream count, since at most
 that many invocations can be in flight at once).

    The estimate is Little's law, applied to the whole system rather than to the
 servers alone. For a module invoked at rate lambda, held for a mean time t and
 waiting a mean time w for a free instance, the mean number of invocations in the
 system is lambda * (t + w), which equals the total of holding and waiting time
 divided by the elapsed wall time. Neither the rate nor the two means have to be
 tracked separately. A headroom multiplier covers estimation error and burstiness.

    Counting the wait, and not only the holding time, matters once a limit is being
 enforced: invocations in service are capped by that limit, so lambda * t saturates
 there and could never ask for more than is already allowed.

    The estimate is taken over the LAST WINDOW alone, never over the job so far. A
 lifetime average would carry the startup queue forever and keep asking for more;
 measured on the Phase-2 HLT menu that pinned 22 modules to the stream count on a
 median true demand of 0.011.

    The size moves by ONE slot per window in either direction. Growth is paced so
 that a window reading far too high, which is what the first windows read while the
 limit is still too low and the gate is itself the cause of the queue it measures,
 cannot open the pool to its cap in a single step. Shrinking waits for
 shrinkPatience consecutive windows that justify fewer slots, so a brief lull does
 not close a pool that is about to be busy again.

    Shrinking here means a slot stops being handed out, not that anything is
 destroyed. That is why it is cheap and why no module needs protecting from it: a
 pool that shrinks too far only loses throughput, and the next busy window grows it
 back. Nothing is reconstructed, so an expensive constructor costs nothing.

    There is no absolute time constant anywhere in the policy, so it behaves the
 same for modules spanning microseconds to hundreds of milliseconds and adapts by
 itself when a module moves between a host and a device backend.

    Holding time is WALL CLOCK, from checkout to release, and never CPU time. Two
 cases make the distinction load bearing:

    - A module that parallelises internally (tbb::parallel_for) consumes several
      threads at once but finishes sooner. Its wall holding time is short, so it
      needs few instances, which is correct: it is already coping with the load.
      Sizing on CPU time instead would multiply the estimate by the internal thread
      count and inflate the pool for exactly the modules that need it least. It
      also means a module has two ways out of being a bottleneck, more instances or
      more internal parallelism, and the second costs no memory. This policy
      rewards the second automatically.
    - An ExternalWork module holds its instance across acquire, the asynchronous
      device work, and produce, because the device is writing into per-instance
      buffers. Measuring checkout to release captures that gap; summing the CPU
      spent in acquire and in produce would not.

 Thread safety:
    recordCompletion may be called concurrently from any thread. Exactly one thread
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
    void evaluate(std::uint64_t invocations) noexcept;

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
