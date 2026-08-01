#ifndef FWCore_Concurrency_ElasticGate_h
#define FWCore_Concurrency_ElasticGate_h
// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     ElasticGate
//
/**\class ElasticGate ElasticGate.h "FWCore/Concurrency/interface/ElasticGate.h"

 Description: Limits how many tasks run at once, and tunes that limit to the load

 Usage:
    Couples an ElasticTaskQueue to a ModulePoolPolicy. Each task is timed from the
 moment it is pushed, so the gate knows both how long it waited for a slot and how
 long it then ran, which are exactly the two quantities the policy needs. After
 every task the policy is asked for the number of slots the load now justifies and
 the queue is widened to match.

    The limit starts at one and only grows, so a caller that never saturates the
 gate keeps a single slot and pays only that slot's dispatch cost.

*/

#include <chrono>
#include <utility>
#include <vector>

#include "FWCore/Concurrency/interface/ElasticTaskQueue.h"
#include "FWCore/Concurrency/interface/ModulePoolPolicy.h"

namespace edm {
  class ElasticGate {
  public:
    explicit ElasticGate(unsigned int maxConcurrency, ModulePoolPolicy::Config const& config = {})
        : queue_{maxConcurrency}, policy_{maxConcurrency, config}, reservations_{queue_.maxConcurrency()} {}

    ElasticGate(ElasticGate const&) = delete;
    ElasticGate& operator=(ElasticGate const&) = delete;

    // Runs iAction on the first free slot, then feeds the observed wait and run
    // times back into the policy and widens the queue if the load justifies it.
    template <typename T>
    void push(oneapi::tbb::task_group& iGroup, T&& iAction);

    // Reserves a slot, runs iAction on it, and KEEPS the slot reserved until
    // releaseSlot is called for the same stream. For a module whose work outlives
    // the first call, acquire starting device work that produce later reads, the
    // slot must not be handed on in between, so the reservation spans the gap.
    // At most one event per stream is in flight, so the stream identifies the
    // reservation across the thread change between acquire and produce.
    template <typename T>
    void pushAndHold(oneapi::tbb::task_group& iGroup, unsigned int streamID, T&& iAction);

    // Ends the reservation made by pushAndHold and charges the policy for the
    // whole span, waiting plus acquire plus the asynchronous work plus produce.
    void releaseSlot(unsigned int streamID);

    unsigned int concurrencyLimit() const noexcept { return queue_.concurrencyLimit(); }

    unsigned int maxConcurrency() const noexcept { return queue_.maxConcurrency(); }

    ModulePoolPolicy const& policy() const noexcept { return policy_; }

    std::uint64_t inlineRuns() const noexcept { return queue_.inlineRuns(); }
    std::uint64_t queuedRuns() const noexcept { return queue_.queuedRuns(); }

  private:
    struct Reservation {
      ElasticTaskQueue::Slot slot;
      std::chrono::steady_clock::time_point pushed;
      std::chrono::steady_clock::time_point started;
    };

    ElasticTaskQueue queue_;
    ModulePoolPolicy policy_;
    std::vector<Reservation> reservations_;
  };

  template <typename T>
  void ElasticGate::push(oneapi::tbb::task_group& iGroup, T&& iAction) {
    const auto pushed = std::chrono::steady_clock::now();
    queue_.push(iGroup, [this, pushed, action = std::forward<T>(iAction)]() mutable {
      const auto started = std::chrono::steady_clock::now();
      action();
      const auto finished = std::chrono::steady_clock::now();
      const unsigned int limit = policy_.recordCompletion(finished - started, started - pushed);
      queue_.setConcurrencyLimit(limit);
    });
  }

  template <typename T>
  void ElasticGate::pushAndHold(oneapi::tbb::task_group& iGroup, unsigned int streamID, T&& iAction) {
    const auto pushed = std::chrono::steady_clock::now();
    queue_.pushAndHold(
        iGroup, [this, streamID, pushed, action = std::forward<T>(iAction)](ElasticTaskQueue::Slot&& slot) mutable {
          auto& reservation = reservations_[streamID];
          reservation.pushed = pushed;
          reservation.started = std::chrono::steady_clock::now();
          reservation.slot = std::move(slot);
          action();
        });
  }

  inline void ElasticGate::releaseSlot(unsigned int streamID) {
    auto& reservation = reservations_[streamID];
    const auto finished = std::chrono::steady_clock::now();
    const unsigned int limit =
        policy_.recordCompletion(finished - reservation.started, reservation.started - reservation.pushed);
    queue_.setConcurrencyLimit(limit);
    // Releasing the slot is what lets the next event in, so it must happen after
    // the policy has been charged and after the caller is done with the instance.
    reservation.slot.release();
  }

}  // namespace edm

#endif
