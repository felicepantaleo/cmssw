#ifndef FWCore_Concurrency_ElasticTaskQueue_h
#define FWCore_Concurrency_ElasticTaskQueue_h
// -*- C++ -*-
//
// Package:     Concurrency
// Class  :     ElasticTaskQueue
//
/**\class ElasticTaskQueue ElasticTaskQueue.h "FWCore/Concurrency/interface/ElasticTaskQueue.h"

 Description: Runs at most a settable number of actions at once, inline when it can

 Usage:
    Unlike LimitedTaskQueue the limit is settable at run time, in either direction.
 When a slot is free the action runs on the calling thread and nothing is allocated;
 only a contended action is queued.

    A claim is a Slot, and it lasts until the Slot is destroyed or reset, so it can
 be carried across an asynchronous gap. The running count is the single source of
 truth: inline and queued execution both claim through it, so the two paths cannot
 exceed the limit between them.

*/

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/task_group.h>

namespace edm {
  class ElasticTaskQueue {
  public:
    explicit ElasticTaskQueue(unsigned int maxConcurrency)
        : max_{maxConcurrency == 0 ? 1u : maxConcurrency}, limit_{max_} {}

    ElasticTaskQueue(ElasticTaskQueue const&) = delete;
    ElasticTaskQueue& operator=(ElasticTaskQueue const&) = delete;

    // Returns a claim to the pool. Note this runs on whatever thread drops the
    // Slot, which for an ExternalWork module is the one that finished produce.
    struct SlotReleaser {
      void operator()(ElasticTaskQueue* queue) const noexcept { queue->releaseOne(); }
    };

    // Holds a claim on one slot. reset() gives it back; release() does NOT, it
    // disowns the claim and leaks it, per unique_ptr's usual meaning.
    using Slot = std::unique_ptr<ElasticTaskQueue, SlotReleaser>;

    void setConcurrencyLimit(unsigned int limit) noexcept {
      limit_.store(std::clamp(limit, 1u, max_), std::memory_order_release);
    }

    unsigned int concurrencyLimit() const noexcept { return limit_.load(std::memory_order_acquire); }
    unsigned int maxConcurrency() const noexcept { return max_; }
    unsigned int running() const noexcept { return running_.load(std::memory_order_acquire); }

    // How often the fast path was taken. If queued dominates, the gate is
    // serialising nearly every invocation and the limit itself is the cost.
    std::uint64_t inlineRuns() const noexcept { return inlineRuns_.load(std::memory_order_relaxed); }
    std::uint64_t queuedRuns() const noexcept { return queuedRuns_.load(std::memory_order_relaxed); }

    // Runs iAction, inline if a slot is free, otherwise queued behind the work
    // already using the slots. Returns true when it ran inline.
    bool push(oneapi::tbb::task_group& iGroup, std::invocable auto&& iAction) {
      if (tryClaim()) [[likely]] {
        inlineRuns_.fetch_add(1, std::memory_order_relaxed);
        Slot slot{this};
        iAction();
        return true;
      }
      queuedRuns_.fetch_add(1, std::memory_order_relaxed);
      enqueue(iGroup, [action = std::forward<decltype(iAction)>(iAction)](Slot&& slot) mutable {
        Slot held{std::move(slot)};
        action();
      });
      return false;
    }

    // As push, but hands the action its Slot so it can keep the claim past its own
    // return. Returns true when it ran inline.
    bool pushAndHold(oneapi::tbb::task_group& iGroup, std::invocable<Slot&&> auto&& iAction) {
      if (tryClaim()) [[likely]] {
        inlineRuns_.fetch_add(1, std::memory_order_relaxed);
        iAction(Slot{this});
        return true;
      }
      queuedRuns_.fetch_add(1, std::memory_order_relaxed);
      enqueue(iGroup,
              [action = std::forward<decltype(iAction)>(iAction)](Slot&& slot) mutable { action(std::move(slot)); });
      return false;
    }

  private:
    friend struct SlotReleaser;

    using Action = std::function<void(Slot&&)>;
    struct Pending {
      std::shared_ptr<Action> action;
      oneapi::tbb::task_group* group = nullptr;
    };

    // Takes a slot if the limit allows, without blocking.
    bool tryClaim() noexcept {
      unsigned int current = running_.load(std::memory_order_acquire);
      while (current < limit_.load(std::memory_order_acquire)) {
        if (running_.compare_exchange_weak(
                current, current + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
          return true;
        }
      }
      return false;
    }

    // Hands the slot to a waiting action if there is one, otherwise gives it back.
    void releaseOne() noexcept {
      Pending next;
      if (pending_.try_pop(next)) {
        next.group->run([queue = this, action = next.action]() { (*action)(Slot{queue}); });
        return;
      }
      running_.fetch_sub(1, std::memory_order_acq_rel);
    }

    void enqueue(oneapi::tbb::task_group& iGroup, Action&& iAction) {
      pending_.push(Pending{std::make_shared<Action>(std::move(iAction)), &iGroup});
      // A slot may have come free between the failed claim and the push above, so
      // retry once; otherwise this action could sit waiting with the queue idle.
      if (tryClaim()) {
        releaseOne();
      }
    }

    const unsigned int max_;
    std::atomic<unsigned int> limit_;
    std::atomic<unsigned int> running_{0};
    std::atomic<std::uint64_t> inlineRuns_{0};
    std::atomic<std::uint64_t> queuedRuns_{0};
    oneapi::tbb::concurrent_queue<Pending> pending_;
  };

}  // namespace edm

#endif
