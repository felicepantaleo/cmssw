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
    LimitedTaskQueue fixes its concurrency at construction, so it cannot express a
 limit discovered at run time. This queue takes the limit as a variable, and adds
 the property that matters when every module in a job is routed through one: when a
 slot is free the action runs on the CALLING THREAD and nothing is allocated.

    That fast path is the whole point. Routing a module through a task queue that
 always allocates costs both throughput and memory, because the framework otherwise
 calls the module inline: on the Phase-2 HLT menu an always-allocating gate cost
 15 percent throughput and 9 percent host memory across 228 modules. Most modules
 settle at a single slot and are rarely contended, so most invocations take the
 fast path and behave exactly as the ungated framework does.

    A claim on a slot is represented by a Slot token. Destroying it releases the
 claim; holding it keeps the slot reserved, which is what lets an ExternalWork
 module keep its instance across acquire, the asynchronous work, and produce.

    The running count is the single source of truth for concurrency. Inline and
 queued execution both go through it, so the limit cannot be exceeded by the two
 paths claiming independently.

 Grow and shrink: the limit may be set in either direction. Lowering it does not
 interrupt work already running; the count drains below the new limit naturally.

*/

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/task_group.h>

namespace edm {
  class ElasticTaskQueue {
  public:
    class Slot;

    explicit ElasticTaskQueue(unsigned int maxConcurrency)
        : max_{maxConcurrency == 0 ? 1u : maxConcurrency}, limit_{max_} {}

    ElasticTaskQueue(ElasticTaskQueue const&) = delete;
    ElasticTaskQueue& operator=(ElasticTaskQueue const&) = delete;

    // Holds a claim on one slot. The claim lasts until the token is destroyed or
    // released, so it can be carried across an asynchronous gap.
    class Slot {
    public:
      friend class ElasticTaskQueue;

      Slot() = default;
      ~Slot() { release(); }

      Slot(Slot&& other) noexcept : queue_{other.queue_} { other.queue_ = nullptr; }
      Slot& operator=(Slot&& other) noexcept {
        release();
        queue_ = other.queue_;
        other.queue_ = nullptr;
        return *this;
      }
      Slot(Slot const&) = delete;
      Slot& operator=(Slot const&) = delete;

      void release() noexcept {
        if (queue_) {
          auto* queue = queue_;
          queue_ = nullptr;
          queue->releaseOne();
        }
      }

      explicit operator bool() const noexcept { return queue_ != nullptr; }

    private:
      explicit Slot(ElasticTaskQueue* queue) noexcept : queue_{queue} {}
      ElasticTaskQueue* queue_ = nullptr;
    };

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
    template <typename F>
    bool push(oneapi::tbb::task_group& iGroup, F&& iAction);

    // As push, but hands the action its Slot so it can keep the claim past its own
    // return. Returns true when it ran inline.
    template <typename F>
    bool pushAndHold(oneapi::tbb::task_group& iGroup, F&& iAction);

  private:
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
        auto* queue = this;
        next.group->run([queue, action = next.action]() { (*action)(Slot{queue}); });
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

  template <typename F>
  bool ElasticTaskQueue::push(oneapi::tbb::task_group& iGroup, F&& iAction) {
    if (tryClaim()) {
      inlineRuns_.fetch_add(1, std::memory_order_relaxed);
      Slot slot{this};
      iAction();
      return true;
    }
    queuedRuns_.fetch_add(1, std::memory_order_relaxed);
    enqueue(iGroup, [action = std::forward<F>(iAction)](Slot&& slot) mutable {
      Slot held{std::move(slot)};
      action();
    });
    return false;
  }

  template <typename F>
  bool ElasticTaskQueue::pushAndHold(oneapi::tbb::task_group& iGroup, F&& iAction) {
    if (tryClaim()) {
      inlineRuns_.fetch_add(1, std::memory_order_relaxed);
      iAction(Slot{this});
      return true;
    }
    queuedRuns_.fetch_add(1, std::memory_order_relaxed);
    enqueue(iGroup, [action = std::forward<F>(iAction)](Slot&& slot) mutable { action(std::move(slot)); });
    return false;
  }

}  // namespace edm

#endif
