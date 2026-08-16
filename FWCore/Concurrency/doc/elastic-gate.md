# Elastic gate: self-tuning per-module concurrency

Prototype, not proposed for merge as it stands. This note is for someone picking
the code up: what the pieces do, why the control law has the shape it has, and the
constraints that are not visible in the diff.

## What it does

Bounds how many streams run a given `edm::stream` module at once, at a limit each
module derives from its own observed load. No configuration: a module discovers
whether it needs one slot or twelve.

Measured on the Phase-2 HLT 75e33 timing menu, TTbar PU200, 32 threads, 24 streams:
the menu runs on 1374 concurrent module slots instead of 5472, at parity in
throughput and host memory, with a floor of 6 slots per module.

It does NOT save memory, and was originally built expecting to. See "What this does
not do" below before investing in it.

## Pieces

| class | header | role |
|---|---|---|
| `edm::ModulePoolPolicy` | `interface/ModulePoolPolicy.h` | decides the limit from observed load |
| `edm::ElasticTaskQueue` | `interface/ElasticTaskQueue.h` | enforces a limit that can change at run time |
| `edm::ElasticGate` | `interface/ElasticGate.h` | couples the two and times each invocation |

`WorkerT<stream::EDProducerAdaptorBase>::serializeRunModule` returns the gate, where
the framework previously returned no queue. The gate itself lives on the stream
adaptor, built in `doPreallocate` where the stream count is known.

## The control law

Little's law over the last window. For a module invoked at rate lambda, held for a
mean time t and waiting a mean w for a slot, the mean number of invocations in the
system is `lambda * (t + w)`, which is just `(holding + waiting) / elapsed`. Neither
the rate nor the means are tracked separately.

Four properties, each of which exists because the obvious alternative was measured
and failed:

- **Waiting is counted, not only holding.** Invocations in service are capped by the
  current limit, so `lambda * t` saturates there and can never ask for more than is
  already allowed.
- **The window is the last one, never the job.** A lifetime average carries the
  startup queue forever. With one, 22 of 228 modules ended pinned at the stream
  count on a median true demand of 0.011.
- **Growth is one slot per window.** While the pool is too small the queue it
  measures is partly its own doing, so the first windows read far too high; a jump
  proportional to the estimate lands on the cap and stays.
- **Shrinking needs `shrinkPatience` windows to agree.** Cheap to be wrong downward,
  since a slot that stops being handed out costs only throughput and nothing is
  destroyed, so the patience sits on the shrink side.

Holding time is wall clock from checkout to release, never CPU. A module that
parallelises internally finishes sooner and needs fewer slots, which is correct;
CPU time would multiply its estimate by its own thread count. An ExternalWork module
holds its instance across acquire, the device work and produce, and checkout to
release spans exactly that.

### The floor is what usually binds

`minInstances` matters more than `headroom` and this is not obvious. Headroom
multiplies demand, but most modules have demand near 0.01, so `ceil(8 * 0.01)` is
still 1: no headroom setting lifts them. Per module the collision rate is only about
4 percent, but an event crosses a few hundred modules, and `1 - 0.96^228` is
essentially 1, so with a floor of one nearly every event stalls somewhere. Measured
cost against ungated, same menu:

| floor | throughput | host memory | total slots |
|---|---|---|---|
| 1 | +16.3% | +8.1% | 255 |
| 2 | +14.0% | +8.8% | 474 |
| 3 | +8.1% | +4.8% | 698 |
| 4 | +7.2% | +3.9% | 922 |
| 6 | +0.0% | +0.8% | 1374 |

Whether `floor = nStreams / 4` is a rule or an accident of 24 streams is untested.

## The inline fast path

When a slot is free the action runs on the calling thread and nothing is allocated.
This is not an optimisation, it is required. With no queue the framework calls
`runModuleAfterAsyncPrefetch` directly; give a module a queue and the same call
becomes a heap-allocated task capturing the `EventTransitionInfo`, the contexts and
the service token. Routing every stream module through an always-allocating queue is
a per-invocation allocation on a path that had none.

The running count is the single source of truth. Inline and queued execution both
claim through it, so the two paths cannot exceed the limit between them; a design
that counted inline separately and fell back to a queue would allow twice the limit.

Measured hit rate: 95 to 96 percent of invocations take the fast path.

## Hazards

**`Slot::reset()` gives a slot back; `release()` does not.** `Slot` is a
`std::unique_ptr` alias, so `release()` has unique_ptr's meaning of disowning the
pointer without running the deleter. Calling it where `reset()` is meant compiles,
does nothing, and leaks the claim until the pool wedges. The
"gives every slot back" test asserts `running() == 0` and catches it.

**This is an ABI break for every prebuilt stream-module plugin.** The gate adds a
member to `ProducingModuleAdaptorBase<T>`, a template that every plugin defining an
`edm::stream` module instantiates, and a pointer to `Worker::TaskQueueAdaptor`, which
is returned by value from a virtual. Plugins compiled against the old headers read
the wrong offsets and segfault inside an unrelated module's `setupStreamModules`.
`git cms-refresh -a` does not help because the consumer set is "everything". Only a
full build works, which makes this an IB-integrated change rather than something
testable in a patch-release area.

**`edm::limited`'s `concurrencyLimit` does not span acquire to produce.**
`AcquireTask` and `RunModuleTask` push to the queue separately, so the slot is
released when `acquire()` returns. The gate deliberately does not copy that: it holds
the claim across the gap, keyed by stream since at most one event per stream is in
flight.

## Prototype knobs

Environment variables, all of which must go before any PR. They exist so that both
legs of a comparison come from one build.

| variable | effect |
|---|---|
| `CMSSW_ELASTIC_GATE=0` | no gate at all, the ungated framework path |
| `CMSSW_ELASTIC_GATE_REPORT=1` | per module at destruction: limit, demand, invocations, inline and queued counts |
| `CMSSW_ELASTIC_HEADROOM=<float>` | headroom on the estimate |
| `CMSSW_ELASTIC_MIN_SLOTS=<int>` | the floor |
| `CMSSW_ELASTIC_NEVER_LIMIT=1` | gate present on the path but never restricting |

`CMSSW_ELASTIC_NEVER_LIMIT` is the useful diagnostic: it separates the cost of the
gate being on the path (+0.5 percent wall, noise) from the cost of what it decides.

## What this does not do

It does not save memory, and the reason is worth knowing before extending it.

Per-stream memory on this menu is 527 MiB, of which 413 is event data in flight, 87
allocator overhead, 19 module state retained across events and 7 module construction.
All module state together is 26 of 527 MiB. Removing three quarters of the instances
would save about 2.6 percent of the job, and dropping from 24 to 16 streams saves
9.4 percent for 4.3 percent throughput with no code at all.

Device memory behaves the same way: the Alpaka caching allocator's high-water mark
follows how many events are in flight holding buffers, not how many modules may run,
so bounding concurrency produced no reproducible change and bounding it hard made it
12 percent worse.

So the value here is concurrency control, not memory. Reducing the instances
themselves, which would need instances decoupled from streams, a new module category
and transition replay, was measured as not worth building.

## Open

- Is the useful floor a function of the stream count, or is 6 particular to 24
  streams and this menu.
- One menu, one machine, CPU and GPU backends of the same menu only.
- The gate applies to stream producers and filters. Analyzers are untouched.
