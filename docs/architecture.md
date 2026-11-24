# BASP Dataflow v0.1 — architecture of the implementation

This document describes what is implemented in this repository and how the pieces map onto the
BASP Adaptive Dataflow Architecture. For what is *not* implemented, see
[limitations.md](limitations.md).

## The three levels

| Level | Question it answers | Where it lives |
|---|---|---|
| System | Where does each operation run, and what moves — weights or activations? | `basp/scheduler`, `basp/dataflow`, `basp/memory` |
| Processor | How is work issued to a device, ordered and synchronised? | `basp/streams`, `basp/backends/virtual`, `basp/compiler/emit.py` |
| Arithmetic | How are compressed weights turned into multiply-accumulates? | `basp/kernels`, kernel descriptors in `basp/simulator/compute.py` |

All three are optimised together because they share one description of a kernel (below).

## One kernel description for estimate and execution

A `Kernel` is a sequence of `Phase` objects; a phase is a set of *simultaneous* demands on
resources (bytes on a memory channel, bytes through a decoder, bytes over a link, MACs on a
compute array). Its uncontended duration is the roofline

    T_phase = max over resources (work / capacity)

so a fused streaming kernel takes `max(W/B, C/P, W/D)`, a staged kernel (decode into a buffer, then
compute) is two phases whose times add, and a kernel that streams weights from another device
simply lists that device's memory and the links among its demands.

The scheduler's cost model evaluates these descriptors analytically; the simulator executes the
same objects under contention. On an idle system the two agree to floating-point precision, which
is tested for every dataflow mode (`tests/test_scheduler.py`).

## Simulation model

* `simulator/engine.py`, `event_queue.py` — deterministic discrete-event kernel.
* `simulator/fluid.py` — every contended quantity is a `Resource`; a `Flow` needs work on several
  resources at once and progresses at the rate of its slowest stage (a ready/valid pipeline).
  Sharing follows dominant-resource fairness with strict priority classes; background traffic
  additionally shares a per-link bandwidth budget. Work accounting is exact by construction.
* `simulator/interconnect.py` — links with independent bandwidth per direction, per-hop DMA
  queues, route latency, multi-hop transfers, optional decode-on-arrival.
* `simulator/compute.py` — per-device FIFO kernel execution and the SRAM tile cache.
* `simulator/dram.py` — channel bandwidth arithmetic (a bandwidth model, not a cycle model).
* `simulator/energy.py` — static power × time plus per-MAC and per-byte energy.

## Unified logical memory

`memory/tensor_registry.py` is the single tensor namespace (`TensorDescriptor` is the 104-byte
record from the specification). `memory/residency.py` knows which device owns each tensor, which
devices hold copies and which copies are valid; immutable weights never go stale, mutable tensors
are versioned and a write invalidates other copies until explicitly synchronised. Each device has
a main arena (weights, replicas) and a scratch arena (temporaries) served by the first-fit
allocator in `memory/allocator.py`. `memory/kv_cache.py` is a paged KV cache with copy-on-write
block sharing. `memory/replication.py` implements adaptive expert caching.

## Seven dataflow modes

| Mode | What it means here | Wins when |
|---|---|---|
| Weight-stationary | Run where a valid weight copy is; activations cross the link | Large weights, single use, slow link |
| Activation-stationary | Run where the activations are; weights are copied in (optionally retained) | Weights reused many times on a fast processor |
| Output-stationary | Fold expert outputs into device-local partial sums; ship one partial per device | Several experts per device feed one reduction |
| Row-stationary | Keep weight tiles in the device SRAM and reuse them across invocations | A hot expert is re-run on a module with enough SRAM |
| Streaming | Weights flow holder → link → decoder → MAC as one overlapped pipeline, never stored | Cold weights used once; destination memory is scarce |
| Distributed | Row shards of one operation run on the devices holding them | Large projections; many modules |
| Predictive | Speculative, budgeted, cancellable loads of predicted tensors | Routing has structure beyond recency |

`dataflow/adaptive.py` unions the candidates of the enabled modes. `experiments/modes.py`
simulates one scenario per mode and `tests/test_scheduler.py::test_each_mode_wins_in_its_own_regime`
asserts that each wins where expected.

## Decision engine

For every valid path the cost model (`scheduler/cost_model.py`) estimates

    T_path = T_queue + T_input + T_kernel + T_output + T_sync

by list scheduling over a forkable `PlanState` (device availability, gap-aware link reservations,
tentative residency, replica memory). `scheduler/scheduler.py` then

1. orders the batch by upward rank (`scheduler/critical_path.py`),
2. assigns greedily by earliest finish time,
3. refines by coordinate descent on the completion time of the *whole batch* (ties: less
   interconnect traffic), so operations do not individually pick the same link or device.

Representation is part of the decision: copies may be stored in different lossless-equivalent
formats, the source with the earliest arrival wins, and a non-canonical copy is decoded once on
arrival (`CostModel.landing_format`). Lossy formats are never exchanged at run time.

The scheduler never changes which experts are selected; routing comes from the trace or from the
model's own router values.

## BASP Stream Protocol

`streams/` defines the wire formats (framing with CRC, 72-byte command descriptor plus id lists,
completion events, activation frames, block-indexed weight streams) independent of the simulator.
`compiler/emit.py` turns each resolved plan into commands (`FREE`/`ALLOCATE` → `TRANSFER` →
`EXECUTE`/`REDUCE` → `TRANSFER`), using `BARRIER` commands to narrow wide dependency lists.
`backends/virtual/system.py` executes them with bounded per-device queues and backpressure, DMA
windows (weight data only lands in memory that was `ALLOCATE`d), version checks, timeouts,
cancellation and an execution log.

## Compiler pipeline

    model shape ──► primitive ops (compiler/lower.py) ──► fused ops (compiler/fusion.py)
        ──► capability filtering + placement (core/capabilities.py, scheduler/placement.py)
        ──► dataflow mapping + scheduling (dataflow/, scheduler/)
        ──► device command streams (compiler/emit.py)

The graph is extended layer by layer because the experts to run are only known after the router.

## Verification

* `backends/cpu/executor.py` computes real values with the INT4 kernels *following the plans* while
  the simulator provides the order (co-simulation); `verification/numerical.py` compares against
  an independent float64 reference.
* `verification/dependency.py` checks that no operation or command started before its
  dependencies finished and that devices saw valid inputs.
* `verification/accounting.py` checks capacity, per-tag sums and byte conservation on links.

## Extending

* **A new hardware profile**: add a JSON file to `configs/` (see `desktop_gpu12.json`).
* **A recorded routing trace**: save it in the `RoutingTrace` JSON format and pass it to
  `Setup.run`.
* **A new dataflow mode**: add a candidate generator in `basp/dataflow/` and register it in
  `adaptive.py`; if it needs a new kernel shape, express it as phases.
