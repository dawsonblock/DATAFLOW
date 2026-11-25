# BASP Dataflow v0.5 — architecture of the implementation

This document describes what is implemented in this repository and how the pieces map onto the
BASP Adaptive Dataflow Architecture. The first part covers the dataflow engine (v0.1); then
[Architecture discovery](#architecture-discovery-v02) (v0.2) and
[Evidence and real inputs](#evidence-and-real-inputs-v03) (v0.3). For what is *not* implemented,
see [limitations.md](limitations.md).

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

## Architecture discovery (v0.2)

v0.2 turns the engine into a tool for comparing architectures rather than validating one.

### Mechanisms added to the simulator

| Mechanism | Where | What it models |
|---|---|---|
| Batched launches | `DeviceSpec.warm_startup_s`, `simulator/compute.py` | A kernel that starts the instant the previous one ended pays a reduced launch cost |
| Persistent kernels | `DeviceSpec.persistent` | Resident operators: no launch cost, never gated |
| Power gating | `PowerSpec.sleep_w`, `gate_after_s`, `wake_j` | Idle devices drop to a sleep state after a threshold and pay a wake energy |
| Energy-aware scheduling | `ExecutionPolicy.energy_weight_s_per_j` | Plans are ranked by finish time plus weighted energy |
| Dependency-event dispatch | `ExecutionPolicy.host_in_loop=False` | Operations are released by completion events, without a host round trip (static dataflows only) |
| Eager activation streams | `ExecutionPolicy.eager_activations` | The hidden state is broadcast to the modules before the routing decision is known |
| Concurrent streams | `StreamGroup`, `Setup.run_concurrent` | Independent requests share devices and links without a per-layer barrier |
| Replicated experts | `place_modules(replicas=...)` | Copies on distinct modules; the scheduler picks the less loaded holder |
| Tensor-parallel experts | `place_tensor_parallel`, `kernels.split_expert` | Every expert partitioned across all modules, partial outputs summed |
| Routing-only nodes | `DeviceKind.SWITCH` | Fabrics and multi-engine modules: carry traffic, hold nothing |
| Dynamic output projection | `place_modules(head="auto")` | Shards on the modules and a copy on the trunk device; chosen per token |

Each is covered by `tests/test_v02_mechanisms.py`, including closed-form expectations (for
example: removing the host from the loop saves exactly one round trip per issued batch).

### Eight architectures

`experiments/architectures.py` defines each architecture as a placement rule plus an execution
policy on a topology — none is privileged:

| | Architecture | Placement | Policy |
|---|---|---|---|
| A | Centralized | experts on the GPU while they fit, the rest on the host | weight- and activation-stationary, expert cache |
| B | Distributed | trunk blocks and experts on modules, no GPU | static, distributed |
| C | Pipeline | consecutive layers on consecutive modules | static, no host in the loop |
| D | Expert-local | whole experts on modules, trunk on the GPU | static, distributed |
| E | Tensor-parallel | every expert split across all modules | static, partial sums |
| F | Adaptive hybrid | experts on modules and cached on the GPU | all modes, cache, prefetch |
| G | Spatial dataflow | as D with resident kernels | no host in the loop, eager activation streams |
| H | Hierarchical | as F on a local fabric joining GPU and modules | all modes, cache, prefetch |

All eight reproduce the float64 reference on the small numeric model
(`tests/test_v02_experiments.py`).

### Parametric hardware and design-space exploration

* `dse/hardware.py` — a `HardwareDesign` (GPU memory, module count, module memory, channels, MAC
  lanes, link bandwidth, fabric, channel splitting, resident kernels) becomes a simulator topology
  on top of a *base profile* for the host and GPU, using `configs/design_estimates.json`. Every
  estimate is a (pessimistic, nominal, optimistic) triple; a design is built at one corner. The
  same table prices the design.
* `dse/space.py`, `dse/pareto.py`, `dse/search.py` — constrained design spaces, Pareto utilities
  and three searches behind one interface: exhaustive enumeration, an NSGA-II evolutionary search
  and a ParEGO Bayesian search with a NumPy Gaussian process.
* `experiments/explore.py` — the space of systems (architecture included as a dimension), its
  feasibility constraints, the seed designs that guarantee every architecture and the conventional
  baseline are evaluated, and the objective function: one simulation, three objectives
  (tokens/s, tokens/J, estimated price).

### Analysis

* `analysis/critical.py` walks the realised critical path of each token backwards through the
  dependency whose result arrived last, and attributes the time to computing, waiting for inputs,
  waiting for a busy device, returning results and host round trips. The components sum to the
  token time exactly. `scaling_limit` derives what more modules could deliver if the divisible
  (expert) work were perfectly balanced.
* `experiments/investigations.py` — one controlled A/B experiment per Phase 2 question.
* `experiments/suite.py` — the architecture table with estimate ranges, the model-size suite and
  the workload-type suite.

### Calibration and the evidence chain

    measurements (run_llama_bench)  ──►  calibration split ──► fit_profile ──► machine profile
                                    └─►  validation split ──► score ──► calibration verdict
    machine profile ──► base profile of every design ──► comparison ──► hardware gate

* `calibration/measurements.py` — measurement records with a split declared at recording time;
  the same benchmark cannot be in both splits.
* `experiments/calibrate.py` — `predict` simulates the benchmark a measurement describes;
  `fit_profile` tunes a profile on the calibration split with the simulator in the loop;
  `score` judges the validation split.
* `experiments/gate.py` — the hardware gate. A criterion passes only on calibrated evidence, at
  the pessimistic corner; a criterion that cannot be evaluated counts against the decision; and
  evidence is "calibrated" only if validation passed *and* the comparison ran on the validated
  profile.
* `experiments/report_v3.py` — the deterministic report that ties these together.

## Evidence and real inputs (v0.3)

    configs/calibration_plan.json ──► benchmarks/run_calibration ──► evidence/<machine>/
        (pre-registered splits)          llama-bench + GGUF headers      measurements, runs, fit
                                                                              │
    configs/<machine>.json ◄── fit on the calibration split ◄────────────────┘
            │                    score on the validation split ──► gate criterion 1 (E3)
            ▼
    base profile of every design ──► comparison (E1, or E2 once validated) ──► gate

* `basp/evidence.py` — evidence levels E0–E3; provenance (fingerprint of the simulator source,
  hashes of configurations and inputs, environment, git state); `stale_reasons` compares a stored
  result with the repository. Exploration results carry their provenance; the report writes
  `provenance.json` beside its deterministic `report.md` and `results.json`.
* `basp/experiments/gate.py` — a `Candidate` is one architecture in one regime with its ratios and
  *its own* scaling result. Each criterion has a required evidence level; insufficient evidence
  can fail a criterion but never pass it.
* `basp/models/gguf.py` — reads a GGUF file's metadata and tensor index (not its data): exact bits
  per weight, the model shape it implies (checked against the simulator's), a header fingerprint.
* `basp/experiments/variants.py` — the experiment `Context`: base profile, recorded routing trace,
  measured compression; every experiment and the exploration use it.
* `basp/backends/base.py` — the `Payload` contract between the runtime (when and where) and a
  backend that computes values (what). `NumericPayload` implements it; a GGML, CUDA or Metal
  backend will.
* Memory domains (`DeviceSpec.memory_domain`) — devices sharing one memory system (Apple unified
  memory) share one bandwidth resource and one allocation pool.
* Prompt programs (`ModelProgram(prompt=True)`) — the rows of a batch are one prompt sharing one KV
  cache, as llama-bench's prompt-processing test is.

## Evidence v2, memory domains and the physical runtime (v0.4)

**Receipts.** `basp/evidence/` is a package: `identity` (source fingerprint, file and profile
identities: full-content SHA-256), `manifests` (provenance of configurations, files and whole
directories), `schema` and `validation` (metrics with units and valid ranges: NaN, infinities and
negative errors are rejected), `receipts` (one metric, one value, one level, one candidate and
regime, the identities it is bound to, its raw data; the id is the hash of the content),
`registry` and `qualification`. A criterion is qualified from receipts of *its* metric, candidate
and regime only; a receipt missing the bindings its level requires (E1: source, config; E2: plus
profile and validation; E3: machine, model, runtime, benchmark), bound to an identity that has
changed since, or about another model, is rejected with a reason. Of the rest the highest level
counts: a miss fails at any level, a pass needs the required level.

**Memory domains.** `basp/memory/domains.py` models physical pools. Devices that share memory
share one `MemoryDomain`; allocations are charged to weights, KV cache, activations or workspace;
reserved working memory is never handed out, and `check_invariant` verifies that the categories
plus the reservation never exceed the capacity and that accounting matches the allocator to the
byte. Blocks are reference counted: a referenced or pinned block cannot be freed or evicted, and
an eviction that cannot reach its target frees nothing. `admission` reserves a request's whole KV
cache when it is admitted (evicting unreferenced cached weights first, keeping a margin under
pressure, refusing otherwise); `migration` copies between domains with the source held and the
destination hidden until the copy completes; `prefetch` issues migrations within a byte budget and
reports wasted traffic; `expert_cache` compares caching policies on routing traces.

**The physical runtime.** `basp/runtime/` executes real models and shares no code with the
simulator's timing model:

```
Engine(path)            GGUF -> ModelConfig; DeviceManager.discover(); make_plan(); WeightStore
  DeviceManager         every backend, available or not (CUDA, Metal, GGML, remote: unavailable)
  ExecutionPlan         operation groups -> backend; fails with UnsupportedOperation before running
  Dispatcher            op -> CommandQueue of the assigned backend -> Completion; telemetry
  WeightStore           dense weights resident (pinned, charged to the host domain); experts on demand, LRU
  Qwen3Model            embedding, RMSNorm, Q/K/V, per-head q/k norm, NEOX RoPE, GQA attention with KV
                        cache, SwiGLU or softmax top-k router + experts, final norm, logits
  Session               KV cache admitted against the domain; prefill / step
  generate()            greedy or seeded sampling, stop tokens
```

Backends implement `ExecutionBackend` (`basp/backends/execution.py`): `capabilities`,
`allocate`, `upload`, `submit`, `synchronize`, `release`, `telemetry`. The CPU backend runs NumPy
float32 kernels (`basp/backends/cpu/kernels.py`); weights are dequantised by
`basp/models/quants.py` (bit-identical to the reference `gguf` package for every supported format).

**Verification.** `basp/verification/` loads the same GGUF into transformers (weights dequantised
by the reference package, not the runtime's code) and compares logits per position, each layer's
output, each expert selection, greedy generation through the KV cache against full recomputation,
and perplexity. `parity_report.run_parity` returns all of it with a verdict and E3 receipts bound to
the machine, the model and its weights' SHA-256, the runtime's source fingerprint and the benchmark.

## Speculative execution and grouped expert reads (v0.5)

`basp/runtime/speculative.py` adds speculative decoding to the physical runtime:

```
drafter.propose(committed, k)        ModelDrafter (a small model, same tokenizer, its own KV cache)
                                     NgramDrafter (prompt lookup: no model call)
target.prefill(pending + proposals)  one pass over k+1 rows; MoE layers group the rows by expert,
                                     so each selected expert is fetched once for all of them
accept                               greedy: while proposal == target's argmax
                                     sampling: min(1, p/q), residual max(0, p - q) on rejection
target.truncate(...)                 rejected proposals leave the KV cache
AdaptiveController.choose()          k maximising expected tokens / (k c_draft + a + b (k+1)),
                                     acceptance and costs measured online; k = 0 is plain decoding
```

The weight store fetches the experts a layer needs as one group, dequantising missing ones on
worker threads (NumPy releases the interpreter lock), holds a reference on each until the layer
is done, and counts the stored bytes of every expert a pass needs — the quantity speculative
grouping reduces. With `embedding="rows"` token embeddings are read row by row from the file
instead of being kept, which, with on-demand experts, runs Qwen3-30B-A3B in about 10 GB.

`basp/runtime/prefix_cache.py` caches the KV rows of block-aligned prompt prefixes (charged to
the memory domain, evicted least recently used) and copies them into new sessions that share
the prefix.

`basp/backends/torch_cpu` is a second implemented backend (optional, PyTorch's CPU kernels):
the same operations, selected by the plan when preferred, and verified against transformers by
the same harness. It exists because NumPy's BLAS multiplies 2–9 rows by a matrix at about four
times the cost of one row, which hides the benefit of verifying several tokens per pass.

## Extending

* **A calibrated machine**: follow [calibration.md](calibration.md), then pass `--base-profile`
  to `benchmarks.explore` and `benchmarks.run_report`.
* **A new architecture**: add an entry to `ARCHITECTURES` and a branch in `configure`; add it to
  the design space's constraints in `experiments/explore.py`.
* **A different estimate**: edit `configs/design_estimates.json`; every design, price and corner
  follows.
* **A new hardware profile**: add a JSON file to `configs/` (see `desktop_gpu12.json`).
* **A recorded routing trace**: save it in the `RoutingTrace` JSON format and pass it to
  `Setup.run`.
* **A new execution backend**: implement `ExecutionBackend` and declare in `capabilities()` the
  operations it runs; register it in `DeviceManager.discover`. It must pass
  `tests/test_v04_verification.py` before it is used for any measurement.
* **A new dataflow mode**: add a candidate generator in `basp/dataflow/` and register it in
  `adaptive.py`; if it needs a new kernel shape, express it as phases.
