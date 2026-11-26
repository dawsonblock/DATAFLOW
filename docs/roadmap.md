# Roadmap

The objective is no longer to validate the architecture originally proposed. It is to build an
engineering system that can discover which architecture is best — and to accept the answer,
including "a conventional CPU/GPU configuration is superior".

The order of work is: make the evidence trustworthy, measure existing machines, execute real
models, and only then decide about hardware. The objective is faster and more economical
inference; if conventional GPUs with better scheduling achieve it, BASP succeeds without a chip.
**No FPGA development and no hardware purchases until the gate below is passed on calibrated
evidence.**

## Decisions on record

| Question | Decision |
|---|---|
| Continuous integration | Yes (`.github/workflows/ci.yml`) |
| Primary calibration machine | Intel i9-11900KF + NVIDIA RTX 3080 Ti 12 GB |
| Secondary calibration machine | Apple MacBook Pro M2 (unified-memory comparison) |
| FPGA board | None selected |
| FPGA toolchain | Verilator and Yosys for initial RTL verification; vendor tools after board selection |
| Hardware purchases | Not yet; baseline on existing computers first |

## Releases

Revised after the v0.2 audit, and again for v0.4: real software execution comes *before* the
hardware decision, because runtime measurements are what make that decision credible.

| Release | Objective | Exit requirement | State |
|---|---|---|---|
| v0.3 | Repair and evidence infrastructure | Trustworthy, reproducible results; missing or stale evidence cannot produce GO | **Done** |
| v0.4 | Evidence-gated heterogeneous execution foundation | Receipt-based evidence; physical memory domains; a runtime that executes real GGUF models and is verified against a reference | **Done for the CPU** (below); calibration still unmeasured |
| Calibration | Held-out prediction error within thresholds on the calibration machines | `benchmarks.run_calibration` on each machine | **Ready to run**; nothing measured (needs the machines) |
| v0.5 | Predictive speculative execution (BASP-X, phase 1): fewer weight reads per token | Speculative decoding exact in greedy and in distribution; grouped expert reads measured on a real MoE model | **Done on the CPU** (below): measured, not yet against llama.cpp or KTransformers |
| v0.6 | Repair and validation before expansion | The 14 audit findings fixed with behaviour tests; a pre-declared request-latency protocol with paired analysis; an installable package whose release check runs the reference verification | **Done** (below); the decision study and the native arms are for the target machine |
| v0.7 | A fair native baseline, then at most one bounded native intervention | llama.cpp pinned and tuned on the i9 + 3080 Ti (arm D); a profile showing the addressable share of request time; the intervention (arm E) only if its headroom `1/(1 - f r)` can reach 1.20x | Not started; needs the target machine |
| v0.8 | Native execution backends; adaptive memory and expert management on real hardware | C++/SIMD, CUDA and Metal/GGML behind `ExecutionBackend`, passing the parity harness; measured improvement over a tuned llama.cpp (incl. `--n-cpu-moe`) on the same machine | Only if v0.7's decision study meets its criterion |
| v0.9 | Architecture I and distributed execution | Adaptive hierarchical execution with cost `C(p) = T + λE + μR + γU`, measured against H and E; remote machines only for whole requests or resident expert groups | Not started (deferred from v0.4) |
| v0.10 | Conditional accelerator RTL | Synthesis and formal verification of a quantised matrix-vector unit | Blocked on the gate |
| v1.0 | Validated inference product | Reproducible hardware benchmarks | Blocked on the gate |

### Evidence levels

| Level | Meaning |
|---|---|
| E0 | Theoretical estimate, argument, or guessed parameter |
| E1 | Uncalibrated simulation |
| E2 | Simulation whose model of existing hardware passed held-out validation |
| E3 | Direct measurement |

Results on existing GPUs are never presented as measurements of hypothetical modules: before a
prototype exists, module claims are at most E2. Every gate criterion states its required level.

## Release v0.5 — Predictive Speculative Execution (BASP-X, phase 1)

The aim changes from executing the same work on faster hardware to **doing less work per token**:
read each expert's weights once for several tokens, and only when the tokens they serve are likely
to be kept. Everything below was measured on the verification machine (four CPU cores, NumPy and
PyTorch kernels); figures are in report section 11 and `evidence/speculative/`.

| Item | Where | Result (E3, this machine) |
|---|---|---|
| Speculative decoding: draft model or prompt lookup proposes, the target verifies `k + 1` rows in one pass; greedy acceptance (exact tokens) and rejection sampling (exact distribution); KV rollback | `basp/runtime/speculative.py` | every greedy run on real models produced exactly the target's tokens; sampling passes a chi-square test against the target's distribution |
| Adaptive controller: chooses `k` per round from measured acceptance and measured draft and verify costs; `k = 0` is plain decoding | same | backs off where speculation does not pay |
| Grouped expert execution: each layer fetches the union of its rows' experts once, dequantising in parallel; stored bytes counted | `basp/runtime/weights.py`, `model.py` | Qwen3-30B-A3B real routing: 8.0 distinct experts per layer per token alone, 4.7 when 4 tokens share a pass, 3.6 for 8 (up to 2.2x fewer reads) |
| Speculative decoding on Qwen3-30B-A3B (0.6B drafter, k = 4) | `benchmarks/speculative_bench.py` | 58% acceptance, 3.0 tokens per pass, 1.28x fewer expert reads per token; prompt lookup (3% acceptance) *increased* reads to 0.64x: rejected proposals still read experts |
| Expert prediction on real routing | `basp/scheduler/predictor.py` evaluated on the recorded trace | 50% of the selected experts are among 8 predicted, 69% among 16, 85% among 32 |
| Real routing trace of Qwen3-30B-A3B in the simulator's format | `evidence/traces/` | replaces synthetic routing in the exploration and the report |
| Memory-lean execution of a 30B MoE model (embedding rows on demand, experts on demand) | `WeightStore(embedding="rows")` | runs in about 10 GB on a 15 GB machine |
| Shared-prefix KV cache | `basp/runtime/prefix_cache.py` | see report section 11 |
| PyTorch CPU backend (optional) behind the same contract, verified by the same harness | `basp/backends/torch_cpu` | needed because NumPy's BLAS makes multi-row verification cost about 4x a single row |

Not done in v0.5, and why: a native C++/SIMD/CUDA engine (v0.6); comparison with llama.cpp or
KTransformers (they must run on the same machine, and building them was not possible in the
development container); mixed-precision quantisation with quality testing (needs its own
sensitivity study and held-out evaluation; deferred rather than guessed); drafting by the target's
own early layers or a learned expert predictor; overlap of prefetching with computation.

## Release v0.4 — Evidence-Gated Heterogeneous Execution

| Item | Where | Evidence |
|---|---|---|
| Evidence v2: content-addressed receipts binding each value to its identities (source, configuration, profile, validation, machine, model, weights, runtime, benchmark); metric-specific qualification; staleness | `basp/evidence/` | `tests/test_v04_evidence.py`: invalid values (NaN, inf, negative) rejected; a throughput receipt never qualifies energy; stale or other-model receipts rejected |
| Gate decided from receipts | `basp/experiments/gate.py`, report section 1 | E1 simulation can fail a criterion, never pass it; an E0 scheduler estimate never passes; missing quality is "not evaluated" |
| Calibration repairs: repeated runs merge into one benchmark; each machine judged alone; stale measurements (changed executable or weights, full SHA-256) excluded; profiles qualify only by fingerprint; run keys include repetitions and identities | `basp/calibration/validate.py`, `basp/experiments/calibrate.py`, `benchmarks/run_calibration.py` | eight repeats count once; a machine cannot borrow another's benchmarks; a tuned profile is not validated |
| Calibration coverage and performance-curve schema | `basp/calibration/coverage.py`, `curves.py` | the report lists which regimes each machine's validation covers |
| Physical memory domains: shared accounting, budget invariant, admission control with pressure feedback, refcounted eviction safety, asynchronous migration, prefetch budget, hierarchy | `basp/memory/` | `tests/test_v04_memory.py` (2,000-step random churn keeps the invariant) |
| Expert-cache policies LRU, LFU, cost-aware `V_e = (N_e(T_cpu − T_gpu) − T_migration)/M_e`, predictive; evaluation harness | `basp/memory/expert_cache.py`, report section 10 | E1 comparison on synthetic routing |
| Unified-memory design repair | `basp/dse/hardware.py` | designs on the M2 base share one pool; the integrated GPU is not priced separately |
| Physical runtime, separate from the simulator: device manager, execution plans that fail before running, dispatcher, command queues, sessions, generation, telemetry; `ExecutionBackend` (capabilities, allocate, upload, submit, synchronize, release, telemetry) | `basp/runtime/`, `basp/backends/execution.py` | unavailable backends refuse every call |
| Real GGUF execution: memory-mapped tensors, dequantisation of 11 formats (bit-identical to `gguf`), Qwen3 dense and MoE forward pass, KV cache, greedy and sampled generation | `basp/models/quants.py`, `gguf_tensors.py`, `basp/backends/cpu/` | `tests/test_v04_runtime.py` |
| Verification harness against transformers: logits, layer outputs, routing, generation, perplexity, parity report with receipts | `basp/verification/`, `benchmarks/verify_runtime.py` | Qwen3-0.6B Q8_0: relative L2 2.8e-6, 16 identical greedy tokens; tiny dense and MoE models in CI; a broken runtime fails |

Deferred, and stated as such everywhere: architecture I (adaptive hierarchical execution); the
native C++, CUDA and Metal engine (the backends exist only as unavailable declarations — there is
no GPU inference); a real MoE model end to end (the float32 runtime would need about 120 GiB for
Qwen3-30B-A3B); a native tokenizer; prefetching inside the runtime.

## Release v0.3 — Evidence-Gated Adaptive Execution Foundation

Defects found in the v0.2 audit, each reproduced and covered by a regression test
(`tests/test_v03_evidence.py`):

| Defect | Consequence | Fix |
|---|---|---|
| Calibration report read a `machine` field the scored rows did not have | Report crashed as soon as validation measurements existed | Rows carry machine, model and kind; the report's lookup is a tested function |
| Gate used architecture E's scaling to qualify architecture H | A candidate could pass on another architecture's evidence | Each criterion is evaluated for one candidate in one regime; borrowed evidence is "not evaluated" |
| Module link bandwidth ignored on the local fabric, but priced | The optimiser chose the cheapest links for free: every hierarchical design on the v0.2 front had 8 GB/s links | Each module's fabric port is its link bandwidth (capped by the fabric) |
| Routing validation accepted NaN coefficients | Invalid traces passed validation | Non-finite coefficients rejected |
| GPU capacity changed without performance | Larger GPUs were modelled as equally fast | GPU classes (12/16/24/32/48 GiB) scale bandwidth and compute; a 32 GiB RTX 5090-class card joins the conventional candidates |
| Real routing traces not used by exploration and report | Experiments always used synthetic routing | `--trace` and `--compression` on both; the report states what it used |
| Apple unified memory double-counted | M2 CPU and GPU each had their own bandwidth and capacity | Memory domains: one shared bus and one capacity pool |

Further problems found while fixing those:

| Problem | Fix |
|---|---|
| Price corners inverted for shared components: the "pessimistic" corner priced host and GPU *high*, which favoured modules | Corners are named from the module proposal's side: pessimistic = cheap, fast conventional parts |
| A prompt was simulated as independent sequences: 512x the KV memory and KV traffic | Prompt rows share one sequence's KV cache; llama-bench's 512-token prompt now simulates |
| CPU arithmetic rate set for decode only: prompt processing several times too slow, handicapping the baseline | 400 GMAC/s priors (250 on the M2), decode still memory-bound |
| llama-bench's `n_depth`, `n_cpu_moe` and physical batch were ignored; one failed configuration aborted the whole matrix | Recorded and modelled; failures logged and skipped |
| transformers 5 routers are not `nn.Linear`: the trace recorder would have found no router | Router modules recognised by weight shape; model-selected experts cross-checked |
| Fitting re-simulated every measurement for every parameter: about five minutes per fit | Dependencies probed once: about half a minute |

New infrastructure: evidence levels and provenance (`basp/evidence.py`: source fingerprint,
configuration and input hashes, staleness detection; `provenance.json` next to each report); a
pre-registered calibration plan and one-command runner (`benchmarks/run_calibration.py`); a GGUF
header reader (`basp/models/gguf.py`: exact bits per weight, shape check, fingerprint; no tensor
data); the backend contract a real execution backend implements (`basp/backends/base.py`); a
`CLAUDE.md` with the evidence rules for sessions on the calibration machines.

## Phase 2 investigations

| Investigation | Required improvement |
|---|---|
| Sequential transformer execution | Measure the critical path and the unavoidable dependencies |
| Expert imbalance | Workload-aware expert placement and replication |
| Synchronisation overhead | Dependency events instead of host round trips where the host adds nothing |
| GPU launch overhead | Batched expert execution; persistent kernels |
| Memory contention | Independent channel scheduling |
| Communication overhead | Overlap activation transfers with computation |
| Output projection | Execution location chosen dynamically |
| Energy consumption | Idle-power modelling and power-aware scheduling |
| Scheduler overhead | Greedy versus bounded graph-level optimisation |
| Weight compression | Measured compression ratio and decoding throughput |

## Competing architectures

All are implemented in the same simulator and evaluated on the same workloads.

| | Architecture | Core concept |
|---|---|---|
| A | Centralized | Conventional CPU/GPU execution |
| B | Distributed | Independent compute-memory modules, no GPU in the loop |
| C | Pipeline | Transformer layers assigned to sequential processing stages |
| D | Expert-local | Whole experts stay on their assigned modules; trunk on a GPU |
| E | Tensor-parallel | Each tensor partitioned across all modules |
| F | Adaptive hybrid | Dynamically combines execution strategies |
| G | Spatial dataflow | Persistent operators connected by activation streams; no host in the loop |
| H | Hierarchical | Local compute clusters with separate fast and slow interconnects |

## Benchmark suite

| Model size | Research purpose |
|---|---|
| 1-4B | Kernel validation |
| 7-8B | Fully GPU-resident baseline |
| 14B | Memory pressure and quantization |
| 30B MoE | Expert routing and CPU/GPU offloading |
| 70B | High-capacity heterogeneous execution |
| 671B MoE | Large-scale simulation |
| 1-3T MoE | Long-term architectural feasibility |

Workload types are measured separately: single-user decoding, batched inference, long-context
attention, prefill, concurrent requests and sustained decoding.

## Architectural optimisation engine

Candidate hardware (memory, processors, topology, bandwidth, compute lanes) is generated within
physically realistic constraints, evaluated by the simulator for speed, energy and cost, and
optimised for the Pareto front: exhaustive search for small spaces, Bayesian optimisation for
expensive simulations, evolutionary search for large combinatorial spaces.

## Hardware development gate

Proposed acceptance thresholds — targets, not results.

| Criterion | Threshold |
|---|---|
| Simulator calibration error | <= 10% median, <= 20% at p95, on held-out measurements |
| Throughput improvement | >= 2x in the selected workload regime |
| Energy efficiency | >= 2x tokens per joule |
| Cost efficiency | >= 2x tokens/s per dollar |
| Four-module scaling efficiency | >= 70% |
| Scheduler overhead | <= 5% |
| Numerical correctness | Relative L2 error of logits <= 0.002 against the reference, on the regime's model |
| Model quality | Perplexity <= 1.01x the reference's on the same weights (threshold added in v0.4; none changed) |

Rules of evidence:

* Calibration and validation measurements are separate sets. Matching a benchmark used to tune
  the simulator does not establish predictive accuracy.
* Calibrating against existing CPUs and GPUs says nothing about the power or performance of a
  hypothetical FPGA module. Every figure for hardware that does not exist is labelled as an
  estimate and carries an uncertainty range; the gate is evaluated at the pessimistic end.

## Release v0.2 — Calibrated Architecture Discovery Engine (historical)

| Item | Where | State |
|---|---|---|
| Automated CI and regression testing | `.github/workflows/ci.yml`, `tests/` | Done |
| Hardware profiling and calibration scripts | `benchmarks/profile_hardware.py`, `run_llama_bench.py`, `calibrate.py` | Written and unit-tested; not yet run on real hardware |
| Real MoE routing trace recorder | `benchmarks/record_routing_trace.py` | Written; collection logic tested on stand-in modules; never run on a model |
| Corrected GPU batching and output placement | `simulator/compute.py`, `scheduler/placement.py` | Done |
| Critical-path and scaling bottleneck analysis | `basp/analysis/`, report section 4 | Done |
| Eight competing dataflow architectures | `experiments/architectures.py` | Done |
| Multi-objective architectural optimisation | `basp/dse/`, `benchmarks/explore.py` | Done |
| Equal-cost and equal-power comparisons | report section 7 | Done, with an estimated price model |
| Numerical verification | `basp/verification/`, report section 8 | Done on a small random-weight model; real weights not run |
| Reproducible benchmark reports | `benchmarks/run_report.py` | Done (deterministic) |
| Evidence-gated hardware development decision | `experiments/gate.py`, report section 1 | Done: NO-GO |

"Calibrated" in the release name describes what the system can do, not its current state: the
engine accepts measurements and gates its conclusions on them, and no measurements exist yet.
