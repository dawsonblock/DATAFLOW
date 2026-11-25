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

Revised after the v0.2 audit: limited real software execution comes *before* the hardware
decision, because runtime measurements are what make that decision credible.

| Release | Objective | Exit requirement | State |
|---|---|---|---|
| v0.3 | Repair and evidence infrastructure | Trustworthy, reproducible results; missing or stale evidence cannot produce GO | **Done** (below) |
| v0.4 | Hardware calibration | Held-out prediction error within thresholds on both machines | **Ready to run**: `benchmarks.run_calibration`; nothing measured yet |
| v0.5 | Real model execution | Tokens from pretrained GGUF models; intermediate tensors and logits verified against a reference; quality parity | Not started (GGUF header reader and backend contract in place) |
| v0.6 | Native execution backends | CPU (SIMD), CUDA and Metal backends behind the `Payload` contract; compiled C++ scheduler | Not started |
| v0.7 | Adaptive memory and expert management | Measured improvement over a tuned llama.cpp (incl. `--n-cpu-moe`) on the same machine | Not started |
| v0.8 | Distributed execution; architecture I | Efficient multi-device scaling, measured; adaptive hierarchical tensor-parallel execution against H and E | Not started |
| v0.9 | Conditional accelerator RTL | Synthesis and formal verification of a quantised matrix-vector unit | Blocked on the gate |
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
| Numerical correctness | Within predefined reference tolerances |
| Model quality | No material degradation |

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
