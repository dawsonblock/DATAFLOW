# Roadmap

The objective is no longer to validate the architecture originally proposed. It is to build an
engineering system that can discover which architecture is best — and to accept the answer,
including "a conventional CPU/GPU configuration is superior".

The v0.1 simulation missed the 2x throughput, 2x energy and 70% scaling targets on its
illustrative profile. Those results rest on assumed hardware parameters and synthetic routing,
so the order of work is: measure, fix the model, explore alternatives, and only then decide about
hardware. **No FPGA development and no hardware purchases until the gate below is passed.**

## Decisions on record

| Question | Decision |
|---|---|
| Continuous integration | Yes (`.github/workflows/ci.yml`) |
| Primary calibration machine | Intel i9-11900KF + NVIDIA RTX 3080 Ti 12 GB |
| Secondary calibration machine | Apple MacBook Pro M2 (unified-memory comparison) |
| FPGA board | None selected |
| FPGA toolchain | Verilator and Yosys for initial RTL verification; vendor tools after board selection |
| Hardware purchases | Not yet; baseline on existing computers first |

## Phases

| # | Phase | Exit evidence |
|---|---|---|
| 0 | Land v0.1, CI | Tests and lint run on every push |
| 1 | Replace assumptions with measurements | Simulator predicts held-out measured runs within the calibration thresholds |
| 2 | Fix architectural weaknesses, explore alternatives | Every investigation below has a simulated answer; eight architectures compared in one simulator |
| 3 | Hardware decision gate | All thresholds met on calibrated evidence in a chosen workload regime |
| 4 | Real CPU/GPU runtime (v0.8) | Measured tokens/s against a tuned llama.cpp baseline |
| 5 | RTL (v0.9) | Bit-exact against the reference kernels; formal checks; synthesis results |
| 6 | FPGA prototype (v1.0) | Hardware results compared with simulator predictions |

Phases 4-6 start only after phase 3. If the gate fails for every distributed design, phase 4 (a
better CPU/GPU runtime) is the product and phases 5-6 are shelved.

### Status after v0.2

| # | Phase | State |
|---|---|---|
| 0 | CI | Done |
| 1 | Measurements | **Tooling done, nothing measured.** The profiling, benchmark, fitting and validation scripts exist ([calibration.md](calibration.md)); they must be run on the two calibration machines. The hardware-facing parts have never been executed |
| 2 | Investigations and alternatives | Done in simulation, **on an uncalibrated simulator**: all ten investigations, eight architectures, the model-size and workload suites and the design-space exploration are in `reports/v0.2/report.md`. They must be re-run on the calibrated profile |
| 3 | Hardware gate | Evaluated: **NO-GO**. Calibration is not evaluated, and at the pessimistic corner no module-based design reaches 2x the conventional baseline in throughput per dollar in any regime examined |
| 4-6 | Runtime, RTL, FPGA | Not started. No FPGA development and no purchases, by decision |

Next actions, in order:

1. Run the calibration procedure on the i9-11900KF + RTX 3080 Ti, then on the MacBook Pro M2.
2. If validation fails, fix the simulator's model of a conventional system until it passes on
   fresh validation measurements.
3. Record real Qwen3-30B-A3B routing and measure real weight compression; replace the synthetic
   trace and the assumed ratio.
4. Re-run the exploration and the report on the calibrated profile and re-evaluate the gate.
5. Only if the gate passes: select an FPGA board and start RTL with Verilator and Yosys. If it
   does not, the conventional CPU/GPU runtime (phase 4) is the product.

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

## Release v0.2 — Calibrated Architecture Discovery Engine

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
