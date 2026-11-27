# BASP Dataflow v0.3 -- evidence-gated architecture discovery: simulation report

> **Read this first.** Every performance number below is the output of a simulator that has **not been validated against measurements** (section 2). Host and GPU parameters come from `configs/desktop_gpu12.json`, an illustrative profile. Everything about
> compute-memory modules (performance, power, price) is an estimate with a
> pessimistic-nominal-optimistic range (`configs/design_estimates.json`) that no measurement on
> existing hardware can narrow; routing is synthetic (no recorded trace was given); compression ratios are assumed. Figures written `x (a-b)` give
> the nominal value and the range over those estimates. Evidence levels: E0 estimate or argument,
> E1 uncalibrated simulation, E2 simulation validated on existing hardware, E3 measurement.
> Regenerate with `python -m benchmarks.run_report` (deterministic; see `provenance.json`).

## 1. Hardware development decision

**NO-GO (4 criteria failed, 4 not evaluated)** for the project's first workload, and **NO-GO (2 criteria failed, 6 not evaluated)** in the
most favourable regime found among published model shapes. Hardware development is not justified by the evidence so far. First blocker: Simulator calibration error: 0 validation measurements (at least 8 needed).

Regime: Qwen3-30B-A3B, single-user decoding, 12 GiB GPU

| criterion | threshold | result | status | evidence | required |
|---|---|---|---|---|---|
| Simulator calibration error | <= 10% median, <= 20% p95 | 0 validation measurements (at least 8 needed) | not evaluated | none | E3 |
| Throughput improvement | >= 2x | 1.15x the conventional baseline (H Hierarchical, pessimistic estimates) | fail | E1 uncalibrated simulation | E2 |
| Energy efficiency | >= 2x | 0.55x the conventional baseline (H Hierarchical, pessimistic estimates) | fail | E1 uncalibrated simulation | E2 |
| Cost efficiency | >= 2x | 0.07x the conventional baseline (H Hierarchical, pessimistic estimates) | fail | E1 uncalibrated simulation | E2 |
| Four-module scaling efficiency | >= 70% | 32% from one to four modules | fail | E1 uncalibrated simulation | E2 |
| Scheduler overhead | <= 5% | 1.4% to 27.3% for 28 plan evaluations per batch at an assumed 50-1000 ns each; needs <= 183 ns per evaluation | not evaluated | E0 estimate or argument; no production scheduler exists | E3 |
| Numerical correctness | relative error <= 0.002 | 4.5e-04 on a small random-weight model; real weights not run | not evaluated | E3 measurement; wrong model | E3 |
| Model quality | no material degradation | not measured (the runtime never selects experts and exchanges only lossless-equivalent weight formats; dataflows differ only in FP32 summation order) | not evaluated | E0 estimate or argument; by construction | E3 |

Regime: Llama-3.3-70B, single-user decoding, 12 GiB GPU (most favourable regime found among published model shapes)

| criterion | threshold | result | status | evidence | required |
|---|---|---|---|---|---|
| Simulator calibration error | <= 10% median, <= 20% p95 | 0 validation measurements (at least 8 needed) | not evaluated | none | E3 |
| Throughput improvement | >= 2x | 2.72x the conventional baseline (E Tensor-parallel, pessimistic estimates) | not evaluated | E1 uncalibrated simulation | E2 |
| Energy efficiency | >= 2x | 1.26x the conventional baseline (E Tensor-parallel, pessimistic estimates) | fail | E1 uncalibrated simulation | E2 |
| Cost efficiency | >= 2x | 0.17x the conventional baseline (E Tensor-parallel, pessimistic estimates) | fail | E1 uncalibrated simulation | E2 |
| Four-module scaling efficiency | >= 70% | 93% from one to four modules | not evaluated | E1 uncalibrated simulation | E2 |
| Scheduler overhead | <= 5% | 1.4% to 27.3% for 28 plan evaluations per batch at an assumed 50-1000 ns each; needs <= 183 ns per evaluation | not evaluated | E0 estimate or argument; no production scheduler exists | E3 |
| Numerical correctness | relative error <= 0.002 | 4.5e-04 on a small random-weight model; real weights not run | not evaluated | E3 measurement; wrong model | E3 |
| Model quality | no material degradation | not measured (the runtime never selects experts and exchanges only lossless-equivalent weight formats; dataflows differ only in FP32 summation order) | not evaluated | E0 estimate or argument; by construction | E3 |

Ratios compare the best module-based architecture with architecture A (conventional CPU/GPU with
per-expert caching) at the pessimistic corner of every estimate. The thresholds are proposed
acceptance criteria (`configs/hardware_gate.json`), not results. A criterion that cannot be
evaluated counts against the decision. In the favourable regime the conventional system runs most of a 37 GiB model on its CPU, so those ratios depend directly on the host's assumed arithmetic rate and memory bandwidth -- the parameters calibration replaces first.

## 2. Calibration status

Validation measurements available: **0** (at least
8 required); status: **not evaluated**.

Nothing has been measured yet.

The procedure for the calibration machines is in `docs/calibration.md`; machine names map to
profiles as `i9-11900kf-3080ti` -> `configs/i9_11900kf_rtx3080ti.json`, `macbook-m2` -> `configs/macbook_m2.json`, `colab-t4` -> `configs/colab_t4.json`. Until `python -m benchmarks.calibrate` has fitted them, these profiles are
priors built from public specifications and assumptions. For orientation, this is what the simulator
currently predicts for conventional layer offloading on the primary machine (a 4.85 bits-per-weight
file is assumed); measurements will show how wrong these are:

| model | layers on GPU | predicted tokens/s |
|---|---|---|
| Qwen3-4B | 36 | 194 |
| Qwen3-8B | 36 | 106 |
| Qwen3-8B | 0 | 7.78 |
| Qwen3-14B | 40 | 57.9 |
| Qwen3-14B | 20 | 7.51 |
| Qwen3-30B-A3B | 0 | 19.2 |
| Qwen3-30B-A3B | 16 | 26.5 |
| Qwen3-30B-A3B | 32 | 42.8 |

Calibration and validation measurements are separate sets, declared when they are recorded; a
benchmark used to tune a profile cannot also validate it. Calibrating against an existing CPU and
GPU does not narrow any estimate for hardware that does not exist.

## 3. Eight architectures in one simulator

Qwen3-30B-A3B (30.5 B parameters, 16.0 GiB at 4.5 bits per weight),
single-user decoding, context 2048; a 12 GiB GPU and, where used, four modules with two DDR5
channels and 1024 MAC lanes each. Ratios are given at the nominal / pessimistic corner.

| architecture | tokens/s | tokens/J | est. price (USD) | tokens/s per k$ | throughput vs A | energy vs A | per-dollar vs A |
|---|---|---|---|---|---|---|---|
| A Centralized | 171 | 1.61 | 1,692 (1,228-2,420) | 100.9 (70.6-139.1) | 1.00x / 1.00x | 1.00x / 1.00x | 1.00x / 1.00x |
| B Distributed | 46 (38-59) | 0.42 (0.21-0.66) | 8,072 (5,151-19,648) | 5.8 (1.9-11.4) | 0.27x / 0.22x | 0.26x / 0.13x | 0.06x / 0.01x |
| C Pipeline | 28 (23-35) | 0.31 (0.18-0.44) | 8,072 (5,151-19,648) | 3.5 (1.2-6.8) | 0.16x / 0.14x | 0.19x / 0.11x | 0.03x / 0.01x |
| D Expert-local | 107 (90-126) | 0.71 (0.40-0.99) | 8,772 (6,151-20,148) | 12.2 (4.5-20.5) | 0.62x / 0.53x | 0.44x / 0.25x | 0.12x / 0.03x |
| E Tensor-parallel | 132 (111-153) | 0.84 (0.48-1.16) | 8,772 (6,151-20,148) | 15.0 (5.5-25.0) | 0.77x / 0.65x | 0.52x / 0.30x | 0.15x / 0.04x |
| F Adaptive hybrid | 200 (182-214) | 1.29 (0.83-1.61) | 8,772 (6,151-20,148) | 22.8 (9.0-34.9) | 1.17x / 1.07x | 0.80x / 0.52x | 0.23x / 0.07x |
| G Spatial dataflow | 110 (93-130) | 0.72 (0.41-1.02) | 8,772 (6,151-20,148) | 12.5 (4.6-21.2) | 0.64x / 0.55x | 0.45x / 0.26x | 0.12x / 0.03x |
| H Hierarchical | 210 (197-235) | 1.34 (0.89-1.72) | 8,072 (5,501-19,848) | 26.0 (9.9-42.7) | 1.23x / 1.15x | 0.83x / 0.55x | 0.26x / 0.07x |

* Fastest at nominal estimates: **H Hierarchical**, 1.23x architecture A.
* Most throughput per dollar: **A Centralized** (101 tokens/s per 1000 USD;
  the best module-based design reaches 26).
* B and C run the trunk blocks on memory-bound modules instead of a GPU and reach
  0.27x and 0.16x of A.

Prices are estimates for ratio purposes only, and the module-based systems contain more memory and
compute than the conventional one: this compares architectures on their natural hardware.

## 4. Phase 2 investigations

Each experiment changes one thing. The question is whether a weakness is an implementation problem
or a property of the architecture.

### 4.1 Sequential execution: the critical path

Expert-local architecture (D), four modules, per token (10.98 ms):

| on the critical path | ms per token | share |
|---|---|---|
| computing (kernels) | 4.32 | 39% |
| waiting for a busy device | 6.21 | 57% |
| returning results | 0.35 | 3% |
| waiting for inputs | 0 | 0% |
| host round trips | 0.098 | 1% |

By operation: experts 8.94 ms, trunk blocks 1.45 ms, output projection
0.35 ms, reductions 0.14 ms. Only the expert phase
(8.94 ms on the path; 19.17 ms of kernel work in total) can be divided among
modules; the remaining 2.04 ms is a serial dependency chain. If the expert work divided
perfectly, four modules would reach 78% scaling efficiency and eight
60%: **the 70% target is reachable at four modules only with near-perfect balance, and not at eight** while the trunk stays serial.

### 4.2 Expert imbalance

| placement | modules | tokens/s | scaling efficiency |
|---|---|---|---|
| one module (baseline) | 1 | 45.9 | 100% |
| round-robin placement | 4 | 91.2 | 50% |
| frequency-balanced placement | 4 | 91.3 | 50% |
| co-activation-aware placement | 4 | 91 | 50% |
| co-activation-aware, every expert on 2 modules | 4 | 107 | 58% |
| tensor-parallel (every expert on all modules) | 4 | 132 | 72% |

With whole experts on modules, the experts selected for a token land unevenly (a balls-in-bins
effect), and the layer waits for the fullest module. Relative to round-robin placement:
frequency-balanced +0.1%, co-activation-aware -0.2%, two copies of every expert
+16.9% (the scheduler picks the emptier holder), tensor-parallel +44.2% (no
imbalance by construction). Co-activation-aware placement needs far more profiling data than a
short trace provides and is regularised towards load balancing until then.

### 4.3 Scaling with module count

| architecture | modules | tokens/s | speed-up | scaling efficiency |
|---|---|---|---|---|
| D expert-local | 1 | 45.9 | 1.00x | 100% |
| E tensor-parallel | 1 | 45.9 | 1.00x | 100% |
| D expert-local | 2 | 67.3 | 1.47x | 73% |
| D with 2 replicas | 2 | 81.3 | 1.77x | 89% |
| E tensor-parallel | 2 | 81.3 | 1.77x | 88% |
| D expert-local | 4 | 91 | 1.98x | 50% |
| D with 2 replicas | 4 | 107 | 2.32x | 58% |
| E tensor-parallel | 4 | 132 | 2.87x | 72% |
| D expert-local | 8 | 118 | 2.58x | 32% |
| D with 2 replicas | 8 | 131 | 2.85x | 36% |
| E tensor-parallel | 8 | 189 | 4.11x | 51% |

Four-module efficiency: expert-local 50%, tensor-parallel 72%.
**The low efficiency measured in v0.1 (47%) was a property of expert-local placement, not of distributed execution as such.**

### 4.4 Synchronisation

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| host in the loop (latencies x1) | 91 | 0.616 | baseline |
| dependency events only (latencies x1) | 91.9 | 0.621 | +0.9% |
| host in the loop (latencies x8) | 55.9 | 0.399 | baseline |
| dependency events only (latencies x8) | 58.6 | 0.416 | +4.7% |
| 4 sequences in lock step (one barrier per layer) | 178 | 1.12 | baseline |
| 4 independent streams (no barrier) | 191 | 1.13 | +7.0% |

Removing the host round trip gains +0.9% at the profile's latencies and
+4.7% when they are eight times larger. A lock-step batch waits for its slowest
module at every layer; independent streams avoid that barrier (+7.0%).

### 4.5 GPU launch overhead

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| one launch per expert | 157 | 1.44 | baseline |
| experts batched into one launch | 171 | 1.61 | +8.7% |
| persistent kernels | 173 | 1.62 | +10.0% |

v0.1 charged one launch per expert, which understated the conventional baseline by
9%. All results in this report use batched launches.

### 4.6 Memory channels

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| D: channels pooled per module | 91 | 0.616 | baseline |
| D: independent channel engines | 71.1 | 0.495 | -21.9% |
| E: channels pooled per module | 132 | 0.841 | baseline |
| E: independent channel engines | 130 | 0.835 | -0.9% |

Running each memory channel as an independent engine does not help (-21.9% for D, -0.9% for E): a kernel already uses the module's full bandwidth, and splitting it halves each engine's rate while worsening imbalance. Memory contention is not the bottleneck; memory bandwidth is.

### 4.7 Communication

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| activations sent after routing (latencies x1) | 91 | 0.616 | baseline |
| activations streamed eagerly (latencies x1) | 91.8 | 0.62 | +0.9% |
| activations sent after routing (latencies x8) | 55.9 | 0.399 | baseline |
| activations streamed eagerly (latencies x8) | 58.4 | 0.415 | +4.5% |

Streaming the hidden state to the modules before the routing decision overlaps one transfer latency
per layer: worth +0.9% here and +4.5% at eight times the latency.

### 4.8 Output projection

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| row shards on the modules | 87.1 | 0.587 | baseline |
| on the trunk device | 91 | 0.616 | +4.5% |
| both; location chosen per token | 91 | 0.616 | +4.5% |

The projection is placed on both the modules and the trunk device when it fits, and the scheduler
picks per token (+4.5% against module shards only). v0.1 pinned it to the modules.

### 4.9 Energy

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| F: no power gating | 200 | 1.29 | baseline |
| F: idle devices gated | 200 | 1.29 | +0.0% |
| F: energy-aware scheduling (0.001 s per joule) | 200 | 1.29 | +0.0% |
| F: energy-aware scheduling (0.01 s per joule) | 200 | 1.29 | +0.0% |
| A: time-only scheduling | 171 | 1.61 | baseline |
| A: energy-aware scheduling (0.01 s per joule) | 171 | 1.61 | +0.0% |

Power gating changes tokens per joule by +0.0% for single-user decoding: no device idles long enough to be gated, so the modules' static power is paid in full. Energy-aware scheduling (0.01 s per joule) changes the hybrid's tokens per joule by +0.0% and the conventional system's by +0.0%, at +0.0% throughput for the latter.

### 4.10 Scheduler

| scheduler | tokens/s | plan evaluations per batch | improvements found | overhead at 50 / 200 / 1000 ns per evaluation |
|---|---|---|---|---|
| greedy earliest-finish | 200 | 27.9 | 0 | 1.4% / 5.5% / 27.3% |
| one refinement pass | 200 | 219 | 0 | 10.7% / 42.8% / 213.9% |
| two refinement passes | 200 | 219 | 0 | 10.7% / 42.8% / 213.9% |
| four refinement passes | 200 | 219 | 0 | 10.7% / 42.8% / 213.9% |

Bounded graph-level refinement changed throughput by +0.0% relative to greedy earliest-finish scheduling on this workload while evaluating 7.8x as many plans. It does fix constructed collisions (see the test suite), but mixture layers present few of them; greedy scheduling is the sensible default. The overhead of either depends on a per-evaluation cost that only a production implementation can establish.

### 4.11 Weight compression

Order-0 entropy of INT4 codes plus scales, on synthetic weights:

| weight distribution (synthetic) | bits per weight | saving vs 4.5-bit INT4 |
|---|---|---|
| gaussian | 4.12 | 8.4% |
| laplace | 3.78 | 16.1% |
| student-t (3 d.o.f.) | 3.67 | 18.4% |

Real weights have not been measured. `python -m benchmarks.profile_compression` reports the achieved
ratio and software decode throughput for a weight file; decode throughput of a hardware decoder
remains an estimate.

## 5. Model-size suite: where does each architecture become advantageous?

Single-user decoding on the smallest standard system that holds each model (a 12 GiB GPU, or the
next size up when the trunk blocks alone do not fit; host memory and four modules sized to the
model). Tokens per second; systems for the largest models are hypothetical.

| model | parameters | weights (GiB) | shape | GPU (GiB) | A | D | E | F | H | best module arch. vs A | per dollar vs A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 4.0 B | 2.1 | published config | 12 | 188 | 32.3 | 101 | 188 | 188 | none faster | 0.22x |
| Qwen3-8B | 8.2 B | 4.3 | published config | 12 | 107 | 16.5 | 55.4 | 107 | 107 | none faster | 0.22x |
| Qwen3-14B | 14.8 B | 7.7 | published config | 12 | 60.1 | 8.54 | 30 | 60.1 | 60.1 | none faster | 0.22x |
| Qwen3-14B (INT8) | 14.8 B | 14.6 | published config | 12 | 12.9 | 4.55 | 16.3 | 12.8 | 12.8 | 1.26x (E) | 0.25x |
| Qwen3-30B-A3B | 30.5 B | 16.0 | published config | 12 | 156 | 109 | 132 | 192 | 200 | 1.28x (H) | 0.28x |
| Llama-3.3-70B | 70.6 B | 37.0 | published config | 12 | 1.88 | 1.65 | 6.04 | 1.87 | 1.87 | 3.22x (E) | 0.64x |
| DeepSeek-V3 | 671.0 B | 351.5 | published config, approximated | 12 | 4.69 | 10.9 | 14.2 | 11.4 | 11.6 | 3.02x (E) | 0.76x |
| MoE-1T (hypothetical) | 1,010.3 B | 529.2 | hypothetical | 12 | 4.07 | 16.7 | 12.5 | 17.5 | 17.6 | 4.33x (H) | 0.90x |
| MoE-3T (hypothetical) | 3,122.3 B | 1,635.7 | hypothetical | 24 | 2.03 | 10.1 | 10.4 | 10.5 | 10.6 | 5.23x (H) | 1.93x |

No module-based architecture is faster than A for Qwen3-4B, Qwen3-8B, Qwen3-14B (each fits on the GPU). Where one is faster, the best is 1.26x to 5.23x A; per estimated dollar the best ratio anywhere is 1.93x.

## 6. Workload types

Qwen3-30B-A3B; each workload is a separate question with its own metric.

| workload | metric | A | E | F | H | best vs A |
|---|---|---|---|---|---|---|
| single-user decoding | tokens/s | 170.8 | 131.6 | 199.7 | 209.9 | 1.23x (H) |
| batched inference (4 sequences in lock step) | aggregate tokens/s | 303.0 | 201.3 | 440.2 | 460.1 | 1.52x (H) |
| concurrent requests (4 independent streams) | aggregate tokens/s | 273.4 | 197.1 | 337.1 | 339.1 | 1.24x (H) |
| long-context attention (32k context) | tokens/s | 75.4 | 73.3 | 86.0 | 89.7 | 1.19x (H) |
| prefill (128-token prompt) | prompt tokens/s | 1,251.1 | 516.5 | 2,559.8 | 2,657.0 | 2.12x (H) |
| sustained decoding (48 tokens) | tokens/s | 174.0 | 131.5 | 204.3 | 216.5 | 1.24x (H) |

## 7. Design-space exploration

1,980 designs were simulated (evolutionary search) out of 10,803 feasible
combinations of architecture, GPU memory, module count, module memory, memory channels, MAC lanes,
link bandwidth and expert replication (27,648 before constraints), on the 30B MoE
single-user workload. Objectives: tokens per second, tokens per joule, estimated price. Each design
gets a short run (4 tokens, 2 of them warm-up), so absolute figures differ by a few percent
from section 3.

Pareto front (3 designs; a spread of them, cheapest first; the pessimistic-corner
value is in brackets):

| design | tokens/s | tokens/J | est. price USD | tokens/s per k$ |
|---|---|---|---|---|
| A, 12 GiB GPU | 166 (166) | 1.59 | 1,692 (1,228) | 98.3 |
| A, 24 GiB GPU | 258 (263) | 2.47 | 2,792 (1,528) | 92.5 |
| A, 32 GiB GPU | 438 (438) | 3.49 | 3,592 (2,728) | 121.9 |

By architecture:

| architecture | designs evaluated | best tokens/s | fastest design | best tokens/s per k$ | designs on the front |
|---|---|---|---|---|---|
| A Centralized | 3 | 438 | A, 32 GiB GPU | 121.9 | 3 |
| B Distributed | 201 | 96 | B, 8 x 32 GiB modules, 4 ch, 1024 lanes, 16 GB/s, 2 replicas | 21.4 | 0 |
| C Pipeline | 29 | 56 | C, 2 x 16 GiB modules, 4 ch, 1024 lanes, 32 GB/s | 16.3 | 0 |
| D Expert-local | 222 | 165 | D, 24 GiB GPU + 8 x 16 GiB modules, 4 ch, 1024 lanes, 32 GB/s | 21.9 | 0 |
| E Tensor-parallel | 276 | 226 | E, 32 GiB GPU + 8 x 64 GiB modules, 2 ch, 2048 lanes, 16 GB/s | 22.3 | 0 |
| F Adaptive hybrid | 205 | 438 | F, 32 GiB GPU + 2 x 16 GiB modules, 4 ch, 256 lanes, 8 GB/s | 62.1 | 0 |
| G Spatial dataflow | 378 | 205 | G, 24 GiB GPU + 8 x 32 GiB modules, 4 ch, 1024 lanes, 16 GB/s, 2 replicas | 22.7 | 0 |
| H Hierarchical | 666 | 438 | H, 32 GiB GPU + 2 x 16 GiB modules, 4 ch, 256 lanes, 8 GB/s | 71.2 | 0 |

Equal cost -- the fastest design within a price budget, against the fastest conventional one:

| budget | fastest design | tokens/s | fastest conventional (A) tokens/s |
|---|---|---|---|
| 2,000 USD | A, 12 GiB GPU | 166 | 166 |
| 5,000 USD | A, 32 GiB GPU | 438 | 438 |
| 10,000 USD | A, 32 GiB GPU | 438 | 438 |
| 20,000 USD | A, 32 GiB GPU | 438 | 438 |

Equal power -- the fastest design within an average power budget:

| budget | fastest design | tokens/s | fastest conventional (A) tokens/s |
|---|---|---|---|
| 130 W | A, 32 GiB GPU | 438 | 438 |
| 160 W | A, 32 GiB GPU | 438 | 438 |
| 200 W | A, 32 GiB GPU | 438 | 438 |

The front holds 3 centralized (A); no B, C, D, E, F, G, H design is on it. The fastest design found is a conventional system.
The conventional candidates are one GPU per system, in the classes of `configs/design_estimates.json`
(12, 24 and 32 GiB here, each with its own bandwidth and compute range); systems with several GPUs,
which a buyer with the same budget could choose, are not modelled, so the equal-cost rows above are
still generous to the module-based designs.

## 8. Numerical verification

A small attention-free MoE language model with real INT4 weights is decoded by the streaming
kernels following the scheduled plans, and compared with an independent float64 reference.

| dataflow | max relative error | routing identical | tokens identical | dependency violations | accounting violations |
|---|---|---|---|---|---|
| adaptive hybrid, fused graph | 4.45e-04 | True | True | 0 | 0 |
| adaptive hybrid, unfused graph | 4.45e-04 | True | True | 0 | 0 |
| tensor-parallel with partial sums, no host in the loop | 4.45e-04 | True | True | 0 | 0 |

With the reduction order fixed, fused and unfused graphs give bit-identical logits:
True. Dataflows that accumulate partial sums on devices, or partition experts across
modules, add the same FP32 terms in a different order: the three runs above differ from each other
by at most 1.9e-04 on logits of magnitude 4.2, and select the same experts.
Such differences are at rounding level but are not zero, and a reordered sum could in principle
tip a near-tied routing decision; none was observed. Real model weights have not been run.

## 9. What changed since earlier reports

Corrected in v0.3 (each found in an audit of v0.2 or while fixing it, and covered by a regression test):

* Module links on the local fabric were free: every hierarchical design on the v0.2 front used the
  cheapest link, because the simulator gave it the fabric's bandwidth anyway.
* Price corners favoured modules: the pessimistic corner priced host and GPU high. It now prices them
  low (the strongest baseline), and GPU classes differ in speed, not only in memory.
* A prompt was simulated as independent sequences, each with its own KV cache; it is now one sequence.
* The host CPU's arithmetic rate (80 GMAC/s in v0.2, 30 in v0.1) made prompt processing several times
  too slow; it is 400 GMAC/s, with generation still memory-bound.
* The gate qualified one architecture with another's scaling result; every criterion now concerns one
  candidate in one regime, at a required evidence level, and borrowed evidence counts as missing.
* Unified memory (Apple M2) is one shared bus and one capacity pool, not two.

Corrected in v0.2: batched GPU launches; the output projection's location chosen per token; the v0.1
claim of bit-identical logits under every policy (true only while the reduction order is unchanged).

The reports in `reports/v0.1/` and `reports/v0.2/` were produced by earlier models and are kept as
records; their numbers are superseded by this report.
