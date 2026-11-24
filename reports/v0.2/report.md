# BASP Dataflow v0.2 -- architecture discovery: simulation report

> **Read this first.** Every performance number below is the output of a simulator that has **not been validated against measurements** (section 2). Host and GPU parameters come from `configs/desktop_gpu12.json`, an illustrative profile. Everything about
> compute-memory modules (performance, power, price) is an estimate with a
> pessimistic-nominal-optimistic range (`configs/design_estimates.json`) that no measurement on
> existing hardware can narrow; routing is synthetic. Figures written `x (a-b)` give the nominal
> value and the range over those estimates. Regenerate with `python -m benchmarks.run_report`
> (deterministic).

## 1. Hardware development decision

**NO-GO (3 criteria failed, 3 not evaluated)** for the project's first workload, and **NO-GO (1 criterion failed, 3 not evaluated)** in the
most favourable regime found among published model shapes. Hardware development is not justified by the evidence so far. First blocker: Simulator calibration error: 0 validation measurements (at least 8 needed).

Regime: Qwen3-30B-A3B, single-user decoding, 12 GiB GPU

| criterion | threshold | result | status | evidence |
|---|---|---|---|---|
| Simulator calibration error | <= 10% median, <= 20% p95 | 0 validation measurements (at least 8 needed) | not evaluated | none |
| Throughput improvement | >= 2x | 1.22x the conventional baseline (H Hierarchical, pessimistic estimates) | fail | simulated, uncalibrated |
| Energy efficiency | >= 2x | 0.84x the conventional baseline (H Hierarchical, pessimistic estimates) | fail | simulated, uncalibrated |
| Cost efficiency | >= 2x | 0.14x the conventional baseline (H Hierarchical, pessimistic estimates) | fail | simulated, uncalibrated |
| Four-module scaling efficiency | >= 70% | 72% from one to four modules (E Tensor-parallel on the 30B MoE, the best-scaling architecture) | pass | simulated, uncalibrated |
| Scheduler overhead | <= 5% | 1.4% to 27.3% for 28 plan evaluations per batch at an assumed 50-1000 ns each; needs <= 183 ns per evaluation | not evaluated | estimated; no production scheduler exists |
| Numerical correctness | relative error <= 0.002 | 4.5e-04 on a small random-weight model; real model weights not verified | not evaluated | measured |
| Model quality | no material degradation | routing is never altered and only lossless-equivalent formats are exchanged | pass | by construction |

Regime: Llama-3.3-70B, single-user decoding, 12 GiB GPU (most favourable regime found among published model shapes)

| criterion | threshold | result | status | evidence |
|---|---|---|---|---|
| Simulator calibration error | <= 10% median, <= 20% p95 | 0 validation measurements (at least 8 needed) | not evaluated | none |
| Throughput improvement | >= 2x | 3.24x the conventional baseline (E Tensor-parallel, pessimistic estimates) | pass | simulated, uncalibrated |
| Energy efficiency | >= 2x | 3.32x the conventional baseline (E Tensor-parallel, pessimistic estimates) | pass | simulated, uncalibrated |
| Cost efficiency | >= 2x | 0.37x the conventional baseline (E Tensor-parallel, pessimistic estimates) | fail | simulated, uncalibrated |
| Four-module scaling efficiency | >= 70% | 72% from one to four modules (E Tensor-parallel on the 30B MoE, the best-scaling architecture) | pass | simulated, uncalibrated |
| Scheduler overhead | <= 5% | 1.4% to 27.3% for 28 plan evaluations per batch at an assumed 50-1000 ns each; needs <= 183 ns per evaluation | not evaluated | estimated; no production scheduler exists |
| Numerical correctness | relative error <= 0.002 | 4.5e-04 on a small random-weight model; real model weights not verified | not evaluated | measured |
| Model quality | no material degradation | routing is never altered and only lossless-equivalent formats are exchanged | pass | by construction |

Ratios compare the best module-based architecture with architecture A (conventional CPU/GPU with
per-expert caching) at the pessimistic corner of every estimate. The thresholds are proposed
acceptance criteria (`configs/hardware_gate.json`), not results. A criterion that cannot be
evaluated counts against the decision. In the favourable regime the conventional system runs most of a 37 GiB model on its CPU, so those ratios depend directly on the host's assumed arithmetic rate and memory bandwidth -- the parameters calibration replaces first.

## 2. Calibration status

Validation measurements available: **0** (at least
8 required); status: **not evaluated**.

Nothing has been measured yet.

The procedure for the calibration machines is in `docs/calibration.md`; machine names map to
profiles as `i9-11900kf-3080ti` -> `configs/i9_11900kf_rtx3080ti.json`, `macbook-m2` -> `configs/macbook_m2.json`. Until `python -m benchmarks.calibrate` has fitted them, these profiles are
priors built from public specifications and assumptions. For orientation, this is what the simulator
currently predicts for conventional layer offloading on the primary machine (a 4.85 bits-per-weight
file is assumed); measurements will show how wrong these are:

| model | layers on GPU | predicted tokens/s |
|---|---|---|
| Qwen3-4B | 36 | 194 |
| Qwen3-8B | 36 | 106 |
| Qwen3-8B | 0 | 7.77 |
| Qwen3-14B | 40 | 57.9 |
| Qwen3-14B | 20 | 7.5 |
| Qwen3-30B-A3B | 0 | 18.9 |
| Qwen3-30B-A3B | 16 | 26.1 |
| Qwen3-30B-A3B | 32 | 42.3 |

Calibration and validation measurements are separate sets, declared when they are recorded; a
benchmark used to tune a profile cannot also validate it. Calibrating against an existing CPU and
GPU does not narrow any estimate for hardware that does not exist.

## 3. Eight architectures in one simulator

Qwen3-30B-A3B (30.5 B parameters, 16.0 GiB at 4.5 bits per weight),
single-user decoding, context 2048; a 12 GiB GPU and, where used, four modules with two DDR5
channels and 1024 MAC lanes each. Ratios are given at the nominal / pessimistic corner.

| architecture | tokens/s | tokens/J | est. price (USD) | tokens/s per k$ | throughput vs A | energy vs A | per-dollar vs A |
|---|---|---|---|---|---|---|---|
| A Centralized | 164 | 1.07 | 1,692 (1,228-2,420) | 96.7 (67.6-133.3) | 1.00x / 1.00x | 1.00x / 1.00x | 1.00x / 1.00x |
| B Distributed | 46 (38-59) | 0.42 (0.21-0.66) | 8,072 (4,459-20,340) | 5.8 (1.8-13.1) | 0.28x / 0.23x | 0.39x / 0.19x | 0.06x / 0.03x |
| C Pipeline | 28 (23-35) | 0.31 (0.18-0.44) | 8,072 (4,459-20,340) | 3.5 (1.1-7.9) | 0.17x / 0.14x | 0.29x / 0.17x | 0.04x / 0.02x |
| D Expert-local | 107 (90-126) | 0.71 (0.40-0.99) | 8,772 (4,959-21,340) | 12.2 (4.2-25.4) | 0.65x / 0.55x | 0.66x / 0.38x | 0.13x / 0.06x |
| E Tensor-parallel | 132 (111-153) | 0.84 (0.48-1.16) | 8,772 (4,959-21,340) | 15.0 (5.2-30.9) | 0.80x / 0.68x | 0.78x / 0.45x | 0.16x / 0.08x |
| F Adaptive hybrid | 200 (182-214) | 1.29 (0.83-1.61) | 8,772 (4,959-21,340) | 22.8 (8.5-43.2) | 1.22x / 1.11x | 1.20x / 0.78x | 0.24x / 0.13x |
| G Spatial dataflow | 110 (93-130) | 0.72 (0.41-1.02) | 8,772 (4,959-21,340) | 12.5 (4.4-26.2) | 0.67x / 0.57x | 0.68x / 0.39x | 0.13x / 0.06x |
| H Hierarchical | 218 (200-244) | 1.38 (0.90-1.76) | 8,072 (4,309-21,040) | 27.0 (9.5-56.5) | 1.33x / 1.22x | 1.29x / 0.84x | 0.28x / 0.14x |

* Fastest at nominal estimates: **H Hierarchical**, 1.33x architecture A.
* Most throughput per dollar: **A Centralized** (97 tokens/s per 1000 USD;
  the best module-based design reaches 27).
* B and C run the trunk blocks on memory-bound modules instead of a GPU and reach
  0.28x and 0.17x of A.

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
| one launch per expert | 148 | 1 | baseline |
| experts batched into one launch | 164 | 1.07 | +10.8% |
| persistent kernels | 169 | 1.1 | +14.5% |

v0.1 charged one launch per expert, which understated the conventional baseline by
11%. All results in this report use batched launches.

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
| A: time-only scheduling | 164 | 1.07 | baseline |
| A: energy-aware scheduling (0.01 s per joule) | 162 | 1.39 | -0.8% |

Power gating changes tokens per joule by +0.0% for single-user decoding: no device idles long enough to be gated, so the modules' static power is paid in full. Energy-aware scheduling (0.01 s per joule) changes the hybrid's tokens per joule by +0.0% and the conventional system's by +29.5%, at -0.8% throughput for the latter.

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
| Qwen3-14B (INT8) | 14.8 B | 14.6 | published config | 12 | 12.9 | 4.55 | 16.3 | 12.8 | 12.9 | 1.26x (E) | 0.25x |
| Qwen3-30B-A3B | 30.5 B | 16.0 | published config | 12 | 157 | 109 | 132 | 192 | 205 | 1.31x (H) | 0.28x |
| Llama-3.3-70B | 70.6 B | 37.0 | published config | 12 | 1.58 | 1.65 | 6.04 | 1.87 | 1.87 | 3.83x (E) | 0.76x |
| DeepSeek-V3 | 671.0 B | 351.5 | published config, approximated | 12 | 4.51 | 10.9 | 14.2 | 11.4 | 10.3 | 3.14x (E) | 0.79x |
| MoE-1T (hypothetical) | 1,010.3 B | 529.2 | hypothetical | 12 | 3.94 | 16.7 | 12.5 | 17.5 | 15.8 | 4.45x (F) | 0.88x |
| MoE-3T (hypothetical) | 3,122.3 B | 1,635.7 | hypothetical | 24 | 1.94 | 9.83 | 10.1 | 10.2 | 11.3 | 5.80x (H) | 1.97x |

No module-based architecture is faster than A for Qwen3-4B, Qwen3-8B, Qwen3-14B (each fits on the GPU). Where one is faster, the best is 1.26x to 5.80x A; per estimated dollar the best ratio anywhere is 1.97x.

## 6. Workload types

Qwen3-30B-A3B; each workload is a separate question with its own metric.

| workload | metric | A | E | F | H | best vs A |
|---|---|---|---|---|---|---|
| single-user decoding | tokens/s | 163.6 | 131.6 | 199.7 | 218.1 | 1.33x (H) |
| batched inference (4 sequences in lock step) | aggregate tokens/s | 297.4 | 201.3 | 440.2 | 403.3 | 1.48x (F) |
| concurrent requests (4 independent streams) | aggregate tokens/s | 271.3 | 197.1 | 337.1 | 333.3 | 1.24x (F) |
| long-context attention (32k context) | tokens/s | 74.1 | 73.3 | 86.0 | 91.4 | 1.23x (H) |
| prefill (128-token prompt) | prompt tokens/s | 763.4 | 513.1 | 2,292.0 | 2,186.4 | 3.00x (F) |
| sustained decoding (48 tokens) | tokens/s | 175.4 | 131.5 | 204.3 | 222.1 | 1.27x (H) |

## 7. Design-space exploration

1,980 designs were simulated (evolutionary search) out of 8,030 feasible
combinations of architecture, GPU memory, module count, module memory, memory channels, MAC lanes,
link bandwidth and expert replication (20,736 before constraints), on the 30B MoE
single-user workload. Objectives: tokens per second, tokens per joule, estimated price. Each design
gets a short run (4 tokens, 2 of them warm-up), so absolute figures differ by a few percent
from section 3.

Pareto front (9 designs; a spread of them, cheapest first; the pessimistic-corner
value is in brackets):

| design | tokens/s | tokens/J | est. price USD | tokens/s per k$ |
|---|---|---|---|---|
| A, 12 GiB GPU | 172 (172) | 1.15 | 1,692 (2,420) | 101.5 |
| A, 24 GiB GPU | 241 (241) | 2.34 | 2,792 (3,920) | 86.2 |
| H, 12 GiB GPU + 2 x 32 GiB modules, 4 ch, 1024 lanes, 8 GB/s, 2 replicas | 248 (224) | 1.78 | 5,136 (12,340) | 48.2 |
| H, 24 GiB GPU + 2 x 16 GiB modules, 4 ch, 512 lanes, 8 GB/s | 257 (241) | 2.04 | 5,612 (11,600) | 45.8 |
| H, 24 GiB GPU + 2 x 32 GiB modules, 4 ch, 512 lanes, 8 GB/s, 2 replicas | 257 (241) | 2.09 | 5,724 (11,792) | 44.9 |
| H, 24 GiB GPU + 2 x 16 GiB modules, 4 ch, 1024 lanes, 8 GB/s | 275 (258) | 2.01 | 6,124 (13,648) | 44.9 |
| H, 24 GiB GPU + 2 x 32 GiB modules, 4 ch, 1024 lanes, 8 GB/s, 2 replicas | 276 (258) | 1.97 | 6,236 (13,840) | 44.2 |
| H, 24 GiB GPU + 4 x 16 GiB modules, 4 ch, 1024 lanes, 8 GB/s | 296 (258) | 1.78 | 8,756 (21,676) | 33.9 |
| H, 24 GiB GPU + 8 x 16 GiB modules, 4 ch, 1024 lanes, 8 GB/s | 310 (263) | 1.40 | 14,020 (37,732) | 22.1 |

By architecture:

| architecture | designs evaluated | best tokens/s | fastest design | best tokens/s per k$ | designs on the front |
|---|---|---|---|---|---|
| A Centralized | 2 | 241 | A, 24 GiB GPU | 101.5 | 2 |
| B Distributed | 259 | 97 | B, 8 x 32 GiB modules, 4 ch, 2048 lanes, 16 GB/s, 2 replicas | 21.4 | 0 |
| C Pipeline | 36 | 56 | C, 2 x 16 GiB modules, 4 ch, 2048 lanes, 8 GB/s | 16.3 | 0 |
| D Expert-local | 240 | 191 | D, 24 GiB GPU + 8 x 32 GiB modules, 4 ch, 1024 lanes, 16 GB/s, 2 replicas | 21.9 | 0 |
| E Tensor-parallel | 233 | 250 | E, 24 GiB GPU + 8 x 16 GiB modules, 4 ch, 2048 lanes, 8 GB/s | 22.9 | 0 |
| F Adaptive hybrid | 200 | 265 | F, 24 GiB GPU + 8 x 16 GiB modules, 4 ch, 2048 lanes, 8 GB/s, 2 replicas | 38.5 | 0 |
| G Spatial dataflow | 356 | 201 | G, 24 GiB GPU + 8 x 32 GiB modules, 4 ch, 1024 lanes, 16 GB/s, 2 replicas | 22.7 | 0 |
| H Hierarchical | 654 | 310 | H, 24 GiB GPU + 8 x 32 GiB modules, 4 ch, 2048 lanes, 8 GB/s, 2 replicas | 51.8 | 7 |

Equal cost -- the fastest design within a price budget, against the fastest conventional one:

| budget | fastest design | tokens/s | fastest conventional (A) tokens/s |
|---|---|---|---|
| 2,000 USD | A, 12 GiB GPU | 172 | 172 |
| 5,000 USD | A, 24 GiB GPU | 241 | 241 |
| 10,000 USD | H, 24 GiB GPU + 4 x 64 GiB modules, 4 ch, 1024 lanes, 8 GB/s, 2 replicas | 296 | 241 |
| 20,000 USD | H, 24 GiB GPU + 8 x 32 GiB modules, 4 ch, 2048 lanes, 8 GB/s, 2 replicas | 310 | 241 |

Equal power -- the fastest design within an average power budget:

| budget | fastest design | tokens/s | fastest conventional (A) tokens/s |
|---|---|---|---|
| 130 W | H, 24 GiB GPU + 2 x 64 GiB modules, 4 ch, 512 lanes, 8 GB/s, 2 replicas | 257 | 241 |
| 160 W | H, 24 GiB GPU + 2 x 64 GiB modules, 4 ch, 2048 lanes, 8 GB/s, 2 replicas | 276 | 241 |
| 200 W | H, 24 GiB GPU + 4 x 32 GiB modules, 4 ch, 2048 lanes, 8 GB/s, 2 replicas | 296 | 241 |

The front holds 2 centralized (A), 7 hierarchical (H); no B, C, D, E, F, G design is on it. The fastest design found is 1.29x the fastest conventional system at 5.0x its estimated price (1.09x at 9.6x the price when both are priced with pessimistic estimates).
The conventional side of this space has only two points (a 12 and a 24 GiB GPU of the same assumed
speed): faster or multiple GPUs, which a real buyer with the same budget could choose, are not modelled,
so the equal-cost rows above are generous to the module-based designs.

## 8. Numerical verification

A small attention-free MoE language model with real INT4 weights is decoded by the streaming
kernels following the scheduled plans, and compared with an independent float64 reference.

| dataflow | max relative error | routing identical | tokens identical | dependency violations | accounting violations |
|---|---|---|---|---|---|
| adaptive hybrid, fused graph | 4.45e-04 | True | True | 0 | 0 |
| adaptive hybrid, unfused graph | 4.45e-04 | True | True | 0 | 0 |
| tensor-parallel with partial sums, no host in the loop | 4.45e-04 | True | True | 0 | 0 |

Fused and unfused graphs give bit-identical logits: True. Tensor-parallel
execution with partial sums differs from whole-expert execution by at most
1.9e-04 on logits of magnitude 4.2 (a different
FP32 summation order). Real model weights have not been run.

## 9. What v0.2 corrected in the v0.1 model

* GPU kernels are batched (one launch for back-to-back experts): the conventional baseline is
  faster than v0.1 reported.
* The illustrative desktop's CPU arithmetic rate was raised from 30 to 80 GMAC/s so that CPU-side
  decoding is limited by memory bandwidth, as public llama.cpp figures suggest; the lower value
  roughly halved the conventional baseline for models that do not fit the GPU.
* The output projection's location is chosen by the scheduler instead of being pinned to modules.
* Expert-local scaling loss is identified as imbalance, not an inherent limit of modules.
* Temporaries are released per layer, as their dependencies end.
* The estimator books link time with gap filling and sharing, tightening estimates under contention.

The v0.1 report in `reports/v0.1/` was produced by the v0.1 model and is kept as a record; its
numbers are superseded by this report.
