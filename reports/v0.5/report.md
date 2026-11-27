# BASP Dataflow v0.5 -- predictive speculative execution: simulation and runtime report

> **Read this first.** Every performance number below is the output of a simulator that has **not been validated against measurements** (section 2). Host and GPU parameters come from `configs/desktop_gpu12.json`, an illustrative profile. Everything about
> compute-memory modules (performance, power, price) is an estimate with a
> pessimistic-nominal-optimistic range (`configs/design_estimates.json`) that no measurement on
> existing hardware can narrow; routing for Qwen3-30B-A3B is recorded (qwen3-30b-a3b.decode.json); other models use synthetic routing; compression ratios are assumed. Figures written `x (a-b)` give
> the nominal value and the range over those estimates. Evidence levels: E0 estimate or argument,
> E1 uncalibrated simulation, E2 simulation validated on existing hardware, E3 measurement.
> Regenerate with `python -m benchmarks.run_report` (with `--trace` for recorded routing; deterministic; see `provenance.json`).

## 1. Hardware development decision

**NO-GO (4 criteria failed, 4 not evaluated)** for the project's first workload, and **NO-GO (2 criteria failed, 6 not evaluated)** in the
most favourable regime found among published model shapes. Hardware development is not justified by the evidence so far. First blocker: Simulator calibration error: 0 validation benchmarks (at least 8 needed).

Regime: Qwen3-30B-A3B, single-user decoding, 12 GiB GPU

| criterion | threshold | result | status | evidence | required |
|---|---|---|---|---|---|
| Simulator calibration error | <= 10% median, <= 20% p95 | 0 validation benchmarks (at least 8 needed) | not evaluated | none | E3 |
| Throughput improvement | >= 2x | 1.01x the conventional baseline (H Hierarchical, pessimistic estimates) | fail | E1 uncalibrated simulation | E2 |
| Energy efficiency | >= 2x | 0.53x the conventional baseline (H Hierarchical, pessimistic estimates) | fail | E1 uncalibrated simulation | E2 |
| Cost efficiency | >= 2x | 0.06x the conventional baseline (H Hierarchical, pessimistic estimates) | fail | E1 uncalibrated simulation | E2 |
| Four-module scaling efficiency | >= 70% | 26% from one to four modules | fail | E1 uncalibrated simulation | E2 |
| Scheduler overhead | <= 5% | 1.5% to 29.4% for 26 plan evaluations per batch at an assumed 50-1000 ns each; needs <= 170 ns per evaluation (best evidence is E0; E3 required) | not evaluated | E0 estimate or argument | E3 |
| Numerical correctness | relative error <= 0.002 | no usable evidence: receipt f9e66f8ae7fd3c69 rejected: about model 'tiny-random', not 'qwen3-30b-a3b' | not evaluated | none | E3 |
| Model quality | perplexity <= 1.01x the reference | no usable evidence: not measured (the runtime never selects experts and exchanges only lossless-equivalent weight formats; that is an argument, not a measurement) | not evaluated | none | E3 |

Regime: Llama-3.3-70B, single-user decoding, 12 GiB GPU (most favourable regime found among published model shapes)

| criterion | threshold | result | status | evidence | required |
|---|---|---|---|---|---|
| Simulator calibration error | <= 10% median, <= 20% p95 | 0 validation benchmarks (at least 8 needed) | not evaluated | none | E3 |
| Throughput improvement | >= 2x | 2.72x the conventional baseline (E Tensor-parallel, pessimistic estimates) (best evidence is E1; E2 required) | not evaluated | E1 uncalibrated simulation | E2 |
| Energy efficiency | >= 2x | 1.26x the conventional baseline (E Tensor-parallel, pessimistic estimates) | fail | E1 uncalibrated simulation | E2 |
| Cost efficiency | >= 2x | 0.17x the conventional baseline (E Tensor-parallel, pessimistic estimates) | fail | E1 uncalibrated simulation | E2 |
| Four-module scaling efficiency | >= 70% | 93% from one to four modules (best evidence is E1; E2 required) | not evaluated | E1 uncalibrated simulation | E2 |
| Scheduler overhead | <= 5% | 1.5% to 29.4% for 26 plan evaluations per batch at an assumed 50-1000 ns each; needs <= 170 ns per evaluation (best evidence is E0; E3 required) | not evaluated | E0 estimate or argument | E3 |
| Numerical correctness | relative error <= 0.002 | no usable evidence: receipt 68c0bbac9dbdf2f8 rejected: about model 'tiny-random', not 'llama-3.3-70b' | not evaluated | none | E3 |
| Model quality | perplexity <= 1.01x the reference | no usable evidence: not measured (the runtime never selects experts and exchanges only lossless-equivalent weight formats; that is an argument, not a measurement) | not evaluated | none | E3 |

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

Validation coverage of the pre-registered plan (regimes the held-out benchmarks exercise; the
simulator is trusted only there): `colab-t4` lacks cpu-only, experts-on-cpu, moe; `i9-11900kf-3080ti` covers every regime; `macbook-m2` lacks experts-on-cpu, moe.

Each machine is judged on its own validation benchmarks; repeated runs of one benchmark count once,
and measurements made with an executable or model file that has since changed are excluded. A profile
counts as validated only while its content is the one the calibration run recorded.

Calibration and validation measurements are separate sets, declared when they are recorded; a
benchmark used to tune a profile cannot also validate it. Calibrating against an existing CPU and
GPU does not narrow any estimate for hardware that does not exist.

## 3. Eight architectures in one simulator

Qwen3-30B-A3B (30.5 B parameters, 16.0 GiB at 4.5 bits per weight),
single-user decoding, context 2048; a 12 GiB GPU and, where used, four modules with two DDR5
channels and 1024 MAC lanes each. Ratios are given at the nominal / pessimistic corner.

| architecture | tokens/s | tokens/J | est. price (USD) | tokens/s per k$ | throughput vs A | energy vs A | per-dollar vs A |
|---|---|---|---|---|---|---|---|
| A Centralized | 230 | 2.21 | 1,692 (1,228-2,420) | 135.7 (94.9-187.0) | 1.00x / 1.00x | 1.00x / 1.00x | 1.00x / 1.00x |
| B Distributed | 47 (38-59) | 0.42 (0.21-0.67) | 8,072 (5,151-19,648) | 5.8 (1.9-11.5) | 0.20x / 0.16x | 0.19x / 0.09x | 0.04x / 0.01x |
| C Pipeline | 28 (23-35) | 0.31 (0.18-0.44) | 8,072 (5,151-19,648) | 3.5 (1.2-6.8) | 0.12x / 0.10x | 0.14x / 0.08x | 0.03x / 0.01x |
| D Expert-local | 109 (92-129) | 0.72 (0.41-1.01) | 8,772 (6,151-20,148) | 12.4 (4.6-20.9) | 0.48x / 0.40x | 0.33x / 0.19x | 0.09x / 0.02x |
| E Tensor-parallel | 132 (111-153) | 0.84 (0.48-1.16) | 8,772 (6,151-20,148) | 15.0 (5.5-25.0) | 0.57x / 0.48x | 0.38x / 0.22x | 0.11x / 0.03x |
| F Adaptive hybrid | 232 (226-235) | 1.54 (1.16-1.75) | 8,772 (6,151-20,148) | 26.4 (11.2-38.3) | 1.01x / 0.98x | 0.70x / 0.52x | 0.19x / 0.06x |
| G Spatial dataflow | 113 (96-133) | 0.74 (0.42-1.04) | 8,772 (6,151-20,148) | 12.8 (4.7-21.7) | 0.49x / 0.42x | 0.33x / 0.19x | 0.09x / 0.03x |
| H Hierarchical | 235 (231-257) | 1.56 (1.18-1.86) | 8,072 (5,501-19,848) | 29.1 (11.6-46.6) | 1.02x / 1.01x | 0.71x / 0.53x | 0.21x / 0.06x |

* Fastest at nominal estimates: **H Hierarchical**, 1.02x architecture A.
* Most throughput per dollar: **A Centralized** (136 tokens/s per 1000 USD;
  the best module-based design reaches 29).
* B and C run the trunk blocks on memory-bound modules instead of a GPU and reach
  0.20x and 0.12x of A.

Prices are estimates for ratio purposes only, and the module-based systems contain more memory and
compute than the conventional one: this compares architectures on their natural hardware.

## 4. Phase 2 investigations

Each experiment changes one thing. The question is whether a weakness is an implementation problem
or a property of the architecture.

### 4.1 Sequential execution: the critical path

Expert-local architecture (D), four modules, per token (10.12 ms):

| on the critical path | ms per token | share |
|---|---|---|
| computing (kernels) | 4.32 | 43% |
| waiting for a busy device | 5.35 | 53% |
| returning results | 0.351 | 3% |
| waiting for inputs | 0 | 0% |
| host round trips | 0.098 | 1% |

By operation: experts 8.08 ms, trunk blocks 1.45 ms, output projection
0.35 ms, reductions 0.14 ms. Only the expert phase
(8.08 ms on the path; 19.18 ms of kernel work in total) can be divided among
modules; the remaining 2.04 ms is a serial dependency chain. If the expert work divided
perfectly, four modules would reach 78% scaling efficiency and eight
60%: **the 70% target is reachable at four modules only with near-perfect balance, and not at eight** while the trunk stays serial.

### 4.2 Expert imbalance

| placement | modules | tokens/s | scaling efficiency |
|---|---|---|---|
| one module (baseline) | 1 | 45.9 | 100% |
| round-robin placement | 4 | 92.4 | 50% |
| frequency-balanced placement | 4 | 95.3 | 52% |
| co-activation-aware placement | 4 | 98.8 | 54% |
| co-activation-aware, every expert on 2 modules | 4 | 109 | 60% |
| tensor-parallel (every expert on all modules) | 4 | 132 | 72% |

With whole experts on modules, the experts selected for a token land unevenly (a balls-in-bins
effect), and the layer waits for the fullest module. Relative to round-robin placement:
frequency-balanced +3.2%, co-activation-aware +6.9%, two copies of every expert
+18.2% (the scheduler picks the emptier holder), tensor-parallel +42.4% (no
imbalance by construction). Co-activation-aware placement needs far more profiling data than a
short trace provides and is regularised towards load balancing until then.

### 4.3 Scaling with module count

| architecture | modules | tokens/s | speed-up | scaling efficiency |
|---|---|---|---|---|
| D expert-local | 1 | 45.9 | 1.00x | 100% |
| E tensor-parallel | 1 | 45.9 | 1.00x | 100% |
| D expert-local | 2 | 70.2 | 1.53x | 77% |
| D with 2 replicas | 2 | 81.3 | 1.77x | 89% |
| E tensor-parallel | 2 | 81.3 | 1.77x | 88% |
| D expert-local | 4 | 98.8 | 2.15x | 54% |
| D with 2 replicas | 4 | 109 | 2.38x | 60% |
| E tensor-parallel | 4 | 132 | 2.87x | 72% |
| D expert-local | 8 | 128 | 2.79x | 35% |
| D with 2 replicas | 8 | 136 | 2.96x | 37% |
| E tensor-parallel | 8 | 189 | 4.11x | 51% |

Four-module efficiency: expert-local 54%, tensor-parallel 72%.
**The low efficiency measured in v0.1 (47%) was a property of expert-local placement, not of distributed execution as such.**

### 4.4 Synchronisation

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| host in the loop (latencies x1) | 98.8 | 0.661 | baseline |
| dependency events only (latencies x1) | 99.8 | 0.667 | +1.0% |
| host in the loop (latencies x8) | 58.8 | 0.417 | baseline |
| dependency events only (latencies x8) | 61.7 | 0.436 | +5.0% |
| 4 sequences in lock step (one barrier per layer) | 216 | 1.37 | baseline |
| 4 independent streams (no barrier) | 192 | 1.13 | -11.3% |

Removing the host round trip gains +1.0% at the profile's latencies and
+5.0% when they are eight times larger. A lock-step batch waits for its slowest
module at every layer; independent streams avoid that barrier (-11.3%).

### 4.5 GPU launch overhead

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| one launch per expert | 192 | 1.73 | baseline |
| experts batched into one launch | 230 | 2.21 | +19.7% |
| persistent kernels | 234 | 2.24 | +22.2% |

v0.1 charged one launch per expert, which understated the conventional baseline by
20%. All results in this report use batched launches.

### 4.6 Memory channels

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| D: channels pooled per module | 98.8 | 0.661 | baseline |
| D: independent channel engines | 78.3 | 0.54 | -20.7% |
| E: channels pooled per module | 132 | 0.841 | baseline |
| E: independent channel engines | 130 | 0.835 | -0.9% |

Running each memory channel as an independent engine does not help (-20.7% for D, -0.9% for E): a kernel already uses the module's full bandwidth, and splitting it halves each engine's rate while worsening imbalance. Memory contention is not the bottleneck; memory bandwidth is.

### 4.7 Communication

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| activations sent after routing (latencies x1) | 98.8 | 0.661 | baseline |
| activations streamed eagerly (latencies x1) | 99.7 | 0.666 | +0.9% |
| activations sent after routing (latencies x8) | 58.8 | 0.417 | baseline |
| activations streamed eagerly (latencies x8) | 61.6 | 0.435 | +4.7% |

Streaming the hidden state to the modules before the routing decision overlaps one transfer latency
per layer: worth +0.9% here and +4.7% at eight times the latency.

### 4.8 Output projection

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| row shards on the modules | 94.2 | 0.628 | baseline |
| on the trunk device | 98.8 | 0.661 | +4.9% |
| both; location chosen per token | 98.8 | 0.661 | +4.9% |

The projection is placed on both the modules and the trunk device when it fits, and the scheduler
picks per token (+4.9% against module shards only). v0.1 pinned it to the modules.

### 4.9 Energy

| variant | tokens/s | tokens/J | throughput change |
|---|---|---|---|
| F: no power gating | 232 | 1.49 | baseline |
| F: idle devices gated | 232 | 1.54 | +0.0% |
| F: energy-aware scheduling (0.001 s per joule) | 232 | 1.54 | +0.0% |
| F: energy-aware scheduling (0.01 s per joule) | 232 | 1.54 | +0.0% |
| A: time-only scheduling | 230 | 2.21 | baseline |
| A: energy-aware scheduling (0.01 s per joule) | 230 | 2.21 | +0.0% |

Power gating changes tokens per joule by +3.4% for single-user decoding. Energy-aware scheduling (0.01 s per joule) changes the hybrid's tokens per joule by +3.4% and the conventional system's by +0.0%, at +0.0% throughput for the latter.

### 4.10 Scheduler

| scheduler | tokens/s | plan evaluations per batch | improvements found | overhead at 50 / 200 / 1000 ns per evaluation |
|---|---|---|---|---|
| greedy earliest-finish | 232 | 25.9 | 0 | 1.5% / 5.9% / 29.4% |
| one refinement pass | 232 | 196 | 0 | 11.1% / 44.5% / 222.7% |
| two refinement passes | 232 | 196 | 0 | 11.1% / 44.5% / 222.7% |
| four refinement passes | 232 | 196 | 0 | 11.1% / 44.5% / 222.7% |

Bounded graph-level refinement changed throughput by +0.0% relative to greedy earliest-finish scheduling on this workload while evaluating 7.6x as many plans. It does fix constructed collisions (see the test suite), but mixture layers present few of them; greedy scheduling is the sensible default. The overhead of either depends on a per-evaluation cost that only a production implementation can establish.

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
| Qwen3-30B-A3B | 30.5 B | 16.0 | published config | 12 | 179 | 109 | 132 | 213 | 222 | 1.24x (H) | 0.27x |
| Llama-3.3-70B | 70.6 B | 37.0 | published config | 12 | 1.88 | 1.65 | 6.04 | 1.87 | 1.87 | 3.22x (E) | 0.64x |
| DeepSeek-V3 | 671.0 B | 351.5 | published config, approximated | 12 | 4.69 | 10.9 | 14.2 | 11.4 | 11.6 | 3.02x (E) | 0.76x |
| MoE-1T (hypothetical) | 1,010.3 B | 529.2 | hypothetical | 12 | 4.07 | 16.7 | 12.5 | 17.5 | 17.6 | 4.33x (H) | 0.90x |
| MoE-3T (hypothetical) | 3,122.3 B | 1,635.7 | hypothetical | 24 | 2.03 | 10.1 | 10.4 | 10.5 | 10.6 | 5.23x (H) | 1.93x |

No module-based architecture is faster than A for Qwen3-4B, Qwen3-8B, Qwen3-14B (each fits on the GPU). Where one is faster, the best is 1.24x to 5.23x A; per estimated dollar the best ratio anywhere is 1.93x.

## 6. Workload types

Qwen3-30B-A3B; each workload is a separate question with its own metric.

| workload | metric | A | E | F | H | best vs A |
|---|---|---|---|---|---|---|
| single-user decoding | tokens/s | 229.6 | 131.6 | 231.6 | 235.1 | 1.02x (H) |
| batched inference (4 sequences in lock step) | aggregate tokens/s | 484.1 | 239.7 | 530.0 | 532.4 | 1.10x (H) |
| concurrent requests (4 independent streams) | aggregate tokens/s | 281.6 | 197.1 | 339.0 | 343.0 | 1.22x (H) |
| long-context attention (32k context) | tokens/s | 94.6 | 73.3 | 95.6 | 96.5 | 1.02x (H) |
| prefill (128-token prompt) | prompt tokens/s | 2,652.5 | 518.2 | 3,258.3 | 3,451.5 | 1.30x (H) |
| sustained decoding (48 tokens) | tokens/s | 221.3 | 131.5 | 228.6 | 235.2 | 1.06x (H) |

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
| A, 12 GiB GPU | 232 (232) | 2.23 | 1,692 (1,228) | 137.4 |
| A, 24 GiB GPU | 258 (263) | 2.47 | 2,792 (1,528) | 92.5 |
| A, 32 GiB GPU | 438 (438) | 3.49 | 3,592 (2,728) | 121.9 |

By architecture:

| architecture | designs evaluated | best tokens/s | fastest design | best tokens/s per k$ | designs on the front |
|---|---|---|---|---|---|
| A Centralized | 3 | 438 | A, 32 GiB GPU | 137.4 | 3 |
| B Distributed | 201 | 99 | B, 8 x 16 GiB modules, 4 ch, 1024 lanes, 16 GB/s, 2 replicas | 21.4 | 0 |
| C Pipeline | 40 | 56 | C, 2 x 16 GiB modules, 4 ch, 1024 lanes, 16 GB/s | 16.3 | 0 |
| D Expert-local | 225 | 199 | D, 12 GiB GPU + 8 x 64 GiB modules, 4 ch, 1024 lanes, 16 GB/s, 2 replicas | 21.9 | 0 |
| E Tensor-parallel | 294 | 250 | E, 12 GiB GPU + 8 x 32 GiB modules, 4 ch, 1024 lanes, 8 GB/s | 22.9 | 0 |
| F Adaptive hybrid | 193 | 438 | F, 32 GiB GPU + 2 x 16 GiB modules, 4 ch, 256 lanes, 8 GB/s | 62.1 | 0 |
| G Spatial dataflow | 393 | 215 | G, 24 GiB GPU + 8 x 32 GiB modules, 4 ch, 1024 lanes, 32 GB/s, 2 replicas | 22.7 | 0 |
| H Hierarchical | 631 | 438 | H, 32 GiB GPU + 2 x 16 GiB modules, 4 ch, 256 lanes, 8 GB/s | 71.2 | 0 |

Equal cost -- the fastest design within a price budget, against the fastest conventional one:

| budget | fastest design | tokens/s | fastest conventional (A) tokens/s |
|---|---|---|---|
| 2,000 USD | A, 12 GiB GPU | 232 | 232 |
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

## 8. Numerical verification of the simulated dataflows

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
tip a near-tied routing decision; none was observed. This co-simulation does not run real weights;
section 9 does.

## 9. The physical runtime and its verification

v0.4 adds a runtime that executes real models, separate from the simulator: it reads GGUF files
(dequantising F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K and Q6_K, bit-identical to the
reference `gguf` package), runs Qwen3 dense and mixture-of-experts models through an execution plan
on the available backends, and keeps a KV cache per session. Backends available: numpy-cpu, torch-cpu.
Declared but not implemented, and refusing every call: cuda: not implemented in v0.4 (no CUDA kernels exist; nothing runs on a GPU); metal: not implemented in v0.4 (no Metal kernels exist; nothing runs on Apple GPUs); ggml: not implemented in v0.4 (no native ggml binding is built); remote: not implemented in v0.4 (no transport exists). No GPU inference exists in this
release, and nothing below says anything about module hardware.

Verification runs the same GGUF weights in transformers (float32; weights dequantised by the reference
package) and compares logits for every prompt position, every layer's output, every expert selection,
16 greedy tokens through the KV cache, and perplexity on a fixed text (target: relative L2 <= 0.002,
perplexity ratio <= 1.01):

| model | backend | logits rel. L2 (max) | layer outputs rel. L2 (max) | expert routing | greedy tokens | perplexity ratio | verdict |
|---|---|---|---|---|---|---|---|
| qwen3-0.6b (`Qwen3-0.6B-Q8_0.gguf`) | numpy-cpu | 2.8e-06 | 2.4e-06 over 27 layers | dense | 16 identical | 1.000001 | PASS |
| qwen3-0.6b (`Qwen3-0.6B-Q8_0.gguf`) | torch-cpu | 1.8e-06 | 1.6e-06 over 27 layers | dense | 16 identical | 1.000000 | PASS |

Prompt `The capital of France is` -> ` Paris. The capital of France is also the capital of the Republic of France.` (greedy, 9 tokens/s on the verification machine's CPU in NumPy). For information, the quantised file against the original checkpoint (not a property of the runtime): perplexity 15.30 vs 16.04 on the same 61 tokens, greedy choice identical for 98% of positions; a text this short does not measure quality.

These are measurements (E3) of the NumPy runtime on the model named in each row. They do not
qualify the gate's numerical-correctness or model-quality criteria for Qwen3-30B-A3B: the receipts are
bound to the model they measured, and the gate requires the model its regime is about.
CI repeats the comparison on small random-weight dense and MoE models, and checks that a deliberately
broken runtime fails it.

## 10. Memory domains and expert caching

Memory is managed per physical domain (`basp.memory`): devices that share memory (the M2's CPU
and GPU) share one pool, every allocation is charged to one of weights, KV cache, activations or
workspace, and weights + KV + activations + workspace + reserved never exceed the physical capacity
(checked after every operation in the tests). Requests are admitted only if their whole KV cache fits
(unreferenced cached experts are evicted first, and under memory pressure a margin is kept free);
blocks in use or being migrated are reference-counted and cannot be evicted.

Expert caching on the GPU of `desktop_gpu12`, for Qwen3-30B-A3B (15.2 GiB
of experts at 4.5 bits per weight, 2.53 MiB each): a miss either runs on the CPU
(49 us) or migrates the expert over PCIe (116 us)
and runs on the GPU (5.3 us). Cost-aware caching values an expert at
`V_e = (N_e (T_cpu - T_gpu) - T_migration) / M_e`; predictive caching estimates `N_e` from the measured
next-step repeat rate. Expert time per token over 159 recorded decoding steps (E1; no overlap of
migration with computation for any policy):

| GPU expert cache | policy | hit rate | run on CPU | MB migrated per token | expert ms per token |
|---|---|---|---|---|---|
| 2 GiB | LRU | 62% | 0% | 387 | 18.9 |
| 2 GiB | LFU | 66% | 0% | 344 | 17.0 |
| 2 GiB | cost-aware | 56% | 40% | 33 | 10.3 |
| 2 GiB | predictive | 57% | 39% | 38 | 10.3 |
| 2 GiB | none (CPU only) | 0% | 100% | 0 | 18.9 |
| 4 GiB | LRU | 81% | 0% | 189 | 10.3 |
| 4 GiB | LFU | 84% | 0% | 166 | 9.3 |
| 4 GiB | cost-aware | 65% | 32% | 29 | 8.6 |
| 4 GiB | predictive | 67% | 30% | 31 | 8.5 |
| 4 GiB | none (CPU only) | 0% | 100% | 0 | 18.9 |
| 8 GiB | LRU | 94% | 0% | 59 | 4.6 |
| 8 GiB | LFU | 94% | 0% | 59 | 4.6 |
| 8 GiB | cost-aware | 66% | 32% | 29 | 8.6 |
| 8 GiB | predictive | 67% | 30% | 31 | 8.4 |
| 8 GiB | none (CPU only) | 0% | 100% | 0 | 18.9 |

An unbounded cache (every expert migrated once) needs 4.6 ms per token over this
trace. Fastest policy at each capacity: 2 GiB: cost-aware (1.84x faster than LRU), 4 GiB: predictive (1.21x faster than LRU), 8 GiB: LFU (1.00x faster than LRU). LRU migrates far more data per token than the
cost-aware policy (387 vs 33 MB at 2 GiB, 189 vs 29 MB at 4 GiB, 59 vs 29 MB at 8 GiB), which instead runs rarely used experts on the CPU; whether that wins
depends on how much of the working set the cache holds. The cost-aware horizon (32 steps of expected
reuse) is a parameter, not tuned on this trace. These are simulated costs (recorded routing), not
runtime measurements.

## 11. Speculative execution and grouped expert reads (measured)

A drafter proposes several tokens; the target model verifies them in one pass, so on a mixture-of-experts model every expert any of them selects is read once for all of them. Greedy acceptance keeps exactly the target's tokens; sampling acceptance keeps its distribution (tested). Everything below was measured on the verification machine's CPU with this runtime (float32 kernels, weights dequantised by the runtime): wall-clock numbers describe this runtime, not an optimised engine; acceptance rates and expert reads describe the models and the method. Evidence E3, about these models only.

**Dense: Qwen3-0.6B-Q8_0.gguf drafting for Qwen3-1.7B-Q8_0.gguf** (64 greedy tokens per prompt, backend numpy-cpu):

| prompt | method | tokens/s | speed-up over plain decoding | acceptance | tokens identical |
|---|---|---|---|---|---|
| code | sequential | 5.81 | 1.00x | - | - |
| code | draft model, adaptive k | 6.83 | 1.18x | 82% | yes |
| code | draft model, k=2 | 5.15 | 0.89x | 95% | yes |
| code | draft model, k=4 | 5.96 | 1.03x | 92% | yes |
| code | prompt lookup, k=4 | 4.38 | 0.75x | 16% | yes |
| explain | sequential | 6.05 | 1.00x | - | - |
| explain | draft model, adaptive k | 5.22 | 0.86x | 69% | yes |
| explain | draft model, k=2 | 4.27 | 0.71x | 73% | yes |
| explain | draft model, k=4 | 4.61 | 0.76x | 57% | yes |
| explain | prompt lookup, k=4 | 4.16 | 0.69x | 4% | yes |
| prose | sequential | 5.25 | 1.00x | - | - |
| prose | draft model, adaptive k | 3.77 | 0.72x | 47% | yes |
| prose | draft model, k=2 | 3.59 | 0.68x | 48% | yes |
| prose | draft model, k=4 | 3.38 | 0.64x | 34% | yes |
| prose | prompt lookup, k=4 | 4.60 | 0.88x | 7% | yes |

Best 1.18x (code, draft model, adaptive k); worst 0.64x (prose, draft model, k=4). Every speculative run produced exactly the target's own greedy tokens.

**Dense: Qwen3-0.6B-Q8_0.gguf drafting for Qwen3-1.7B-Q8_0.gguf** (64 greedy tokens per prompt, backend torch-cpu):

| prompt | method | tokens/s | speed-up over plain decoding | acceptance | tokens identical |
|---|---|---|---|---|---|
| code | sequential | 4.61 | 1.00x | - | - |
| code | draft model, adaptive k | 5.90 | 1.28x | 83% | yes |
| code | draft model, k=2 | 6.77 | 1.47x | 95% | yes |
| code | draft model, k=4 | 6.22 | 1.35x | 92% | yes |
| code | prompt lookup, k=4 | 3.89 | 0.84x | 16% | yes |
| explain | sequential | 4.22 | 1.00x | - | - |
| explain | draft model, adaptive k | 5.10 | 1.21x | 73% | yes |
| explain | draft model, k=2 | 4.97 | 1.18x | 73% | yes |
| explain | draft model, k=4 | 3.62 | 0.86x | 57% | yes |
| explain | prompt lookup, k=4 | 3.21 | 0.76x | 4% | yes |
| prose | sequential | 4.49 | 1.00x | - | - |
| prose | draft model, adaptive k | 4.80 | 1.07x | 12% | yes |
| prose | draft model, k=2 | 4.05 | 0.90x | 48% | yes |
| prose | draft model, k=4 | 3.29 | 0.73x | 34% | yes |
| prose | prompt lookup, k=4 | 4.18 | 0.93x | 7% | yes |

Best 1.47x (code, draft model, k=2); worst 0.73x (prose, draft model, k=4). Every speculative run produced exactly the target's own greedy tokens.

**Mixture of experts: Qwen3-30B-A3B-Q4_K_M.gguf** (real weights, real routing; perplexity 1.68 on the 61-token check text -- a broken expert path gives nonsense; the text is probably memorised by a model this size).

Ordinary decoding read 384 experts per token (8.0 per layer: top-8 of 128, 1097 MB of stored weights). If `g` consecutive tokens of the same greedy continuation are processed together, each layer needs the union of their experts:

| tokens per pass (g) | distinct experts per layer per token | fewer expert reads |
|---|---|---|
| 1 | 8.00 | 1.00x |
| 2 | 6.12 | 1.31x |
| 3 | 5.25 | 1.52x |
| 4 | 4.72 | 1.70x |
| 5 | 4.37 | 1.83x |
| 6 | 3.99 | 2.01x |
| 8 | 3.57 | 2.24x |

Speculative decoding on the same prompt (the 0.6B model drafts; rejected proposals are read too, so the saving is below the union curve's):

| drafter | acceptance | tokens per target pass | expert reads per token | fewer expert reads | seconds | tokens identical |
|---|---|---|---|---|---|---|
| draft model, k=4 | 58% | 3.00 | 300 | 1.28x | 199 | yes |
| prompt lookup, k=4 | 3% | 1.04 | 596 | 0.64x | 387 | yes |

Wall clock in this runtime: 6.0 s per token for plain decoding, 4.1 s with draft model, k=4, 8.1 s with prompt lookup, k=4 (dominated by dequantising experts; plain decoding ran first, with a colder disk cache).

Expert prediction on the recorded routing (share of the selected experts inside the predicted set, predicting before each token; 128 experts, 8 selected):

| predictor | 8 predicted | 16 predicted | 32 predicted |
|---|---|---|---|
| frequency | 46% | 62% | 81% |
| recency + frequency | 49% | 65% | 82% |
| recency + frequency + transitions | 50% | 68% | 85% |

The routing of 159 decoded tokens is saved in `evidence/traces/` in the simulator's format; `--trace` replaces synthetic routing with it.

**Shared-prefix KV cache** (Qwen3-0.6B-Q8_0.gguf; four questions after the same system prompt; the first fills the cache):

| prompt tokens | reused from cache | full prefill (s) | with cache (s) | speed-up | same next token |
|---|---|---|---|---|---|
| 270 | 0 | 4.23 | 4.18 | 1.0x | yes |
| 268 | 256 | 3.41 | 0.88 | 3.9x | yes |
| 267 | 256 | 3.33 | 0.29 | 11.5x | yes |
| 268 | 256 | 3.18 | 0.28 | 11.2x | yes |

## 12. What changed since earlier reports

New in v0.5: speculative execution in the physical runtime (draft-model and prompt-lookup drafters,
exact greedy and distribution-preserving sampling acceptance, KV rollback, an adaptive draft-length
controller), grouped expert reads with byte accounting, a memory-lean mode that runs Qwen3-30B-A3B,
a shared-prefix KV cache, an optional PyTorch CPU backend under the same parity checks, and a
recorded routing trace of Qwen3-30B-A3B (section 11). When the report is generated with that trace,
the simulated experiments use real routing instead of synthetic routing.

New in v0.4: the gate decides each criterion from **metric-specific receipts** bound to the
identities that produced them (simulator source, configuration, profile, validation, model weights,
runtime, benchmark), so a throughput result cannot stand in for energy, a stale result is rejected,
and a result about another model cannot qualify the regime. Calibration merges repeated runs of a
benchmark into one data point, judges each machine on its own, excludes measurements made with an
executable or model file that has since changed, and qualifies a profile only if it is still the
profile that was validated. A physical runtime (section 9) executes real GGUF models; memory is
managed in physical domains with shared accounting, admission control and safe eviction (section 10).
Repaired: designs built on a unified-memory machine (Apple M2) split its one memory domain in two
and were rejected; they now share one pool, and the integrated GPU is not priced separately.

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

The reports in `reports/v0.1/` to `reports/v0.4/` were produced by earlier models and are kept as
records; their numbers are superseded by this report.
