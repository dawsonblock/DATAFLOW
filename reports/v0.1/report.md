# BASP Dataflow v0.1 -- simulation report

> **Read this first.** Everything below is the output of a discrete-event *simulation* driven by
> the illustrative hardware profiles in `configs/` and a *synthetic* routing trace.  No number here
> is a measurement of real hardware or of a real model's router.  The report shows what the dataflow
> engine concludes *given those assumptions*, and how sensitive the conclusions are to them.
> Regenerate with `python -m benchmarks.run_report` (deterministic).

## 1. Analytical dataflow comparison (specification section 6)

One expert with hidden size 4096 and intermediate size 14336 at 4.5 bits per weight; 24 GB/s
interconnect, 54 GB/s local memory, 128 GMAC/s local compute.

| quantity | value |
|---|---|
| Expert parameters | 176.2M |
| Weight transfer size | 94.5 MiB |
| Input/output activations | 16.0 KiB |
| Interconnect traffic reduction | 6,048x |
| Cold weight transfer, excluding execution | 4.129 ms |
| Near-memory execution, including activations | 1.836 ms |
| Near-memory kernel limited by | memory bandwidth |

Arithmetic sizing (section 10): two DDR5-5600 channels deliver 89.6 GB/s
in theory; consuming that stream at 4.5 bits per weight needs 159 GMAC/s,
and 1024 lanes at 250 MHz provide 256 GMAC/s (1.61x headroom).
The simulator reproduces both paths of the table on an idle system to within floating-point error
(`tests/test_scheduler.py::test_section_6_numbers_are_reproduced_by_the_simulator`).

## 2. Assumptions

**Workload.** Qwen3-30B-A3B: 30.5 B parameters, 3.4 B
activated per token, 128 experts per layer of which 8 are selected, 48
layers; INT4 weights with one FP16 scale per 32 weights (4.5 bits per weight, 16.0 GiB).
Single-sequence decoding at context length 2048, 24 tokens of which
the first 8 are warm-up.  Routing is synthetic: Zipf(0.8) expert
popularity per layer and per-slot stickiness 0.6; placement and the predictor see
64 profiling steps, evaluation uses held-out steps of the same generator.

**Hardware profile `desktop_gpu12`** (configurations 1-4, 6, 7):

| device | kind | memory | sustained bandwidth | compute | kernel start-up | tile SRAM | idle power |
|---|---|---|---|---|---|---|---|
| host | CPU | 64 GiB | 54 GB/s | 30 GMAC/s | 0.5 us | 0 MiB | 45 W |
| gpu | GPU | 12 GiB | 500 GB/s | 20,000 GMAC/s | 3.0 us | 0 MiB | 30 W |

| link | bandwidth per direction | latency |
|---|---|---|
| pcie-gpu | 24 GB/s | 5 us |

**Hardware profile `basp_4module`** (configurations 5 and 8) adds four compute-memory modules:

| device | kind | memory | sustained bandwidth | compute | kernel start-up | tile SRAM | idle power |
|---|---|---|---|---|---|---|---|
| host | CPU | 64 GiB | 54 GB/s | 30 GMAC/s | 0.5 us | 0 MiB | 45 W |
| gpu | GPU | 12 GiB | 500 GB/s | 20,000 GMAC/s | 3.0 us | 0 MiB | 30 W |
| mod0 | NEAR_MEMORY | 32 GiB | 54 GB/s | 256 GMAC/s | 1.0 us | 16 MiB | 12 W |
| mod1 | NEAR_MEMORY | 32 GiB | 54 GB/s | 256 GMAC/s | 1.0 us | 16 MiB | 12 W |
| mod2 | NEAR_MEMORY | 32 GiB | 54 GB/s | 256 GMAC/s | 1.0 us | 16 MiB | 12 W |
| mod3 | NEAR_MEMORY | 32 GiB | 54 GB/s | 256 GMAC/s | 1.0 us | 16 MiB | 12 W |

| link | bandwidth per direction | latency |
|---|---|---|
| pcie-gpu | 24 GB/s | 5 us |
| pcie-mod0 | 16 GB/s | 2 us |
| pcie-mod1 | 16 GB/s | 2 us |
| pcie-mod2 | 16 GB/s | 2 us |
| pcie-mod3 | 16 GB/s | 2 us |

Format assumptions: lossless entropy coding of INT4 codes is taken as 4.12
bits per weight (the order-0 entropy of INT4-quantized Gaussian weights measured by the reference
kernels is 3.62 bits per code, plus 0.5 bit of scales); real
weights will differ.

## 3. Eight execution configurations

| configuration | purpose | tokens/s | ms/token | link MB/token | tokens/J | expert runs on GPU-resident weights | stall ms/token | vs. baseline |
|---|---|---|---|---|---|---|---|---|
| GPU-only | conventional reference | 127 | 7.89 | 80.3 | 1.38 | 92% | 5.49 | - |
| CPU-only | high-capacity memory reference | 7.76 | 129 | 0 | 0.0342 | 0% | 0 | - |
| Static CPU/GPU | standard layer offloading | 21.7 | 46.1 | 0.308 | 0.1 | 73% | 0 | - |
| Adaptive CPU/GPU | dynamic scheduling benefit | 145 | 6.88 | 60.5 | 1.2 | 93% | 2.33 | 6.70x static_split |
| Distributed memory compute | near-memory benefit | 85.7 | 11.7 | 5.22 | 0.6 | 0% | 0 | 3.96x static_split |
| Predictive dataflow | latency-hiding benefit | 145 | 6.89 | 60.9 | 1.2 | 93% | 2.34 | 1.00x adaptive |
| Compression-aware dataflow | reduced memory traffic | 131 | 7.61 | 73.5 | 1.42 | 92% | 5.04 | 1.04x gpu_only |
| Complete BASP | combined performance | 165 | 6.05 | 37.6 | 1.11 | 83% | 0 | 1.14x adaptive |

"Link MB/token" counts bytes carried by links, once per hop.  "Stall" is time operations spent
waiting for synchronous weight movement.  "Expert runs on GPU-resident weights" is the share of
expert executions that found their weights already on the GPU (the rest ran elsewhere or waited
for a transfer).  The module configurations have additional memory and
compute installed, so rows 5 and 8 compare dataflows on their natural hardware, not at equal cost.

Dataflow modes chosen by the complete configuration:
distributed 16, output-stationary 6144, row-stationary 1067, weight-stationary 6613.

## 4. No dataflow is universally best

Each scenario runs the same operations under single-mode policies on `basp_4module`.

| scenario | metric | simulated result per mode | best | as predicted |
|---|---|---|---|---|
| Expert used once, weights on a module | completion time (us) | weight-stationary: 64.7, activation-stationary: 181.2, streaming: 175.9 | weight-stationary | yes |
| Same expert used 64 times | total time (us) | weight-stationary: 4,138.5, activation-stationary: 704.6 | activation-stationary | yes |
| 8 experts on 2 modules, 16 rows, one reduction | interconnect traffic (KiB) | weight-stationary: 1,280.0, output-stationary: 512.0 | output-stationary | yes |
| Same expert run 16 times on its module | total time (us) | weight-stationary: 1,034.6, row-stationary: 573.8 | row-stationary | yes |
| Large expert on NVMe only, used once on a module | completion time (us) | activation-stationary: 16,003.8, streaming: 14,168.8 | streaming | yes |
| Output projection (151936 x 2048), 4 row shards | completion time (us) | weight-stationary: 3,275.5, distributed: 838.7 | distributed | yes |

## 5. Sensitivity: GPU memory and fixed overheads

Tokens per second as GPU memory varies (model size 16.0 GiB):

| GPU memory (GiB) | static CPU/GPU | adaptive CPU/GPU | distributed memory | complete BASP |
|---|---|---|---|---|
| 6 | 11.3 | 84 | 85.7 | 135 |
| 8 | 13.4 | 104 | 85.7 | 146 |
| 12 | 21.7 | 145 | 85.7 | 165 |
| 16 | 56.3 | 183 | 85.7 | 173 |
| 24 | 186 | 186 | 85.7 | 173 |

Tokens per second as every link latency, kernel start-up time and the host synchronisation
latency are scaled together (1 = the profile values):

| latency scale | adaptive CPU/GPU | distributed memory | complete BASP |
|---|---|---|---|
| 0.25x | 170 | 93.1 | 195 |
| 1x | 145 | 85.7 | 165 |
| 4x | 91.9 | 65 | 108 |
| 16x | 37.5 | 33.1 | 43 |

## 6. Scaling with module count

Distributed-memory configuration, trunk on the GPU:

| modules | tokens/s | speed-up | scaling efficiency |
|---|---|---|---|
| 1 | 45.1 | 1.00x | 100% |
| 2 | 60.9 | 1.35x | 68% |
| 4 | 85.7 | 1.90x | 47% |

## 7. Single-user latency versus multi-user throughput

| configuration | concurrent sequences | aggregate tokens/s | per-sequence ms/token |
|---|---|---|---|
| adaptive | 1 | 148 | 6.76 |
| complete | 1 | 164 | 6.02 |
| adaptive | 8 | 284 | 28.1 |
| complete | 8 | 409 | 19.6 |

## 8. Predictive prefetching on held-out routing

A mid-sized model whose GPU holds about 30% of the experts; all expert computation on the GPU, so
every miss is a synchronous weight transfer.

| routing | policy | tokens/s | stall ms/token | GPU-resident runs | prefetches | hits | wasted MB |
|---|---|---|---|---|---|---|---|
| cyclic | reactive cache | 380 | 5.92 | 1% | 0 | 0 | 0 |
| cyclic | with prefetch | 391 | 4.17 | 25% | 288 | 288 | 0 |
| sticky | reactive cache | 1,045 | 0.957 | 72% | 0 | 0 | 0 |
| sticky | with prefetch | 1,144 | 0.833 | 77% | 74 | 42 | 18.9 |

`cyclic` routing is a constructed best case in which the selection changes completely every token
in a learnable order; `sticky` is the default synthetic trace, whose structure a reactive cache
already captures.  Whether real routers have exploitable structure beyond recency must be measured
on recorded traces (`RoutingTrace.load`).

## 9. Correctness

| graph | max relative error vs float64 reference | routing identical | tokens identical | dependency violations | accounting violations |
|---|---|---|---|---|---|
| fused | 4.45e-04 | True | True | 0 | 0 |
| unfused | 4.45e-04 | True | True | 0 | 0 |

A small attention-free MoE language model with real INT4 weights is decoded autoregressively by the
streaming kernels *following the scheduled plans* (sharded output projection, fused partial sums,
replication) while the simulator provides the execution order.  Fused and unfused graphs produce
bit-identical logits: True.

## 10. Acceptance targets (specification section 17)

Targets are goals, not achievements.  "Simulated" means the figure comes from this model under the
assumptions of section 2 and says nothing yet about real hardware.

| criterion | target | result in simulation | status |
|---|---|---|---|
| Correctness | match independent numerical reference | max relative error 4.5e-04; fused = unfused bit-exact: True | met on the small numeric model |
| Interconnect traffic | 50% reduction on offloading workloads | 53% vs GPU paging, 38% vs adaptive CPU/GPU | simulated: below target |
| Scheduler overhead | below 5% of execution time | 17.5 plan evaluations per operation; the Python prototype's decision time is not representative | not measurable yet |
| Memory utilisation | above 60% of sustainable bandwidth | 98% of module bandwidth while a kernel runs | simulated: meets target |
| Inference throughput | 2x optimized comparable baseline | 1.14x adaptive CPU/GPU | simulated: below target |
| Energy efficiency | 2x tokens per joule | 0.92x adaptive CPU/GPU | simulated: below target |
| Scaling efficiency | 70% from one to four modules | 47% at 4 modules | simulated: below target |
| Model quality | no material degradation | routing never altered; only lossless-equivalent formats exchanged | by construction |
