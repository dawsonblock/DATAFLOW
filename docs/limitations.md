# Limitations of BASP Dataflow v0.1

This release is a Python research prototype: a simulator, a scheduler and reference kernels. It
contains **no hardware measurements**. Read every performance number it produces as "what the model
concludes under the stated assumptions".

## Assumptions that drive the results

* **Hardware profiles are illustrative.** Bandwidths, compute rates, latencies, kernel start-up
  times and all energy coefficients in `configs/*.json` are plausible guesses, not measurements.
  Conclusions that depend on them (for example whether a module beats a transfer to the GPU) can
  flip when they change; the report includes a GPU-memory sweep, but latency and throughput
  sweeps should be added before any hardware decision.
* **Routing traces are synthetic.** `synth_trace` models Zipf popularity and per-slot stickiness;
  `cyclic_trace` is a constructed best case for prediction. Neither is measured router behaviour.
  Cache hit rates, prefetch benefit and expert placement quality all depend on the real
  distribution. Recorded traces can be loaded with `RoutingTrace.load`.
* **Compression ratios are assumptions** for the entropy-coded, sparse-block and predictive
  formats (`FormatSpec.assumed`). The entropy figure is the order-0 entropy of INT4-quantized
  *Gaussian* weights.

## Modelling simplifications

* Memory is a bandwidth model: no row-buffer behaviour, refresh, NUMA or cache hierarchy.
  Ramulator2 and Timeloop are not integrated; the local loop-nest mapping inside a device is
  represented only by the SRAM tile-reuse model.
* Each device runs one kernel at a time. A GPU's ability to batch several experts into one launch
  is not modelled, which penalises GPU execution by one start-up time per expert.
* Link sharing is fluid (dominant-resource fairness); the estimator approximates it with
  gap-aware serialisation, so estimates under contention are a few percent off the simulation.
* A transfer over several hops is cut-through and needs a DMA slot on every hop.
* Attention is cost-modelled (projection MACs, KV-cache traffic) but not computed numerically;
  the numeric co-simulation uses an attention-free MoE stack. RoPE, sampling and the embedding
  lookup have no cost.
* The scheduler takes zero simulated time unless `ExecutionPolicy.decision_latency_s` is set, and
  its Python run time says nothing about a production implementation, so the "< 5% scheduler
  overhead" target is not evaluated.
* Eviction examines a seeded random sample of replicas (sampled LFU/LRU), not all of them.
* Energy is linear in MACs and bytes plus static power; there is no DVFS, no power gating.
* Fault injection is limited to protocol-level checks (CRC, protection faults, version
  mismatches, timeouts); link errors and retransmission are not simulated.

## Not implemented

| Specification item | Status |
|---|---|
| `hardware/*.sv` (decoder, MAC array, accumulator, stream controller) | Not started (milestone v0.9) |
| `backends/cuda`, `backends/metal` | Not started (milestone v0.8); no physical performance has been measured |
| `integrations/` (llama.cpp backend) | Not started |
| C++ production runtime | Not started |
| Real-model expert computation (milestone v0.6 evidence) | Kernels and co-simulation are verified on a small random-weight model only |
| Distributed learning (P2) | Only versioned tensors with explicit synchronisation |
| INT2 / ternary / sparse / entropy *kernels* | Formats are modelled for traffic and decode cost; only INT4 has a numeric kernel |
| Replica-based row splitting of a single tensor | Distributed mode requires one weight tensor per shard |
| Hardware-aware routing | Deliberately out of scope: the runtime never changes which experts are selected |

## Known rough edges

* `Command` descriptors carry at most 8 tensor ids; longer operand lists live in the kernel
  descriptor passed alongside the command (a wire format for kernel descriptors is future work).
* Background replication and prefetch share one budget per link direction; there is no
  per-destination fairness.
* The module configurations of the report have more memory and compute than the CPU+GPU
  configurations. The comparison is between dataflows on their natural hardware, not at equal
  cost; an equal-cost study needs a price model.
