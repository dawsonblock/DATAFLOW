# BASP Dataflow

The dataflow engine of the Block Adaptive Stream Processor (BASP): a bidirectional,
topology-aware, asynchronous dataflow layer that decides, per operation, whether to move data to
a processor or to execute where the data already resides.

This repository is **v0.1**: a Python research prototype consisting of a discrete-event
simulator, a cost-based joint scheduler, a distributed tensor memory model, the BASP Stream
Protocol, INT4 reference kernels and a reproducible experiment harness. It runs on a laptop and
needs only NumPy.

> **Status of the numbers.** Nothing here has been measured on hardware. Every performance figure
> is the output of a simulation under the illustrative hardware profiles in [`configs/`](configs)
> and a synthetic routing trace. See [docs/limitations.md](docs/limitations.md) before drawing
> conclusions.

## Quick start

```bash
pip install -e ".[dev]"                     # or just: pip install numpy pytest
python -m pytest -q                         # 370 tests, about 15 s
python -m benchmarks.dataflow_model         # the closed-form comparison of specification section 6
python -m benchmarks.run_report --quick     # every experiment, small sizes, under a minute
python -m benchmarks.run_report             # the full report -> reports/v0.1/
```

Scheduling one expert on a simulated system:

```python
from basp.core.graph import OpCode
from basp.dataflow.base import ALL_MODES, ExecutionPolicy
from basp.experiments.microbench import MicroSystem
from basp.experiments.profiles import load_topology

topology = load_topology("basp_4module")            # host CPU, GPU, four compute-memory modules
gpu, module = topology.by_name("gpu").device_id, topology.by_name("mod0").device_id
system = MicroSystem(topology, ExecutionPolicy("adaptive", ALL_MODES))

params = 3 * 2048 * 768                             # one Qwen3-30B-A3B expert
weights = system.weight("expert", params, module)   # INT4, 4.5 bits/weight, stored on the module
x = system.activation("x", 4096, gpu)               # FP16 hidden state on the GPU
y = system.activation("y", 4096)
op = system.op(OpCode.EXPERT, "expert", inputs=(x,), weights=(weights,), outputs=(y,),
               macs=params, output_device=gpu)

result = system.run([op])
plan = result.plan
print(plan.choice.mode.name, "on device", plan.devices)      # where and how it ran
print(plan.estimate)                                         # T_queue, T_input, T_kernel, ...
print(f"estimated {result.estimate_s * 1e6:.1f} us, simulated {result.actual_s * 1e6:.1f} us")
```

## What v0.1 implements

The first-implementation checklist of the architecture, with the evidence for each item:

| # | Checklist item | Implementation | Evidence |
|---|---|---|---|
| 1 | Dataflow intermediate representation | `basp/core/graph.py`, `compiler/lower.py`, `compiler/fusion.py` | `tests/test_core.py`, `tests/test_compiler.py` |
| 2 | Discrete-event simulation | `basp/simulator/` | closed-form completion times, conservation under random contention (`tests/test_simulator.py`) |
| 3 | CPU, GPU and near-memory device models | `core/topology.py`, `core/capabilities.py`, `simulator/compute.py`, `configs/*.json` | `tests/test_simulator.py`, `tests/test_scheduler.py` |
| 4 | Memory channels and interconnect contention | `simulator/fluid.py`, `simulator/interconnect.py` | fairness, priority, DMA-queue and bidirectional-link tests |
| 5 | Tensor ownership and residency tracking | `basp/memory/` | `tests/test_memory_*.py` (invariants, versioning, eviction rules) |
| 6 | Seven execution strategies | `basp/dataflow/` | each mode wins its own scenario (`experiments/modes.py`) |
| 7 | Deterministic cost-based scheduler | `basp/scheduler/` | estimate equals simulation on an idle system; joint refinement beats greedy |
| 8 | INT4 reference computation | `basp/kernels/` | differential tests against a float64 reference, exact integer path |
| 9 | Distributed MoE expert execution | `backends/virtual/runtime.py`, `scheduler/placement.py` | numeric co-simulation (`tests/test_end_to_end.py`) |
| 10 | Adaptive expert caching | `memory/replication.py`, `Scheduler.propose_fills` | `tests/test_memory_residency.py`, report section 3 |
| 11 | Numerical and dependency verification | `basp/verification/`, `backends/cpu/` | every policy and graph form matches the float64 reference within FP16 rounding, and each other bit-for-bit |
| 12 | Comparison against baselines | `experiments/configurations.py` | eight configurations, report section 3 |
| 13 | Reproducible research reports | `experiments/report.py`, `benchmarks/` | deterministic report (`tests/test_experiments.py`) |

How the pieces fit together is described in [docs/architecture.md](docs/architecture.md).

## Simulated results

Full report: [reports/v0.1/report.md](reports/v0.1/report.md) (machine-readable:
[results.json](reports/v0.1/results.json)). Workload: Qwen3-30B-A3B shapes at 4.5 bits per weight
(16.0 GiB), 16 measured tokens after 8 warm-up tokens at context 2048, synthetic
routing, and a 12 GiB GPU that holds about 70% of the experts.

| Configuration | tokens/s | ms/token | link MB/token | tokens/J |
|---|---|---|---|---|
| GPU-only (demand paging) | 126.7 | 7.89 | 80.3 | 1.38 |
| CPU-only | 7.8 | 128.82 | 0.0 | 0.03 |
| Static CPU/GPU layer split | 21.7 | 46.13 | 0.3 | 0.10 |
| Adaptive CPU/GPU | 145.3 | 6.88 | 60.5 | 1.20 |
| Distributed memory compute | 85.7 | 11.66 | 5.2 | 0.60 |
| Predictive dataflow | 145.2 | 6.89 | 60.9 | 1.20 |
| Compression-aware dataflow | 131.4 | 7.61 | 73.5 | 1.42 |
| Complete BASP | 165.0 | 6.05 | 37.6 | 1.11 |

What the simulation says under these assumptions:

* **Deciding per expert beats splitting by layer.** Caching hot experts on the GPU and scheduling
  each expert individually is 6.7x faster than offloading whole layers.
* **Near-memory execution removes weight traffic.** The distributed configuration moves
  5.2 MB per token against 80 MB for GPU demand paging.
* **The combined dataflow is 1.14x the adaptive CPU/GPU baseline at 12 GiB, not 2x**, and
  the gain depends on how much of the model the GPU holds: 1.60x at 6 GiB, 1.40x at 8 GiB,
  and 0.93x at 24 GiB, where the whole model fits on the GPU.
* **With 8 concurrent sequences** the combined dataflow reaches
  1.44x the baseline's aggregate throughput (more distinct experts per layer, which the modules
  execute in parallel).
* **Module scaling is 47% efficient from one to four modules**, below the 70% target. The serial
  trunk on the GPU and uneven per-token expert load are the likely limits; they have not been
  isolated yet.
* **Fixed overheads matter.** Scaling every link latency and kernel start-up time by 16x takes
  the combined configuration from 165 to 43 tokens/s.
* **Energy per token is worse, not better** (0.92x the baseline's tokens per joule): four idle
  modules cost more static power than the faster decode saves, under the assumed power figures.
* **Prefetching helps only when routing has structure beyond recency.** On constructed cyclic
  routing it cuts stall time by 30%; on the default trace it changes nothing measurable for
  Qwen3 because a reactive cache already holds what recency predicts.
* **Lossless entropy coding saves about 8% of weight traffic** under the Gaussian-weight
  estimate, and only where the destination decodes faster than the link.

Against the acceptance targets of the specification, on this profile the simulation meets the
memory-utilisation target, meets the 50% traffic-reduction target against GPU demand paging
(53%) but not against the adaptive baseline (38%), and misses the 2x throughput, 2x energy
and 70% scaling targets. Scheduler overhead cannot be judged from a Python prototype. These are
statements about the assumed hardware and workload; measurements must replace them.

## Repository layout

```
basp/
  core/         tensor descriptors and formats, dataflow IR, topology, device capabilities
  simulator/    event kernel, fluid bandwidth model, interconnect, kernel execution, energy
  dataflow/     the seven modes, plan vocabulary, adaptive candidate enumeration, prefetcher
  scheduler/    cost model, planning state, critical path, joint scheduler, placement, predictor
  memory/       tensor registry, residency, replication (expert cache), allocator, paged KV cache
  streams/      BASP Stream Protocol: commands, completions, activation and weight streams
  kernels/      float64 reference, INT4 format, fused streaming kernels, MoE expert
  compiler/     lowering, operator fusion, command emission
  backends/     virtual (simulated devices executing BSP commands), cpu (numeric execution)
  workloads/    model shapes (Qwen3-30B-A3B), routing traces, small numeric MoE model
  verification/ numerical, dependency and accounting checks
  experiments/  harness, standard configurations, mode study, analytic model, report
benchmarks/     command-line entry points
configs/        hardware profiles and workload descriptions (all values are assumptions)
reports/v0.1/   the generated report and its machine-readable results
tests/          regression suite
docs/           architecture and limitations
```

Compared with the layout sketched in the architecture document, the Python packages live under
`basp/` so they are importable as one distribution, and `hardware/`, `integrations/`,
`backends/cuda` and `backends/metal` do not exist yet.

## Milestones

| Release | Scope | State |
|---|---|---|
| v0.1 | Dataflow simulator | Implemented; event and bandwidth accounting verified |
| v0.2 | Tensor execution graph | Implemented; dependency resolution verified in co-simulation |
| v0.3 | Adaptive scheduler | Implemented; compared against static scheduling in simulation |
| v0.4 | Distributed tensor memory | Implemented; ownership and replication invariants tested |
| v0.5 | Low-bit tensor engine | INT4 implemented with numerical differential tests; other bit-widths modelled only |
| v0.6 | MoE execution | Verified on a small random-weight model; **no real model weights yet** |
| v0.7 | Predictive prefetching | Implemented; benefit shown only on synthetic held-out routing |
| v0.8 | CPU/GPU integration | Not started |
| v0.9 | Hardware RTL | Not started |
| v1.0 | FPGA prototype | Not started |

## License

MIT, see [LICENSE](LICENSE).
