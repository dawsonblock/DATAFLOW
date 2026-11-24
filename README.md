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

<!-- results:begin -->
Generate the report with `python -m benchmarks.run_report`; it is written to `reports/v0.1/`.
<!-- results:end -->

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
