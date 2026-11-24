# BASP Dataflow

The dataflow engine of the Block Adaptive Stream Processor (BASP): a bidirectional,
topology-aware, asynchronous dataflow layer that decides, per operation, whether to move data to
a processor or to execute where the data already resides — and, since v0.2, an engineering system
for finding out **which architecture is best**, including "a conventional CPU/GPU system".

This repository is **v0.2, the Calibrated Architecture Discovery Engine**: a discrete-event
simulator, a cost-based scheduler, a distributed tensor memory model, the BASP Stream Protocol,
INT4 reference kernels, eight competing dataflow architectures, a multi-objective design-space
explorer, calibration tooling and an evidence-gated hardware decision. It is a Python research
prototype that runs on a laptop and needs only NumPy.

> **Status of the numbers.** Nothing here has been measured on hardware. Every performance figure
> is the output of a simulator that has not been validated against measurements; the host and GPU
> come from an illustrative profile, everything about compute-memory modules is an estimate with a
> pessimistic–nominal–optimistic range, and routing is synthetic. "Calibrated" describes what the
> system can do once measurements exist, not its current state. See
> [docs/limitations.md](docs/limitations.md).

## Where things stand

**Hardware development: NO-GO.** No FPGA or RTL work has been started, by decision
([docs/roadmap.md](docs/roadmap.md)). The gate requires every criterion to pass on calibrated
evidence at the pessimistic end of every estimate; today the simulator is uncalibrated, and even
taking its numbers at face value no module-based design reaches the thresholds.

Full report: [reports/v0.2/report.md](reports/v0.2/report.md)
(machine-readable: [results.json](reports/v0.2/results.json),
[dse.json](reports/v0.2/dse.json)). First workload: Qwen3-30B-A3B shapes at 4.5 bits per weight
(16.0 GiB), single-user decoding, 12 GiB GPU.

| Architecture | tokens/s (range over estimates) | vs A, nominal / pessimistic | per dollar vs A |
|---|---|---|---|
| A Centralized (conventional CPU/GPU) | 164 | 1.00x | 1.00x |
| B Distributed (modules only) | 46 (38–59) | 0.28x / 0.23x | 0.06x / 0.03x |
| C Pipeline | 28 (23–35) | 0.17x / 0.14x | 0.04x / 0.02x |
| D Expert-local | 107 (90–126) | 0.65x / 0.55x | 0.13x / 0.06x |
| E Tensor-parallel | 132 (111–153) | 0.80x / 0.68x | 0.16x / 0.08x |
| F Adaptive hybrid | 200 (182–214) | 1.22x / 1.11x | 0.24x / 0.13x |
| G Spatial dataflow | 110 (93–130) | 0.67x / 0.57x | 0.13x / 0.06x |
| H Hierarchical | 218 (200–244) | 1.33x / 1.22x | 0.28x / 0.14x |

What the simulation says, under its assumptions:

* **The conventional system wins on cost.** The fastest module-based design is 1.33x architecture A
  on the first workload, at roughly five times the estimated price: about a quarter of the
  throughput per dollar. Prices are guesses for ratio purposes; the conclusion survives the whole
  estimate range.
* **Design-space exploration agrees.** Of 1,980 simulated systems (from 8,030 feasible), the
  Pareto front holds two conventional systems and seven hierarchical hybrids; no other
  architecture is on it. Within 2,000 and 5,000 USD budgets the best system is conventional. The
  fastest design found is 1.29x the fastest conventional one at 5.0x the price (1.09x at 9.6x
  with pessimistic estimates).
* **Modules add nothing while a model fits on the GPU**, and gain as it outgrows it: 1.3x for the
  30B MoE, 3.8x for a 70B dense model, 3.1x for DeepSeek-V3 shapes. Per estimated dollar no
  published model shape exceeds 0.8x the conventional system.
* **The 47% scaling efficiency of v0.1 was expert imbalance, not a limit of distribution.** From
  one to four modules: whole experts 50%, two copies of each expert 58%, experts partitioned
  across all modules 72%. The serial trunk caps the ideal at 78% for four modules and 60% for
  eight.
* **Most "obvious" fixes are worth little here.** Removing host round trips: +0.9%. Streaming
  activations before routing is known: +0.9%. Independent memory-channel engines: −22%.
  Graph-level schedule refinement: no improvement at 7.8x the search effort. Power gating: no
  effect on single-user decoding.
* **Two corrections moved the baseline up.** Batching GPU kernel launches (+11%) and a CPU
  arithmetic rate consistent with memory-bound decoding. v0.1 understated the conventional system
  on both counts.
* **Prefill is the workload where hybrids help most** (3.0x prompt tokens/s); other workload
  types range from 1.2x to 1.5x.
* **Numerics hold.** All eight architectures reproduce a float64 reference on a small
  random-weight model (relative error 4.5e-4, identical routing and tokens); fused and unfused
  graphs are bit-identical. Real model weights have not been run.

## What you need to run

The simulator's verdict means little until it is calibrated. The procedure for the two chosen
machines (Intel i9-11900KF + RTX 3080 Ti, MacBook Pro M2) is in
[docs/calibration.md](docs/calibration.md):

```bash
python -m benchmarks.profile_hardware --machine i9-11900kf-3080ti ...   # memory, GPU link, roofline
python -m benchmarks.run_llama_bench --split calibration ...            # benchmarks used for tuning
python -m benchmarks.run_llama_bench --split validation ...             # benchmarks used for judging
python -m benchmarks.calibrate measurements.json --machine i9-11900kf-3080ti \
    --profile-out configs/i9_11900kf_rtx3080ti.json                     # fit, then validate
python -m benchmarks.record_routing_trace ...                           # real expert routing
python -m benchmarks.explore --base-profile i9_11900kf_rtx3080ti        # rerun on the calibrated machine
python -m benchmarks.run_report --base-profile i9_11900kf_rtx3080ti
```

Calibration and validation measurements are separate sets, declared when recorded; only the
validation split decides whether the simulator is trusted (median error at most 10%, 95th
percentile at most 20%). The scripts that touch real hardware were written without access to a
GPU, llama.cpp or model files and **have never been executed**; expect small fixes on first use.

## Quick start

```bash
pip install -e ".[dev]"                          # or just: pip install numpy pytest
python -m pytest -q                              # regression suite, about two minutes
python -m benchmarks.dataflow_model              # closed-form comparison of specification section 6
python -m benchmarks.explore --budget 60 --workers 4 --out /tmp/dse.json   # a small exploration
python -m benchmarks.run_report --quick          # every experiment, short runs, a few minutes
python -m benchmarks.explore --budget 2000 --workers 4                     # full exploration (~15 min)
python -m benchmarks.run_report                  # the full report -> reports/v0.2/ (~10 min)
```

Both the exploration and the report are deterministic: the same code, profiles, estimates and
seeds reproduce the committed files exactly.

Simulating one architecture on one hardware design:

```python
from basp.dse.hardware import HardwareDesign
from basp.experiments.variants import run_architecture, traces
from basp.workloads.zoo import MODELS

spec = MODELS["qwen3-30b-a3b"]
prior, held = traces(spec, tokens=6)             # profiling trace, held-out evaluation trace
design = HardwareDesign(gpu_gib=12, modules=4, module_gib=32, lanes=1024, fabric="local")
out = run_architecture("H", design, spec, prior, held, replicas=2, corner="pessimistic")
print(f"{out.tokens_per_s:.0f} tokens/s, {out.result.tokens_per_joule:.2f} tokens/J, "
      f"about {out.cost_usd:,.0f} USD")
```

Scheduling a single operation on the engine underneath:

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
print(result.plan.choice.mode.name, "on device", result.plan.devices)
print(f"estimated {result.estimate_s * 1e6:.1f} us, simulated {result.actual_s * 1e6:.1f} us")
```

## What v0.2 adds

| Release item | Implementation | Evidence |
|---|---|---|
| CI and regression testing | `.github/workflows/ci.yml` | lint, tests on Python 3.10 and 3.13, exploration and report smoke runs |
| Hardware profiling and calibration | `benchmarks/profile_hardware.py`, `run_llama_bench.py`, `calibrate.py`, `basp/calibration/` | fitting recovers a synthetic machine; the fit provably ignores the validation split; **not run on real hardware** |
| Real MoE routing trace recorder | `benchmarks/record_routing_trace.py` | collection logic tested on stand-in modules; **not run on a model** |
| Corrected GPU batching and output placement | `simulator/compute.py`, `scheduler/placement.py` | closed-form launch-cost tests; report sections 4.5 and 4.8 |
| Critical-path and scaling analysis | `basp/analysis/` | components sum to the token time exactly; report sections 4.1–4.3 |
| Eight competing architectures | `experiments/architectures.py` | each runs the same workload and matches the numeric reference |
| Multi-objective optimisation | `basp/dse/`, `experiments/explore.py` | exhaustive search returns the true Pareto front; two full runs reproduced bit for bit |
| Equal-cost and equal-power comparisons | report section 7 | estimated price model with ranges |
| Numerical verification | `basp/verification/`, `backends/cpu/` | report section 8 |
| Reproducible reports | `benchmarks/run_report.py` | deterministic; commentary is derived from the results |
| Evidence-gated hardware decision | `experiments/gate.py`, `configs/hardware_gate.json` | report section 1; missing evidence counts against |

The v0.1 engine underneath (dataflow IR, discrete-event simulator, device and interconnect
models, tensor residency, the seven execution modes, the cost-based scheduler, INT4 kernels,
expert caching, prefetching) is described in [docs/architecture.md](docs/architecture.md), which
also covers how the v0.2 pieces fit together.

## Repository layout

```
basp/
  core/         tensor descriptors and formats, dataflow IR, topology, device capabilities
  simulator/    event kernel, fluid bandwidth model, interconnect, kernel execution, energy
  dataflow/     the seven modes, plan vocabulary, adaptive candidate enumeration, prefetcher
  scheduler/    cost model, planning state, joint scheduler, placement, predictor
  memory/       tensor registry, residency, replication (expert cache), allocator, paged KV cache
  streams/      BASP Stream Protocol: commands, completions, activation and weight streams
  kernels/      float64 reference, INT4 format, fused streaming kernels, MoE expert, expert shards
  compiler/     lowering, operator fusion, command emission
  backends/     virtual (simulated devices executing BSP commands), cpu (numeric execution)
  workloads/    model shapes from 4 B to 3 T parameters, routing traces, small numeric MoE model
  verification/ numerical, dependency and accounting checks
  analysis/     critical-path attribution and scaling limits
  dse/          design spaces, Pareto utilities, exhaustive / evolutionary / Bayesian search,
                parametric hardware designs with cost and estimate corners
  calibration/  measurement store with enforced calibration/validation split, fits, validation
  experiments/  architectures, investigations, suites, exploration, calibration glue, gate, reports
benchmarks/     command-line entry points (report, exploration, profiling, calibration, traces)
configs/        hardware profiles, design estimates and gate thresholds (assumptions, labelled)
reports/v0.2/   the generated report, its results and the exploration data
reports/v0.1/   the v0.1 report, kept as a record (produced by the v0.1 model; superseded)
tests/          regression suite
docs/           architecture, calibration, limitations, roadmap
```

`hardware/`, `integrations/`, `backends/cuda` and `backends/metal` from the original layout do
not exist: nothing below the simulator is built until the hardware gate passes.

## Milestones

| Stage | Scope | State |
|---|---|---|
| Dataflow engine | simulator, execution graph, scheduler, tensor memory, INT4 kernels, MoE execution, prefetching | Implemented and tested in simulation; no real model weights |
| Architecture discovery (this release) | eight architectures, investigations, design-space exploration, gate | Implemented; conclusions are uncalibrated |
| Calibration | measured profiles for the two machines, real routing traces | **Tooling ready; measurements not taken** |
| Hardware decision | all gate criteria on calibrated evidence | **NO-GO** today |
| CPU/GPU runtime | real execution against a tuned llama.cpp baseline | Not started |
| RTL, FPGA prototype | Verilator/Yosys first, vendor tools after board selection | Not started; blocked on the gate |

## License

MIT, see [LICENSE](LICENSE).
