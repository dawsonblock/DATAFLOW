# BASP Dataflow

The dataflow engine of the Block Adaptive Stream Processor (BASP): a bidirectional,
topology-aware, asynchronous dataflow layer that decides, per operation, whether to move data to
a processor or to execute where the data already resides — and an engineering system for finding
out, from evidence, **which architecture is best**, including "a conventional CPU/GPU system".

This repository is **v0.4, Evidence-Gated Heterogeneous Execution**: the v0.3 simulator,
scheduler, design-space explorer, calibration plan and evidence-gated decision, now with

* **evidence receipts** — every value the hardware gate uses is bound to the identities that
  produced it (simulator source, configuration, profile, validation, machine, model weights,
  runtime, benchmark); a stale, mislabelled or other-model receipt is rejected;
* **physical memory domains** — shared accounting (a unified-memory Mac has one pool), admission
  control, reference-counted eviction safety, asynchronous migration, and expert-cache policies;
* **a physical runtime that executes real GGUF models** on the CPU (Qwen3 dense and MoE, 11 GGUF
  weight formats), verified against Hugging Face transformers.

It is a Python research system that needs only NumPy (plus PyTorch and transformers to run the
verification reference).

> **Status of the numbers.** No calibration measurement of the target machines exists yet: every
> architecture figure is an uncalibrated simulation (E1), everything about compute-memory modules
> is an estimate with a pessimistic–nominal–optimistic range, and routing is synthetic. The only
> measurements (E3) are the runtime's numerical parity with transformers. The CPU runtime is not
> GPU inference: the CUDA, Metal and GGML backends are declared and report themselves
> unavailable. See [docs/limitations.md](docs/limitations.md).

## Where things stand

**Hardware development: NO-GO**, and the simulation no longer leaves much doubt about the
direction. No FPGA or RTL work has been started, by decision ([docs/roadmap.md](docs/roadmap.md)).

Full report: [reports/v0.4/report.md](reports/v0.4/report.md) (machine-readable:
[results.json](reports/v0.4/results.json), including every gate receipt; [dse.json](reports/v0.4/dse.json);
inputs and environment: [provenance.json](reports/v0.4/provenance.json)). First workload: Qwen3-30B-A3B
shapes at 4.5 bits per weight (16.0 GiB), single-user decoding, 12 GiB GPU.

| Architecture | tokens/s (range over estimates) | vs A, nominal / pessimistic | per dollar vs A, nominal / pessimistic |
|---|---|---|---|
| A Centralized (conventional CPU/GPU) | 171 | 1.00x | 1.00x |
| B Distributed | 46 (38–59) | 0.27x / 0.22x | 0.06x / 0.01x |
| C Pipeline | 28 (23–35) | 0.16x / 0.14x | 0.03x / 0.01x |
| D Expert-local | 107 (90–126) | 0.62x / 0.53x | 0.12x / 0.03x |
| E Tensor-parallel | 132 (111–153) | 0.77x / 0.65x | 0.15x / 0.04x |
| F Adaptive hybrid | 200 (182–214) | 1.17x / 1.07x | 0.23x / 0.07x |
| G Spatial dataflow | 110 (93–130) | 0.64x / 0.55x | 0.12x / 0.03x |
| H Hierarchical | 210 (197–235) | 1.23x / 1.15x | 0.26x / 0.07x |

What the simulation says, under its assumptions:

* **A bigger GPU beats every module design for this model.** The 16 GiB model fits a 32 GiB
  (RTX 5090-class) GPU: 438 tokens/s for an estimated $3,600. That is faster than any of the 1,980
  systems explored and cheaper than every module design. The Pareto front (speed, tokens per
  joule, price) is entirely conventional: 12, 24 and 32 GiB GPUs.
* **With the 12 GiB GPU, the best module design is 1.23x faster at about five times the price.**
  That is a quarter of the throughput per dollar at nominal estimates, and 7% at the pessimistic
  corner.
* **Modules help only when the model is far larger than the GPU, and never per dollar.** With a
  12 GiB GPU: 3.2x for a 70B dense model and 3.0x for DeepSeek-V3 shapes, at 0.64x and 0.76x the
  throughput per dollar. The gate's most favourable regime (70B, pessimistic estimates) gives 2.7x
  throughput and 1.3x energy efficiency, but only 0.17x per dollar.
* **Each candidate is judged on its own scaling now.** From one to four modules H scales at 32%
  (its GPU does most of the work), E on the 70B model at 93%.
* **Two v0.2 results were artefacts.** Module links on the local fabric were free, so the v0.2
  front's hierarchical designs all used the cheapest link. The pessimistic price corner also
  favoured modules. Both are fixed, and H dropped off the front.
* **Numerics hold within tolerance** on a small random-weight model (relative error 4.5e-4,
  identical routing and tokens). Dataflows that keep the reduction order are bit-identical; those
  that reorder FP32 sums agree within rounding.
* **Real weights now run, and match the reference.** The CPU runtime executes Qwen3-0.6B (Q8_0
  GGUF) with logits within 2.8e-6 relative error of transformers on the same weights (target
  0.002), every layer's output matching, 16 identical greedy tokens ("The capital of France is" →
  " Paris. The capital of France is also the capital of the Republic of France.") and perplexity
  1.000001x the reference. MoE routing matches transformers exactly on small random-weight
  models. These receipts concern Qwen3-0.6B and do not qualify the 30B regime's gate criteria.
* **Expert caching on a small GPU depends on capacity.** On synthetic Qwen3-30B-A3B routing, the
  cost-aware and predictive caches (run rarely used experts on the CPU instead of migrating them)
  are about 14% faster than LRU with a 2 GiB expert cache, while LFU and LRU win at 4 and 8 GiB
  (report section 10; simulated, E1).

## What you need to run

The verdict means little until the simulator is calibrated. Each machine needs one command (and
Claude Code started in the repository picks up the rules in [CLAUDE.md](CLAUDE.md)):

```bash
python -m benchmarks.run_calibration --machine i9-11900kf-3080ti \
    --models-dir ~/models --llama-bench ~/llama.cpp/build/bin/llama-bench
python -m benchmarks.run_calibration --machine macbook-m2 \
    --models-dir ~/models --llama-bench ~/llama.cpp/build/bin/llama-bench
```

It runs the pre-registered plan (`configs/calibration_plan.json`: 8 calibration and 28 validation
measurements on the desktop, 8 and 14 on the MacBook), checks every GGUF file against the
simulator's model shape, keeps going when a configuration does not fit GPU memory, fits the
machine profile on the calibration split only and judges it on the validation split. Results go
to `evidence/<machine>/`; commit that folder. Details, prerequisites and what to do on a failed
validation: [docs/calibration.md](docs/calibration.md).

No local machine at hand? [`notebooks/calibrate_colab.ipynb`](notebooks/calibrate_colab.ipynb) runs
a third plan section (`colab-t4`) on a free Colab T4: it tests the tooling and the simulator's model of
a CPU plus discrete GPU on real hardware, though it says nothing about the i9 or the M2.

The plan's model files exist on Hugging Face under the listed names, and the headers of the real
files match the simulator's shapes to the parameter. llama-bench itself has never been run by this
project: its output format was checked against its source, and the whole runner was exercised
against a fake llama-bench. Expect small fixes on first use.

## Quick start

```bash
pip install -e ".[dev]"                          # or just: pip install numpy pytest
python -m pytest -q                              # regression suite, about five minutes
python -m benchmarks.dataflow_model              # closed-form comparison of specification section 6
python -m benchmarks.explore --budget 60 --workers 4 --out /tmp/dse.json   # a small exploration
python -m benchmarks.run_report --quick          # every experiment, short runs, a few minutes
python -m benchmarks.explore --budget 2000 --workers 4                     # full exploration (~25 min)
python -m benchmarks.run_report                  # the full report -> reports/v0.4/ (~15 min)
```

Both the exploration and the report are deterministic: on the same Python and NumPy versions
(the committed files were generated with Python 3.13 and NumPy 2.5) the same code, profiles,
estimates and seeds reproduce the committed files exactly. Across Python versions the figures
agree to about one part in 10^16 (float summation changed in Python 3.12), with the same designs
on the Pareto front.

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

## Running real models

```bash
pip install -e ".[dev,verify]"     # verify: torch and transformers, the reference
python - <<'PY'
from basp.runtime import Engine, generate
engine = Engine("Qwen3-0.6B-Q8_0.gguf")          # any Qwen3 / Qwen3-MoE GGUF the runtime supports
with engine.session(64) as s:
    print(generate(s, [785, 6722, 315, 9625, 374], 12).tokens)   # token ids in, token ids out
print(engine.report()["plan"])                    # which backend runs what; why others were rejected
PY
python -m benchmarks.verify_runtime --gguf Qwen3-0.6B-Q8_0.gguf --model qwen3-0.6b \
    --tokenizer <HF tokenizer dir> --out evidence/verification/qwen3-0.6b
```

The runtime dequantises weights to float32 (about four bytes per parameter of memory), so it is a
reference implementation for correctness, not a fast engine.

## What v0.4 adds

| Item | Where | Evidence |
|---|---|---|
| Evidence v2: receipts, metric-specific qualification, identity binding, directory manifests | `basp/evidence/` | `tests/test_v04_evidence.py`, including every adversarial case of the v0.4 plan |
| Gate decided from receipts | `basp/experiments/gate.py` | a throughput receipt never qualifies energy; stale or other-model receipts are rejected |
| Calibration repairs (repeats count once, per-machine verdicts, executable and weight hashes, profile fingerprints, coverage) | `basp/calibration/`, `benchmarks/run_calibration.py` | as above |
| Memory domains, admission, migration, prefetch, expert-cache policies | `basp/memory/` | `tests/test_v04_memory.py` |
| Unified-memory design repair | `basp/dse/hardware.py` | designs on the M2 base are now buildable |
| Physical runtime and `ExecutionBackend` | `basp/runtime/`, `basp/backends/` | `tests/test_v04_runtime.py` |
| GGUF tensor data and dequantisation (11 formats) | `basp/models/quants.py`, `gguf_tensors.py` | bit-identical to the reference `gguf` package |
| Verification against transformers | `basp/verification/`, `benchmarks/verify_runtime.py`, `evidence/verification/` | real Qwen3-0.6B; tiny dense and MoE models in CI |

Deferred: architecture I, the native C++/CUDA/Metal engine, a real MoE model end to end
([docs/roadmap.md](docs/roadmap.md)). v0.3 and earlier: [docs/architecture.md](docs/architecture.md).

## Repository layout

```
basp/
  core/         tensor descriptors and formats, dataflow IR, topology, device capabilities
  simulator/    event kernel, fluid bandwidth model, interconnect, kernel execution, energy
  dataflow/     the seven modes, plan vocabulary, adaptive candidate enumeration, prefetcher
  scheduler/    cost model, planning state, joint scheduler, placement, predictor
  memory/       tensor registry, residency, replication, allocator, paged KV cache; physical memory
                domains, admission, migration, prefetch, expert-cache policies
  streams/      BASP Stream Protocol: commands, completions, activation and weight streams
  kernels/      float64 reference, INT4 format, fused streaming kernels, MoE expert, expert shards
  compiler/     lowering, operator fusion, command emission
  backends/     virtual (simulated devices executing BSP commands), cpu (numeric payload and the
                runtime's NumPy backend), execution contract; cuda/metal/ggml/remote: declared, unavailable
  workloads/    model shapes from 4 B to 3 T parameters, routing traces, small numeric MoE model
  verification/ numerical, dependency and accounting checks; runtime parity with transformers
  analysis/     critical-path attribution and scaling limits
  dse/          design spaces, Pareto utilities, exhaustive / evolutionary / Bayesian search,
                parametric hardware designs with cost and estimate corners
  calibration/  measurement store with enforced calibration/validation split, fits, validation
  models/       GGUF headers, tensor data, dequantisation, writer for test models
  runtime/      the physical runtime: device manager, plans, dispatcher, sessions, Qwen3 model, generation
  evidence/     evidence levels, identities, receipts, qualification, provenance
  experiments/  architectures, investigations, suites, exploration, calibration glue, gate, reports
benchmarks/     command-line entry points (report, exploration, profiling, calibration, traces)
configs/        hardware profiles, design estimates and gate thresholds (assumptions, labelled)
reports/v0.4/   the generated report, its results (with receipts), provenance and the exploration data
reports/v0.1/ to reports/v0.3/   earlier reports, kept as records (superseded)
evidence/       verification/ (runtime parity); <machine>/ measurements (created by run_calibration)
tests/          regression suite
docs/           architecture, calibration, limitations, roadmap
```

`hardware/` and `integrations/` from the original layout do not exist: nothing below the simulator
is built until the hardware gate passes. `backends/cuda` and `backends/metal` exist only as
declarations that refuse every call.

## Milestones

| Release | Objective | State |
|---|---|---|
| v0.1–v0.2 | Dataflow engine; architecture discovery | Done, simulated |
| v0.3 | Repair and evidence infrastructure | Done |
| v0.4 | Evidence receipts, memory domains, a verified runtime for real GGUF models | **Done for the CPU** (this release) |
| Calibration | Held-out validation on the calibration machines | **Ready to run**; nothing measured |
| v0.5 | Native CPU, CUDA and Metal backends behind the same contract and parity harness | Not started |
| v0.6 | Adaptive expert placement on real hardware, measured against llama.cpp | Components in place |
| v0.7 | Architecture I; distributed execution | Not started |
| v0.9–v1.0 | Accelerator RTL, FPGA prototype | Blocked on the gate |

## License

MIT, see [LICENSE](LICENSE).
