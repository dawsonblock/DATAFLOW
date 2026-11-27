# BASP Dataflow

The dataflow engine of the Block Adaptive Stream Processor (BASP): a bidirectional,
topology-aware, asynchronous dataflow layer that decides, per operation, whether to move data to
a processor or to execute where the data already resides — and an engineering system for finding
out, from evidence, **which architecture is best**, including "a conventional CPU/GPU system".

This repository is **v0.6, a repair-and-validation release**. Before anything is expanded, it fixes
the fourteen defects an audit found in v0.5:

* expert memory that could outlive its owner or exceed its budget;
* sampling memory quadratic in the vocabulary;
* inconsistent session state after generation;
* simulator lowering that left out the embedding, the final norm and shared experts;
* evidence that could be selected favourably, was scoped too broadly, or lacked identity;
* a package that worked only from the checkout.

It also adds a pre-declared protocol for any speed claim: paired, counterbalanced request-latency
trials with prompts as the independent units, frozen before the decision study runs.

v0.5 (speculative decoding, grouped expert reads, real Qwen3-30B-A3B execution in about 10 GB, a
recorded routing trace, a shared-prefix KV cache, a PyTorch CPU backend) and v0.4 (evidence
receipts, memory domains, a runtime verified against transformers) remain. The v0.6 results are
exploratory studies on a 4-core development container. The decision study and the native-engine
comparison (llama.cpp) are for the target machine.

It is a Python research system that needs only NumPy (plus PyTorch and transformers for the
reference and the optional backend).

> **Status of the numbers.** Runtime results are measurements (E3) on one 4-core CPU with this
> runtime, from exploratory studies that cannot support a decision. Speed-ups are against the
> runtime's own plain decoding, **not** against llama.cpp or KTransformers. No calibration measurement of the target machines exists yet, so every
> architecture figure is an uncalibrated simulation (E1), and everything about compute-memory modules
> is an estimate with a range. Nothing here is GPU inference: the CUDA, Metal and GGML backends are
> declared and report themselves unavailable. See [docs/limitations.md](docs/limitations.md).

## Fewer weight reads per token: what was measured

Report section 11, from `evidence/speculative/`:

* **Reading each expert once for several tokens works on real routing.** In Qwen3-30B-A3B every token
  selects 8 of 128 experts per layer. If 4 consecutive tokens share one pass, a layer needs 4.7
  distinct experts per token (1.7x fewer reads); with 8 tokens, 3.6 (2.2x fewer). That is the ceiling
  if every proposed token is accepted.
* **Speculative decoding gets part of it.** With Qwen3-0.6B drafting 4 tokens, 58% were accepted,
  3.0 tokens came out of each 30B pass, and expert reads fell from 384 to 300 per token (1.28x fewer),
  with exactly the same tokens. In this dequantisation-bound runtime that was 6.0 → 4.1 s per token.
* **A poor drafter makes it worse.** Prompt lookup (3% acceptance on that prose) *raised* expert reads
  to 596 per token: rejected proposals still read their experts. Speculation must be switched off
  when acceptance is low; that is the controller's job.
* **Dense models gain only where drafting is easy and verification is cheap.** Qwen3-0.6B drafting for
  Qwen3-1.7B: code (92–95% acceptance) 1.28–1.47x faster on the PyTorch backend, explanations 1.2x,
  prose (34–48% acceptance) no gain. On the NumPy backend most configurations were *slower* (down to
  0.64x): its BLAS multiplies a few rows at about four times the cost of one, so verification is not
  cheap. Every run produced exactly the target's greedy tokens.
* **Expert prediction on real routing:** 50% of the experts a token selects are among 8 predicted
  ahead (frequency, recency and transitions), 68% among 16, 85% among 32.
* **Shared prefixes:** a 256-token system prompt reused from the cache cut time to the first token
  from about 3.3 s to 0.3–0.9 s.

These say the direction works: fewer weight reads per token, with output unchanged. They do not show
that BASP beats an optimised engine; that comparison (llama.cpp with `--n-cpu-moe`, KTransformers)
on the same machine is the next requirement (v0.6).

## Where things stand

**Hardware development: NO-GO**, and real routing makes the case for module hardware weaker still.
No FPGA or RTL work has been started, by decision ([docs/roadmap.md](docs/roadmap.md)).

Full report: [reports/v0.6/report.md](reports/v0.6/report.md) (machine-readable:
[results.json](reports/v0.6/results.json), including every gate receipt; [dse.json](reports/v0.6/dse.json);
inputs and environment: [provenance.json](reports/v0.6/provenance.json)). First workload: Qwen3-30B-A3B
shapes at 4.5 bits per weight (16.0 GiB), single-user decoding, 12 GiB GPU, with the model's **recorded
routing**. v0.6 adds the embedding lookup, the final norm and shared experts to the simulator. For
this model the figures move by about 1%.

| Architecture | tokens/s (range over estimates) | vs A, nominal / pessimistic | per dollar vs A, nominal / pessimistic |
|---|---|---|---|
| A Centralized (conventional CPU/GPU) | 227 | 1.00x | 1.00x |
| B Distributed | 47 (38–59) | 0.21x / 0.17x | 0.04x / 0.01x |
| C Pipeline | 28 (23–35) | 0.12x / 0.10x | 0.03x / 0.01x |
| D Expert-local | 109 (92–129) | 0.48x / 0.41x | 0.09x / 0.02x |
| E Tensor-parallel | 131 (111–153) | 0.58x / 0.49x | 0.11x / 0.03x |
| F Adaptive hybrid | 231 (225–235) | 1.02x / 0.99x | 0.20x / 0.06x |
| G Spatial dataflow | 113 (96–133) | 0.50x / 0.42x | 0.10x / 0.03x |
| H Hierarchical | 234 (230–256) | 1.03x / 1.01x | 0.22x / 0.06x |

What the simulation says, under its assumptions:

* **Real routing helps the conventional system most.** Qwen3-30B-A3B's real expert selections are
  more concentrated than synthetic ones, so the 12 GiB GPU's expert cache hits more often.
  Architecture A reaches 227 tokens/s (171 with synthetic routing in v0.4). The best module
  design's advantage is 1.03x (1.23x with synthetic routing), at about five times the price:
  0.22x the throughput per dollar.
* **A bigger GPU beats every module design for this model.** The 16 GiB model fits a 32 GiB
  (RTX 5090-class) GPU: 436 tokens/s for an estimated $3,600. None of the 1,980 systems explored is
  faster; designs adding modules to the same GPU only match it, at higher prices. The Pareto front
  (speed, tokens per joule, price) is entirely conventional.
* **Modules help only when the model is far larger than the GPU, and never per dollar.** With a
  12 GiB GPU they reach 3.2x for a 70B dense model and 2.9x for DeepSeek-V3 shapes (synthetic
  routing for those). The gate's most favourable regime (70B, pessimistic estimates) gives 2.7x
  throughput and 1.3x energy efficiency, but only 0.17x per dollar.
* **Each candidate is judged on its own scaling.** From one to four modules, H scales at 26% (its
  GPU does most of the work) and E on the 70B model at 93%.
* **Expert caching policy matters on a small GPU** (recorded routing, simulated costs, E1). With a
  2 GiB expert cache, valuing experts by expected reuse and running rare ones on the CPU is 1.84x
  faster than LRU. At 4 GiB the predictive policy is 1.21x faster. At 8 GiB, LRU and LFU are best.
* **Real weights run and match the reference** (re-measured on the v0.6 runtime). Qwen3-0.6B (Q8_0
  GGUF) agrees with transformers to 2.8e-6 relative error (NumPy backend) and 1.8e-6 (PyTorch
  backend), with 16 identical greedy tokens and perplexity 1.000001x. MoE routing matches exactly on
  small random-weight models. These receipts concern Qwen3-0.6B and do not qualify the 30B regime's
  gate criteria.

Exploratory request-latency studies under the v0.6 protocol, all on a 4-core development container.
Each is paired AB/BA, has the prompt as its unit, and decides nothing:

* **Tuning on Qwen3-1.7B** (0.6B draft, PyTorch CPU backend, six development prompts) selects
  drafting 2 tokens: 1.13x (1.03–1.24). Adaptive drafting gives 1.04x, prompt lookup 0.90x, and
  drafting 4 tokens 0.83x.
* **Qwen3-30B-A3B feasibility** (same settings, 2 GiB hard ceiling on decoded experts, which
  held): 1.13x (1.10–1.16) over the six prompts. Coding 1.23x, technical 1.13x, prose 1.03x.
  Speculation cut logical expert requests per returned token from 423 to 363. Storage reads per
  token rose from 0.53 to 0.65 GB, and the cause is not established.
* **None of this compares with llama.cpp.** The competitive comparison (tuned llama.cpp against
  llama.cpp plus one BASP-derived change) and the frozen decision study on 24 held-out prompts are
  for the target machine.

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
python -m pytest -q                              # regression suite, about nine minutes
python -m benchmarks.dataflow_model              # closed-form comparison of specification section 6
python -m benchmarks.explore --budget 60 --workers 4 --out /tmp/dse.json   # a small exploration
python -m benchmarks.run_report --quick          # every experiment, short runs, a few minutes
python -m benchmarks.explore --budget 2000 --workers 4 --trace evidence/traces/qwen3-30b-a3b.decode.json
python -m benchmarks.run_report --trace evidence/traces/qwen3-30b-a3b.decode.json   # -> reports/v0.6/
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
from basp.runtime.speculative import AdaptiveController, ModelDrafter, speculative_generate
target = Engine("Qwen3-1.7B-Q8_0.gguf", prefer=("torch", "cpu"))   # any Qwen3 / Qwen3-MoE GGUF
draft = Engine("Qwen3-0.6B-Q8_0.gguf", prefer=("torch", "cpu"))    # same tokenizer
prompt = [785, 6722, 315, 9625, 374]                                # token ids in, token ids out
with target.session(64) as s:
    print(generate(s, prompt, 12).tokens)
with target.session(64) as s:
    r = speculative_generate(s, ModelDrafter(draft, 64), prompt, 12, controller=AdaptiveController())
    print(r.tokens, r.summary()["acceptance"])                     # the same tokens, fewer target passes
PY
python -m benchmarks.speculative_bench moe --target Qwen3-30B-A3B-Q4_K_M.gguf --draft Qwen3-0.6B-Q8_0.gguf \
    --tokenizer <HF tokenizer dir> --out evidence/speculative/qwen3-30b-a3b.json --trace evidence/traces/qwen3-30b-a3b
python -m benchmarks.verify_runtime --gguf Qwen3-0.6B-Q8_0.gguf --model qwen3-0.6b \
    --tokenizer <HF tokenizer dir> --out evidence/verification/qwen3-0.6b
```

The runtime dequantises dense weights to float32 and experts on demand, so it is a reference
implementation for correctness and for counting what a method saves, not a fast engine.

## What v0.6 adds

A repair-and-validation release. It fixes fourteen audit findings, each with a test of the
behaviour (details in [docs/roadmap.md](docs/roadmap.md)), and sets up a measurement protocol for
any speed claim:

| Item | Where |
|---|---|
| Owner-safe, transactional expert store with a hard memory ceiling (bounded expert groups, same summation order) | `basp/runtime/weights.py`, `basp/memory/domains.py` |
| Linear-memory sampling acceptance; one documented session state after generation, with continuation | `basp/runtime/speculative.py`, `generation.py` |
| Shared experts, embedding lookup and final norm in the simulator, reconciled with an independent count; routing provenance per experiment | `basp/compiler/lower.py`, `basp/verification/reconcile.py` |
| Calibration evidence scoped to the regimes it covers; receipts chosen by content hash, worst value deciding; complete identities and one validity rule | `basp/calibration/coverage.py`, `basp/evidence/` |
| Request-latency protocol: arms A-E, paired AB/BA trials, prompt-level bootstrap, freezing, Amdahl headroom | `configs/benchmark_protocol.json`, `basp/bench/`, `benchmarks/run_study.py` |
| Installable package with its configurations; release check that runs the reference verification | `setup.py`, `benchmarks/release_check.py` |

```bash
python -m benchmarks.run_study tune --target <target.gguf> --draft <draft.gguf> --tokenizer <dir> --out evidence/studies/tuning
python -m benchmarks.run_study freeze --comparison internal        # only on the machine that decides
python -m benchmarks.run_study run --study decision --comparison internal ... --out evidence/studies/decision
python -m benchmarks.run_study headroom --fraction 0.30 --reduction 0.50 --target 1.20
python -m benchmarks.release_check --require-reference
```

## What v0.5 added

| Item | Where | Evidence |
|---|---|---|
| Speculative decoding (draft model, prompt lookup), exact greedy and sampling acceptance, KV rollback | `basp/runtime/speculative.py` | `tests/test_v05_speculative.py` (identical tokens; chi-square test of the sampling distribution) |
| Adaptive draft-length controller | same | chooses `k` from measured acceptance and costs; `k = 0` when speculation does not pay |
| Grouped expert reads with byte accounting; parallel on-demand expert dequantisation | `basp/runtime/weights.py`, `model.py` | fewer expert reads for the same tokens (tests and Qwen3-30B-A3B) |
| Memory-lean execution of Qwen3-30B-A3B | `Engine(..., embedding="rows", expert_cache_bytes=...)` | runs in about 10 GB; coherent output |
| Shared-prefix KV cache | `basp/runtime/prefix_cache.py` | logits match a full prefill |
| PyTorch CPU backend (optional) | `basp/backends/torch_cpu` | passes the same transformers parity checks |
| Measurement tool and recorded evidence | `benchmarks/speculative_bench.py`, `evidence/speculative/`, `evidence/traces/` | report section 11 |

v0.4 (evidence receipts, memory domains, the verified runtime) and earlier:
[docs/roadmap.md](docs/roadmap.md), [docs/architecture.md](docs/architecture.md).

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
  runtime/      the physical runtime: device manager, plans, dispatcher, sessions, Qwen3 model, generation,
                speculative decoding, prefix cache
  evidence/     evidence levels, identities, receipts, qualification, provenance, validity
  bench/        request-latency protocol: schedule, arms, paired analysis, decisions
  experiments/  architectures, investigations, suites, exploration, calibration glue, gate, reports
benchmarks/     command-line entry points (report, exploration, profiling, calibration, traces)
configs/        hardware profiles, design estimates, gate thresholds and the benchmark protocol (labelled)
reports/v0.6/   the generated report, its results (with receipts), provenance and the exploration data
reports/v0.1/ to reports/v0.5/   earlier reports, kept as records (superseded)
evidence/       verification/ (runtime parity), studies/ (request-latency studies under the protocol),
                speculative/ (v0.5 measurements), traces/ (recorded routing); <machine>/ measurements
                (created by run_calibration)
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
| v0.4 | Evidence receipts, memory domains, a verified runtime for real GGUF models | Done for the CPU |
| v0.5 | Speculative execution and grouped expert reads, measured on real models | Done on the CPU |
| v0.6 | Repairs of the audit findings; request-latency protocol; installable package | **Done** (this release); no decision study yet |
| Calibration | Held-out validation on the calibration machines | **Ready to run**; nothing measured |
| v0.7 | Tuned llama.cpp baseline (arm D) on the i9 + 3080 Ti; at most one bounded native intervention (arm E), only if its headroom allows 1.20x | Not started; needs the target machine |
| v0.8 | Native CPU/CUDA/Metal backends; adaptive expert placement on real hardware | Only if v0.7's decision study meets its criterion |
| v0.9 | Architecture I; distributed execution | Not started |
| v0.10–v1.0 | Accelerator RTL, FPGA prototype | Blocked on the gate |

## License

MIT, see [LICENSE](LICENSE).
