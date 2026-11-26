# BASP Dataflow — working rules

Research software for deciding, from evidence, whether a compute-memory accelerator beats a
conventional CPU/GPU system for large-model inference. Read `docs/limitations.md` before trusting
any number, and `docs/calibration.md` before measuring anything.

## Rules of evidence (non-negotiable)

- **Never change a benchmark's split after seeing a result.** The calibration/validation split is
  pre-registered in `configs/calibration_plan.json`. Do not move entries between splits, delete
  validation entries, or edit thresholds in `configs/hardware_gate.json` to make anything pass. To
  test a model fix, add *new* validation benchmarks under a new plan `version`.
- **Only the validation split judges the simulator.** In-sample (calibration-split) error is
  informational. `fit_profile` must never read validation measurements.
- **Label evidence.** E0 estimate/argument, E1 uncalibrated simulation, E2 simulation validated on
  held-out measurements of *existing* hardware, E3 direct measurement (`basp/evidence.py`). A claim
  about hypothetical modules can never exceed E2 before a prototype exists.
- **A gate criterion passes only for one candidate architecture in one regime, on its own
  evidence** (`basp/experiments/gate.py`). Missing evidence counts against.
- **Report failures, don't hide them.** A configuration that fails (e.g. out of GPU memory) is
  recorded in `runs.json`. A validation failure means the model of a conventional system is wrong:
  diagnose and fix the model, then validate on fresh measurements.
- **No FPGA/RTL work and no hardware purchases** until the gate passes on calibrated evidence.
- Conventional baselines must be properly tuned (e.g. `--n-cpu-moe` for MoE models on small
  GPUs), never artificially weak.

## Calibrating a machine (the current task)

```bash
pip install -e ".[dev]"
python -m benchmarks.run_calibration --machine i9-11900kf-3080ti --dry-run      # see the plan
python -m benchmarks.run_calibration --machine i9-11900kf-3080ti \
    --models-dir ~/models --llama-bench ~/llama.cpp/build/bin/llama-bench
```

Machines: `i9-11900kf-3080ti` (primary), `macbook-m2` (secondary), `colab-t4` (Google Colab;
run it with `notebooks/calibrate_colab.ipynb`). The runner checks each GGUF
file's shape against the simulator, records bits per weight, resumes after interruption, fits the
profile on the calibration split and scores the validation split into `evidence/<machine>/`.
Before the first run, set the installed RAM (`mem_capacity` of the host) in the machine's profile
in `configs/` and say so in the commit.

The hardware-facing scripts were written without access to a GPU or llama.cpp (the model files'
headers were checked against the simulator's shapes; `tests/fixtures/qwen3_gguf_headers.json`).
If one fails on first contact, fix the script (keep the fix minimal, add a test reproducing the
real output format), not the plan. Then:

```bash
python -m benchmarks.run_calibration --machine <m> ... --update-profile    # if validation passed
python -m benchmarks.explore --base-profile <profile> --budget 2000 --workers 4
python -m benchmarks.run_report --base-profile <profile>
```

Commit `evidence/<machine>/`, the updated profile and the regenerated `reports/` together, and
state the validation verdict in the commit message.

## The physical runtime (v0.4)

`basp/runtime/` executes real GGUF models (Qwen3 dense and MoE) on the CPU in NumPy float32.
CUDA, Metal, GGML and remote backends are declared but **not implemented**: never describe them,
or anything here, as GPU inference. Verify any runtime change against transformers before using it:

```bash
pip install -e ".[dev,verify]"     # torch and transformers: the reference
python -m pytest -q tests/test_v04_runtime.py tests/test_v04_verification.py
python -m benchmarks.verify_runtime --gguf ~/models/Qwen3-0.6B-Q8_0.gguf --model qwen3-0.6b \
    --tokenizer <HF tokenizer dir> --out evidence/verification/qwen3-0.6b
```

The parity receipts are bound to the runtime's source fingerprint: after changing runtime code,
rerun `verify_runtime` (the report marks older results stale). Receipts about one model never
qualify a gate regime about another.

Speculative execution (v0.5) is measured, never assumed: `python -m benchmarks.speculative_bench
dense|moe|prefix ...` writes `evidence/speculative/*.json` (and with `moe --trace`, a recorded
routing trace). Report slowdowns as they are; the adaptive controller exists so that speculation
that does not pay is switched off, not hidden. Wall-clock figures from this runtime are not
comparisons with llama.cpp or KTransformers; such a comparison needs those engines on the same
machine.

## Request-latency studies (v0.6)

Any claim that something is faster comes from a study under `configs/benchmark_protocol.json`
(`basp/bench/`, `python -m benchmarks.run_study`), never from an ad-hoc run:

- The primary outcome is end-to-end request latency for exactly the requested tokens, in paired,
  counterbalanced (AB/BA) trials at concurrency 1. The prompt is the independent unit: the result
  is the equal-weight geometric mean of per-prompt paired ratios, with a prompt-level bootstrap
  stratified by category. An interval that includes 1 is *inconclusive*.
- Tuning (`run_study tune`) uses only the development prompts and only the protocol's listed
  candidates. Then `run_study freeze --comparison <c>` fixes the protocol (fingerprinted). Only a
  frozen decision study on the held-out prompts can support a decision; feasibility studies,
  pilots (`--tokens`, `--limit-prompts`, `--repetitions`) and tuning are exploratory.
- **Never add prompts, repetitions or candidates, change criteria or re-tune after seeing a
  decision result.** A changed protocol is a new version and a new study. Do not freeze or run a
  decision study for the user without being asked.
- Arms A/B (this runtime) are internal; the competitive comparison is D (tuned native engine) vs
  E (BASP's change inside that engine). Native arms need a running server pinned in the protocol
  (`setup.native`: revision, binary hash, command). None has been run from this repository.
- Before a native intervention, compute its headroom from a profile
  (`run_study headroom --fraction f --reduction r`): `1/(1 - f r)`. If it cannot reach the
  criterion, do not build it.
- Studies carry complete identities (`basp.evidence.validity`): a study measured by other runtime
  or tool code is shown NOT APPLICABLE in the report. Rerun it; never relabel it.

## Packaging

`python -m benchmarks.release_check --require-reference` builds the wheel and sdist, installs each
in a clean environment outside the checkout and runs the commands, a tiny model end to end and the
reference verification (torch and transformers; with `BASP_REQUIRE_REFERENCE=1` a missing reference
is a failure, not a skip). Configurations ship inside the package (`basp/_configs`, `setup.py`).

## Development

- `python -m pytest -q` (about 4 minutes) and `ruff check .` must pass before every commit.
- Python 3.10+; NumPy is the only runtime dependency.
- Generated reports are deterministic: never edit `reports/*/report.md` or `results.json` by hand;
  regenerate them. Report commentary is computed from the results.
- Simulator changes make a stored exploration stale (`basp/evidence.py` detects it); rerun
  `benchmarks.explore` after changing anything under `basp/` other than report or gate code.
