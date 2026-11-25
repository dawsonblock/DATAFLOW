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

Machines: `i9-11900kf-3080ti` (primary), `macbook-m2` (secondary). The runner checks each GGUF
file's shape against the simulator, records bits per weight, resumes after interruption, fits the
profile on the calibration split and scores the validation split into `evidence/<machine>/`.
Before the first run, set the installed RAM (`mem_capacity` of the host) in the machine's profile
in `configs/` and say so in the commit.

The hardware-facing scripts were written without access to a GPU, llama.cpp or model files.
If one fails on first contact, fix the script (keep the fix minimal, add a test reproducing the
real output format), not the plan. Then:

```bash
python -m benchmarks.run_calibration --machine <m> ... --update-profile    # if validation passed
python -m benchmarks.explore --base-profile <profile> --budget 2000 --workers 4
python -m benchmarks.run_report --base-profile <profile>
```

Commit `evidence/<machine>/`, the updated profile and the regenerated `reports/` together, and
state the validation verdict in the commit message.

## Development

- `python -m pytest -q` (about 4 minutes) and `ruff check .` must pass before every commit.
- Python 3.10+; NumPy is the only runtime dependency.
- Generated reports are deterministic: never edit `reports/*/report.md` or `results.json` by hand;
  regenerate them. Report commentary is computed from the results.
- Simulator changes make a stored exploration stale (`basp/evidence.py` detects it); rerun
  `benchmarks.explore` after changing anything under `basp/` other than report or gate code.
