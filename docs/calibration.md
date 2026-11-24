# Calibration: replacing assumptions with measurements

The simulator's conclusions are only as good as its hardware parameters. This guide describes how
to measure the two machines chosen for calibration, fit their profiles, and establish how well the
simulator predicts benchmarks it was **not** tuned on.

| Machine label | Hardware | Prior profile |
|---|---|---|
| `i9-11900kf-3080ti` (primary) | Intel i9-11900KF, NVIDIA RTX 3080 Ti 12 GB | `configs/i9_11900kf_rtx3080ti.json` |
| `macbook-m2` (secondary) | Apple MacBook Pro M2, unified memory | `configs/macbook_m2.json` |

Both profiles are **uncalibrated priors** built from public specifications. Edit the installed
memory (`mem_capacity`) before use; everything else is replaced by the steps below.

> **What has and has not been run.** These scripts were developed in a container with no GPU, no
> PyTorch, no llama.cpp and no model files. The measurement store, the fitting, the validation
> logic and the report integration are tested, the fitting only on synthetic measurements generated
> by the simulator itself. The parts that touch real hardware — GPU profiling, running
> `llama-bench`, hooking a real model's routers, reading `.safetensors` — **have never been
> executed**. Each script says so in its docstring and fails loudly when an assumption about an
> external tool's output is wrong. Expect to fix small things on first contact.

## Rules of evidence

1. **Calibration and validation are separate sets.** Every benchmark is assigned to one split
   when it is recorded (`--split` has no default). The calibration split may be used to tune
   parameters. The validation split is used only to judge predictions. The measurement file
   refuses to hold the same benchmark in both splits, and `fit_profile` ignores validation
   measurements whatever it is given.
2. **Only the validation split decides.** The criterion is a median relative error of at most 10%
   and a 95th-percentile error of at most 20% over at least 8 validation measurements
   (`configs/hardware_gate.json`). In-sample error is printed for information and never counts.
3. **Decide the split before looking at results.** A reasonable plan: calibrate on the extremes
   (`-ngl 0` and full offload) of two models; validate on partial offloads and on models that were
   not used for calibration at all.
4. **Calibration validates the model of the existing machine, nothing else.** Performance, power
   and price of compute-memory modules remain estimates with ranges
   (`configs/design_estimates.json`). No measurement on a CPU or GPU narrows them.

## Step 1 — hardware micro-profile (5 minutes per machine)

```bash
pip install -e ".[dev]"            # plus PyTorch (CUDA or MPS build) for the GPU measurements
python -m benchmarks.profile_hardware --machine i9-11900kf-3080ti --out hw_i9.json \
    --template i9_11900kf_rtx3080ti --host-device host --device gpu --link pcie-gpu \
    --profile-out configs/i9_11900kf_rtx3080ti.json
```

On the MacBook use `--machine macbook-m2 --template macbook_m2 --link unified
--profile-out configs/macbook_m2.json`.

This measures sustained memory bandwidth, and with PyTorch the host-to-GPU transfer latency and
bandwidth, the GPU kernel launch overhead and a GPU roofline (memory-bound and arithmetic-bound
matrix products). Fitted fields are listed under `_calibration.measured` in the written profile;
all others remain assumptions. The NumPy INT4 kernel rate it prints is a reference-kernel figure
and is deliberately **not** written into the profile.

## Step 2 — end-to-end benchmarks with llama.cpp

Build llama.cpp with CUDA (Metal on the MacBook) and download GGUF files. Suggested models, all
at the same quantisation (Q4_K_M, about 4.85 bits per weight; compute the exact figure as file
size in bits divided by the parameter count):

| Measurement model name | Purpose |
|---|---|
| `qwen3-4b` | fits the GPU with room to spare |
| `qwen3-8b` | fully GPU-resident baseline |
| `qwen3-14b` | memory pressure: partial offload is necessary at 12 GB |
| `qwen3-30b-a3b` | the project's first workload (mixture of experts) |

Calibration split — the extremes of two models:

```bash
python -m benchmarks.run_llama_bench --machine i9-11900kf-3080ti --split calibration \
    --model-key qwen3-8b=/models/Qwen3-8B-Q4_K_M.gguf \
    --model-key qwen3-30b-a3b=/models/Qwen3-30B-A3B-Q4_K_M.gguf \
    --ngl 0 --ngl 99 --bits-per-weight 4.85 --out measurements.json
```

(`-ngl 99` on a model that does not fit simply fails in llama.cpp; use the largest value that
loads, and record that value.)

Validation split — partial offloads of those models, and two models the fit never sees:

```bash
python -m benchmarks.run_llama_bench --machine i9-11900kf-3080ti --split validation \
    --model-key qwen3-8b=/models/Qwen3-8B-Q4_K_M.gguf \
    --ngl 12 --ngl 24 --bits-per-weight 4.85 --out measurements.json
python -m benchmarks.run_llama_bench --machine i9-11900kf-3080ti --split validation \
    --model-key qwen3-30b-a3b=/models/Qwen3-30B-A3B-Q4_K_M.gguf \
    --ngl 16 --ngl 32 --bits-per-weight 4.85 --out measurements.json
python -m benchmarks.run_llama_bench --machine i9-11900kf-3080ti --split validation \
    --model-key qwen3-4b=/models/Qwen3-4B-Q4_K_M.gguf \
    --model-key qwen3-14b=/models/Qwen3-14B-Q4_K_M.gguf \
    --ngl 0 --ngl 20 --ngl 99 --bits-per-weight 4.85 --out measurements.json
```

Each `llama-bench` run yields a prompt-processing (`prefill`) and a generation (`decode`) record,
so this matrix gives well over the 8 validation measurements required. `--dry-run` prints the
commands without running them. Repeat on the MacBook with `--machine macbook-m2` and the models
that fit 16 GB.

## Step 3 — fit and validate

```bash
python -m benchmarks.calibrate measurements.json --machine i9-11900kf-3080ti \
    --profile-out configs/i9_11900kf_rtx3080ti.json --report calibration_i9.json
```

The command tunes the CPU and GPU arithmetic rate and memory bandwidth on the calibration split
(simulator in the loop), then prints every measurement with its prediction and the verdict on the
validation split: `PASS`, `FAIL`, or `NOT EVALUATED` when there are too few validation
measurements. `--no-fit` scores a profile as it is. The fit starts from whatever the profile
holds and a parameter moves only as far as the measurements require, so run step 1 first: the
micro-profiled values become the starting point.

If validation fails, the simulator's model of a conventional system is wrong somewhere; look at
which measurements are off (partial offloads point at the transfer model, prompt processing at
the arithmetic rate, long contexts at the attention cost model) and fix the model, not the
thresholds. New measurements taken to check a fix belong in the validation split of a fresh file.

## Step 4 — real routing and real weights (the Qwen3-30B-A3B workload)

```bash
python -m benchmarks.record_routing_trace --prompt-file prompt.txt --tokens 256 \
    --out traces/qwen3_chat                       # needs PyTorch, transformers and the model
python -m benchmarks.profile_compression /models/qwen3-30b-a3b/model-00001-of-000NN.safetensors \
    --max-tensors 12 --out compression.json       # needs the safetensors package
```

The first replaces the synthetic routing trace: it records which experts the model's own routers
select, and reports the popularity, stickiness and concentration statistics the synthetic
generator assumes. Loading the 30B model in FP16 needs about 61 GB of memory; `--device-map auto`
spreads it over GPU and host. The second measures what lossless coding of INT4 codes achieves on
real weights and how fast a software decoder runs, replacing the assumed ratio of the `ENTROPY`
format.

## Step 5 — rerun the comparison on the calibrated machine

```bash
python -m benchmarks.explore --base-profile i9_11900kf_rtx3080ti --budget 2000 --workers 4
python -m benchmarks.run_report --base-profile i9_11900kf_rtx3080ti --measurements measurements.json
```

The report's calibration section then lists every measurement against its prediction, and the
hardware gate labels its evidence "simulated, calibrated" only if validation passed **and** the
comparison ran on the validated profile. Commit `measurements.json`, the fitted profiles and the
regenerated report together.

## What calibration cannot establish

* Anything about compute-memory modules: their arithmetic rate, memory efficiency, idle and active
  power, link latency and price are estimates, evaluated at a pessimistic, nominal and optimistic
  corner. The gate is evaluated at the pessimistic corner.
* The cost of the scheduler in a production runtime. The Python scheduler's run time is not
  evidence; the gate reports the per-decision cost budget instead.
* Model quality under the INT4 format on real weights. The numeric co-simulation uses a small
  random-weight model.
