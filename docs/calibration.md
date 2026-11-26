# Calibration: replacing assumptions with measurements

The simulator's conclusions are only as good as its model of real hardware. This guide measures
the two calibration machines, fits their profiles, and establishes how well the simulator predicts
benchmarks it was **not** tuned on.

| Machine label | Hardware | Profile |
|---|---|---|
| `i9-11900kf-3080ti` (primary) | Intel i9-11900KF, NVIDIA RTX 3080 Ti 12 GB | `configs/i9_11900kf_rtx3080ti.json` |
| `macbook-m2` (secondary) | Apple MacBook Pro M2, 16 GB unified memory | `configs/macbook_m2.json` |
| `colab-t4` (tooling and model check) | Google Colab T4 runtime: 2 vCPUs, about 12.7 GB RAM, Tesla T4 16 GB | `configs/colab_t4.json` |

The Colab machine says nothing about the other two; it tests the tooling on real hardware and the
simulator's model of a CPU plus discrete GPU, without needing either of the calibration machines.

Both profiles are **uncalibrated priors** built from public specifications. Set the installed RAM
(`mem_capacity` of `host`; on the M2, of both devices, which share one memory domain) before use.

> **What has and has not been run.** These tools were developed without a GPU or llama.cpp. What
> *was* checked: the plan's model files exist under the listed names, and the headers of the real
> files (Qwen3 4B, 8B, 14B and 30B-A3B, Q4_K_M) match the simulator's model shapes to the parameter
> (`tests/fixtures/qwen3_gguf_headers.json`); the llama-bench JSON format against the llama-bench
> source vendored in llama-cpp-python 0.3.38; the GGUF header reader against the reference `gguf`
> package; the Qwen3-MoE router interface against transformers 5.24; the whole runner end to end
> against a fake llama-bench whose results come from a simulated machine with known parameters
> (the fit recovers them). What was **not**: llama-bench itself, and any real measurement. Expect
> small fixes on first contact; fix the scripts, never the plan.

## Rules of evidence

1. **The split is pre-registered.** `configs/calibration_plan.json` fixes, before any result
   exists, which benchmarks may tune the simulator (calibration) and which only judge it
   (validation). Never move a benchmark between splits after seeing a result. A fix to the model
   is tested on *new* validation benchmarks under a new plan version.
2. **Only the validation split decides.** Median relative error at most 10%, 95th percentile at
   most 20%, over at least 8 held-out measurements (`configs/hardware_gate.json`). In-sample error
   is printed for information only. `fit_profile` ignores validation measurements whatever it is
   given (tested).
3. **The validated profile must be the one the comparison uses.** Gate evidence counts as
   calibrated (E2) only if validation passed *and* the architecture comparison ran on the
   validated profile.
4. **Calibration validates the model of the existing machine, nothing else.** Performance,
   power and price of compute-memory modules remain estimates with ranges
   (`configs/design_estimates.json`).
5. **A pass covers only the regimes that were validated (v0.6).** Each successful validation
   observation is tagged with the regime it exercised: dense or MoE, CPU only, split layers, all
   layers on the GPU, experts on the CPU (`--n-cpu-moe`), long context, decode or prefill, and
   whether energy was measured (`basp.calibration.observed_coverage`). A candidate states the regimes
   its gate regime needs (`required_regimes`: a model that does not fit the GPU needs split or
   experts-on-CPU observations, a long context needs long-context observations, an energy criterion
   needs energy measurements). A criterion whose required regimes have fewer than
   `min_validation_per_regime` successful observations (`configs/hardware_gate.json`, 2) is *not
   evaluated*, however small the pooled validation error: eight dense CPU measurements cannot
   validate a prediction about a MoE model with experts on the CPU. Failed runs (e.g. out of GPU
   memory) count for nothing.

## The plan

| | Calibration (tunes 4 parameters) | Validation (judges) |
|---|---|---|
| i9 + 3080 Ti | Qwen3-4B and 8B, all layers on CPU / all on GPU | 8B partial offload (12, 24 layers) and at 4096 tokens of context; 14B CPU, partial, full GPU; 30B-A3B CPU, 12 and 24 layers, and all layers on GPU with the experts of the first 24/32/40/48 layers on the CPU (`--n-cpu-moe`) |
| M2 | the same four | 8B partial offload and at 4096 context; 4B at 4096 context; 14B CPU, partial, full GPU |
| Colab T4 | the same four | 8B partial offload and at 4096 context; 4B at 4096 context; 14B partial and full GPU |

Each llama-bench run gives a prompt-processing and a generation measurement, so the desktop plan
yields 8 calibration and 28 validation measurements. The calibration runs pin down CPU and GPU
memory bandwidth (generation) and arithmetic rate (prompt processing) separately; everything in
the validation column exercises behaviour the fit never saw: split execution, PCIe transfers,
attention over long contexts, expert routing.

## Running it

Prerequisites on each machine: this repository (`pip install -e ".[dev]"`), llama.cpp built with
CUDA (Metal on the Mac), and the plan's GGUF files in one directory (any Q4_K_M GGUF of the same
models works; the header is checked):

| Plan model | File | Source (e.g.) |
|---|---|---|
| `qwen3-4b` | `Qwen3-4B-Q4_K_M.gguf` | `Qwen/Qwen3-4B-GGUF` on Hugging Face |
| `qwen3-8b` | `Qwen3-8B-Q4_K_M.gguf` | `Qwen/Qwen3-8B-GGUF` |
| `qwen3-14b` | `Qwen3-14B-Q4_K_M.gguf` | `Qwen/Qwen3-14B-GGUF` |
| `qwen3-30b-a3b` | `Qwen3-30B-A3B-Q4_K_M.gguf` | `Qwen/Qwen3-30B-A3B-GGUF` (desktop only) |

These names and repositories exist on Hugging Face (checked); their bits per weight are 4.95,
4.90, 4.87 and 4.86. Any other copy of the same model also works, renamed or passed with
`--model NAME=PATH`.

```bash
python -m benchmarks.run_calibration --machine i9-11900kf-3080ti --dry-run     # list the runs
python -m benchmarks.run_calibration --machine i9-11900kf-3080ti \
    --models-dir ~/models --llama-bench ~/llama.cpp/build/bin/llama-bench
```

It takes roughly half an hour on the desktop (CPU-only runs of the larger models dominate). For
each model it prints the file's parameter count, exact bits per weight and quantisation type,
refusing a file whose shape does not match the simulator's model. It then runs every planned
configuration, saving after each one. A configuration that fails, typically because it does not
fit GPU memory, is recorded in `runs.json` and the plan continues. Run the same command again to
resume after an interruption: completed configurations are skipped.

Output, in `evidence/<machine>/`:

| File | Contents |
|---|---|
| `measurements.json` | every measurement, with its split, the llama.cpp build, the GGUF header fingerprint and the plan hash |
| `runs.json` | every configuration: command, status, failure output |
| `environment.json` | CPU, GPU, driver, memory, llama-bench version, repository state |
| `identity.json` | SHA-256 of the llama-bench executable and of every model file, and of the plan |
| `calibration.json` | scores of the prior and of the fitted profile, fitted scales, and the fingerprints of both profiles |
| `profile_fitted.json` | the profile fitted on the calibration split |

The last line says `PASS`, `FAIL` or `NOT EVALUATED` for the validation split.

How measurements are judged (v0.4):

* **Repeats count once.** Several runs of the same benchmark (same machine, file, offload,
  threads, model, phase, context and batch) are merged into one data point, their median; eight
  runs of one workload are one validation benchmark, not eight.
* **Each machine on its own.** The verdict is computed per machine; the combined status fails if
  any machine fails and passes only if every machine passes. A machine with too few validation
  benchmarks cannot borrow another machine's.
* **Changed executable or weights.** Each model file is hashed in full (SHA-256), as is the
  llama-bench executable; both are recorded with every measurement and in `identity.json`. A
  rebuilt llama.cpp or a different model file makes earlier measurements stale: the runner runs
  those configurations again, and scoring excludes the stale ones.
* **The validated profile is the fingerprinted one.** `calibration.json` records the fingerprint
  of the prior and of the fitted profile. A profile counts as validated only while its content
  (comments excluded) equals one of them: editing a profile after validation un-validates it.
* **Coverage.** The report lists which execution regimes each machine's validation split covers
  (CPU only, split layers, GPU only, experts on the CPU, long context, dense, MoE); the simulator
  is trusted only there. The M2 and Colab plans do not cover MoE execution.

- **PASS:** rerun with `--update-profile` (or `--fit-only --update-profile`) to write the fitted
  profile into `configs/`, then regenerate the comparison on it:

  ```bash
  python -m benchmarks.explore --base-profile i9_11900kf_rtx3080ti --budget 2000 --workers 4
  python -m benchmarks.run_report --base-profile i9_11900kf_rtx3080ti
  ```

- **FAIL:** the simulator's model of a conventional system is wrong somewhere. `calibration.json`
  lists every measurement against its prediction; the pattern says where to look (partial offload:
  the transfer model; long context: attention; `--n-cpu-moe`: expert execution on the CPU). Fix the
  model, and test the fix on new validation benchmarks (a new plan version). Do not commit a
  fitted profile that failed validation as the machine's profile.

Commit `evidence/<machine>/` (and the profile if updated) with the verdict in the commit message.
`run_report` reads every `evidence/*/measurements.json` by default.

## On Google Colab

[`notebooks/calibrate_colab.ipynb`](../notebooks/calibrate_colab.ipynb) runs the `colab-t4` plan on a
free T4 runtime: open it in Colab (the badge in its first cell), choose *Runtime -> Change runtime
type -> T4 GPU*, then *Run all*. It builds llama.cpp with CUDA, downloads the three models, micro-profiles
the hardware with PyTorch (`benchmarks.profile_hardware`, which seeds the prior), runs the plan, fits and
validates, prints every measurement against its prediction, and downloads `evidence/colab-t4/` as a zip
(or pushes it, if you set `PUSH = True` and give a token). About an hour, mostly CPU-only runs on two
vCPUs. Commit the evidence as for the other machines.

## Optional measurements

```bash
python -m benchmarks.profile_hardware --machine i9-11900kf-3080ti --out evidence/i9-11900kf-3080ti/hw.json \
    --template i9_11900kf_rtx3080ti --host-device host --device gpu --link pcie-gpu --require-gpu
```

Micro-profiles memory bandwidth, and with PyTorch the PCIe transfer fit, kernel launch overhead and
a GPU roofline. Its fitted values can seed the profile before the plan runs; the fit then only moves
a parameter as far as the measurements require.

```bash
python -m benchmarks.record_routing_trace --prompt-file prompt.txt --tokens 256 --out traces/qwen3_chat
python -m benchmarks.profile_compression /models/qwen3-30b-a3b/model-00001-of-000NN.safetensors \
    --max-tensors 12 --out evidence/compression.json
python -m benchmarks.explore --trace traces/qwen3_chat.decode.json --compression evidence/compression.json
python -m benchmarks.run_report --trace traces/qwen3_chat.decode.json --compression evidence/compression.json
```

The first records Qwen3-30B-A3B's real expert routing (needs PyTorch, transformers and about 61 GB
of memory for the FP16 model with `--device-map auto`); it also counts how often its recomputed
top-k disagrees with the model's own selection. The second measures what lossless coding of INT4
codes achieves on real weights. Passing both to the exploration and the report replaces synthetic
routing and the assumed compression ratio; the report states which were used.

## What calibration cannot establish

- Anything about compute-memory modules: their arithmetic rate, memory efficiency, idle and active
  power, link latency and price are estimates, evaluated at three corners. The gate uses the
  pessimistic corner (slow, expensive modules; fast, cheap conventional components).
- The cost of a production scheduler. The gate reports the per-decision budget instead.
- Model quality or numerical correctness of anything but the software that was measured. The
  physical runtime's parity with transformers (`benchmarks.verify_runtime`) is measured on the model
  it ran, and qualifies nothing about other models.
