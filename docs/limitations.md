# Limitations of BASP Dataflow v0.6

This release is a Python research system: a simulator, a scheduler, reference kernels, a
design-space explorer, calibration tooling, an evidence-gated decision decided from receipts,
physical memory-domain management, and a CPU runtime that executes real GGUF models. It contains
**no calibration measurements of the target machines**. Every performance number about
architectures means "what the model concludes under the stated assumptions", and the gate's
evidence column says which level (E0–E3) each claim has. The only measurements (E3) are the
runtime's numerical parity with transformers, on the machine that ran the verification.

## Repairs and the measurement protocol (new in v0.6)

* **The protocol is proposed, not frozen, and no decision study has run.** Only exploratory
  studies exist (tuning on the six development prompts; a feasibility study and timing pilot of the
  30B model, if listed in report section 11). They cannot support any acceleration claim. Freezing
  the protocol and running the decision study on the held-out prompts is for the target machine.
* **No native-engine arm has ever run.** Arms C, D and E (stock llama.cpp, tuned llama.cpp, and
  BASP's change inside it) are implemented against an OpenAI-compatible `/v1/completions` server,
  tested only against a stand-in server. The competitive comparison (D vs E) has no data.
* **One 4-core machine, CPU only.** Every study here ran on the development container: no GPU, a
  shared host whose load the study cannot control beyond pairing and counterbalancing, and an OS
  page cache the 30B model (18.6 GB) does not fit in.
* **The expert ceiling bounds decoded expert arrays, not process memory.** Memory-mapped model
  pages, Python objects and the KV cache come on top; the report shows the process's peak resident
  memory separately. "Decoded bytes on misses" counts what the loader decoded, which may have come
  from the page cache; only `/proc/self/io` read bytes are storage reads.
* **Coverage tags are coarse.** A regime is dense/MoE x placement (CPU only, split layers, all on
  GPU, experts on CPU) x long context x decode/prefill x energy. Two observations per required tag
  is a minimum, not evidence of accuracy across, say, all `--n-cpu-moe` values.
* **The lowering reconciliation is a consistency check.** It compares the simulator's lowered
  graph with an independent dimensional count written from the same model shapes: it catches
  omissions and double counts (it found the missing embedding, final norm and shared experts), not
  a misunderstanding of the architecture shared by both.
* **Earlier evidence is not applicable.** v0.5 speculative measurements carry no complete identity
  and are shown NOT APPLICABLE; their numbers are kept as records only.

## Speculative execution (new in v0.5)

* **Measured on one 4-core CPU with this runtime only.** Speed-ups are against this runtime's own
  plain decoding, not against llama.cpp, KTransformers or any optimised engine; those comparisons
  need the engines on the same machine (v0.6). The runtime dequantises weights in NumPy, so on the
  30B MoE model its time is dominated by dequantisation and disk reads — which is why fewer expert
  reads show up as wall-clock gains there, and why the absolute speed (a fraction of a token per
  second) means nothing for deployment.
* **The 30B wall-clock comparison is order-dependent.** Plain decoding ran first, with a colder
  disk cache, than the speculative runs. Expert-read counts and acceptance rates do not have this
  problem; the wall-clock ratio does.
* **Few prompts, short runs.** Three prompts of 64 tokens (dense) and one prompt of 160 / 48 tokens
  (30B). Acceptance rates depend strongly on the text (code is far easier to draft than prose).
* **Exactness.** Greedy speculative output equalled plain greedy output in every measured run,
  but batched and single-row float32 arithmetic are not bit-identical, so a near-tied choice could
  flip in principle. Sampling exactness is tested statistically on a small model.
* **Grouped reads are not free compute.** Verifying `k + 1` rows multiplies every weight by more
  rows; with NumPy's BLAS that costs about four single-row products for two to nine rows, which is
  why dense speculation loses on the NumPy backend; the PyTorch backend amortises it better.
* **Rejected proposals cost memory traffic.** On the MoE model a poor drafter increased expert
  reads per token (prompt lookup: 0.64x). Speculation must be switched off when acceptance is low;
  the adaptive controller does this from measurements, but its cost model is simple (linear in rows).
* **Expert prediction** is evaluated on 159 decoded tokens of one prompt; predicted experts are not
  yet prefetched in the runtime.

## The physical runtime (new in v0.4)

* **CPU only, float32 (NumPy, or PyTorch's CPU kernels when installed).** Dense weights are
  dequantised to float32 when a model is loaded (about four bytes per dense parameter); experts of
  MoE models are dequantised on demand and cached under a byte limit, and token embeddings can be
  read row by row, so Qwen3-30B-A3B runs in about 10 GB. Its speed (13 tokens/s for Qwen3-0.6B,
  0.17 for Qwen3-30B-A3B on four cores) says nothing about BASP or about tuned engines such as
  llama.cpp.
* **No GPU execution.** The CUDA, Metal, GGML and remote backends are declared and report
  themselves unavailable; any plan that needs them fails before running. Nothing here is GPU
  inference.
* **Qwen3 dense and Qwen3-MoE only**, without RoPE scaling. Other architectures, unknown tensors
  (biases, extra norms) and quantisation formats without a dequantiser (Q2_K, Q3_K, IQ*, TQ*)
  are refused, never approximated.
* **No tokenizer.** The runtime takes token ids; the verification tool uses the model's Hugging
  Face tokenizer.
* **Verification coverage.** A real model was verified for the dense path (Qwen3-0.6B, Q8_0):
  logits, 27 layer outputs, 16 greedy tokens and perplexity on a 61-token text. The MoE path
  (router, expert selection, expert execution) is verified against transformers on small
  random-weight models only (in CI). The real Qwen3-30B-A3B runs and produces coherent text with a
  plausible perplexity, but no reference small enough to compare its logits exists here. The
  perplexity text is far too short to say anything about model quality.
* **Receipts are bound to the model they measured.** The parity receipts concern Qwen3-0.6B and
  the NumPy runtime; they do not qualify the gate's numerical or quality criteria for the 30B
  regime, and the report says so.

## Memory domains and expert caching (new in v0.4)

* The memory-domain manager enforces the budget invariant and eviction safety in the runtime and
  in tests; the simulator still uses its own allocator model, so the two are not yet one system.
* Expert-cache policies are compared on synthetic routing with costs from a profile (E1), on the
  critical path only (no overlap of migration with computation, for any policy). The cost-aware
  policy's horizon is a parameter; results depend on it and on the trace.
* Prefetching exists as a component (budgeted, reports waste) but is not used by the runtime.

## Nothing is calibrated yet

* **No measurement has been taken.** The calibration runner, the plan and the fitting are tested,
  including end to end against a fake llama-bench, but the parts that touch real hardware
  (running llama.cpp, PyTorch GPU profiling, hooking a real model's routers, reading
  `.safetensors`) have never been executed. Their external formats were checked against the
  tools' sources (llama-bench in llama-cpp-python 0.3.38, `gguf` 0.19, transformers 5.24). The
  real model files were checked: the plan's Qwen3 GGUF files exist, and their headers match the
  simulator's shapes to the parameter.
* **Calibration is per machine and fingerprinted (v0.4).** Repeated runs of one benchmark count
  once; each machine is judged on its own validation split; measurements made with an executable
  or model file that has since changed are excluded; a profile counts as validated only while its
  content equals the profile the calibration run recorded. The Colab and M2 plans do not validate
  every regime (coverage is listed in the report).
* **The fitting procedure has only seen data generated by the simulator itself.** That shows it
  recovers the parameters of a machine that behaves like the model, not that a real machine
  does. If validation fails, the model of a conventional system needs work before any comparison
  means anything.
* **The baseline machine is illustrative.** The architecture comparison runs on
  `configs/desktop_gpu12.json`. Its CPU arithmetic rate was raised in v0.3 from 80 to 400 GMAC/s
  (v0.1 had 30) so that prompt processing is compute-bound at a plausible rate and generation
  stays memory-bound; each earlier value handicapped the conventional system. These are still
  judgements from public figures.

## Estimates for hardware that does not exist

* **Compute-memory modules are hypothetical.** Arithmetic rate, usable memory bandwidth, launch
  and link latencies, idle, sleep and active power are (pessimistic, nominal, optimistic) triples
  chosen by judgement in `configs/design_estimates.json`. Calibrating on a CPU and a GPU narrows
  none of them.
* **Corners are named from the module proposal's side.** The pessimistic corner makes modules
  slow and expensive *and* conventional components (host, GPU) fast and cheap. v0.2 priced shared
  components high at the pessimistic corner, which favoured modules.
* **Prices are guesses** for ratios only: no quotes, no volume, no development cost.
* **GPU classes are scale factors** relative to the base profile's GPU (12, 16, 24, 32, 48 GiB),
  from spec-sheet memory bandwidth and rough compute ratios. Multi-GPU systems are not modelled,
  so the conventional side of every comparison is still weaker than what a buyer could assemble.
* **The local fabric of architecture H** (a switch between GPU and modules) has not been
  designed; its bandwidth and latency are estimates.
* **Corners are not confidence intervals.**

## Workload assumptions

* **Routing is synthetic unless a recorded trace is given** (`--trace`). The report states which
  was used. A recorded single-stream trace is cut into a profiling part and disjoint windows for
  batched rows.
* **Co-activation-aware placement is regularised towards load balancing** because short traces
  cannot estimate pairwise statistics.
* **Model shapes, not models, in the simulator.** The simulator executes no real weights (the
  physical runtime does, separately). GGUF quantisation formats are not the BASP INT4 format.
* **Compression ratios are assumed** unless a measured profile is given (`--compression`).

## Modelling simplifications

* Memory is a bandwidth model: no row-buffer behaviour, refresh, NUMA or cache hierarchy.
* Unified memory (M2) is one bandwidth resource and one capacity pool shared by CPU and GPU.
  Not modelled: macOS's limit on the GPU working set, and zero-copy sharing (a transfer between
  the two devices is charged as a copy over the shared bus).
* A prompt is one sequence processed as a batch: rows share one KV cache and attend on average to
  half the prompt. Batched KV-cache reads are counted once per prompt, which is optimistic for
  attention without tiling.
* A device runs one kernel at a time at its roofline rate; batching is a reduced launch cost.
* A CPU has one arithmetic rate for matrix-vector and matrix-matrix work; generation is
  memory-bound at that rate, prompt processing compute-bound.
* Link sharing is fluid; the estimator approximates it with gap-aware booking.
* Attention is cost-modelled but not computed numerically.
* Power gating has one sleep state; there is no DVFS or thermal model.
* The scheduler takes no simulated time; its overhead is reported as a per-decision budget.
* The numeric co-simulation carries expert outputs in FP32 while the traffic model charges FP16
  sizes for them: one of the two is optimistic.
* Results are equal within rounding, not bit for bit, across dataflows that reorder FP32
  additions (device-local partial sums, tensor-parallel experts); routing was never affected in
  60 random models, but a reordered sum could in principle tip a near-tied routing decision.

## Design-space exploration

* The space covers architecture, GPU class (none, 12, 24 or 32 GiB), module count, module memory,
  channels, MAC lanes, link bandwidth and expert replication, for one workload (30B MoE,
  single-user decoding) at the nominal corner; front designs are re-evaluated at the other corners.
* The evolutionary search evaluates about a fifth of the feasible designs, seeded with every
  conventional system and one standard design per architecture.
* A stored exploration records its provenance; the report refuses to draw conclusions from one
  produced by different simulator code, profiles, estimates or traces.

## Not implemented

| Item | Status |
|---|---|
| Native CPU (C++/SIMD), CUDA, Metal and GGML backends | Declared, unavailable; deferred to v0.6 (a PyTorch CPU backend exists) |
| Comparison with llama.cpp / KTransformers on the same machine | Not done; required before any speed claim against existing engines |
| Mixed-precision (sensitivity-based) quantisation with quality tests | Not started; deferred rather than guessed |
| Architecture I (adaptive hierarchical execution, cost `T + λE + μR + γU`) | Deferred; not simulated |
| Real MoE model verified against a reference | Qwen3-30B-A3B runs (coherent output, perplexity 1.68 on a check text), but no reference fits in memory to compare logits |
| `hardware/*.sv`, FPGA | Not started, by decision: blocked on the gate |
| Multi-GPU conventional systems | Not modelled |
| INT2 / ternary / sparse / entropy kernels | Modelled for traffic and decode cost only |
| Hardware-aware routing | Out of scope: the runtime never changes which experts are selected |

## Known rough edges

* `Command` descriptors carry at most 8 tensor ids; longer operand lists live in the kernel
  descriptor.
* Background replication and prefetch share one budget per link direction.
* The quick report (`--quick`) uses short runs; its numbers differ from the full report's.
* Generated files are byte-identical only on the same Python and NumPy versions; across Python
  3.10 and 3.13 simulated figures differ in the last digit.
* `reports/v0.1/` to `reports/v0.4/` were produced by earlier models and cannot be regenerated
  from the current code; they are kept as records.
