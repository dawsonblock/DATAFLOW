"""Runtime-versus-reference parity on one GGUF model: logits, layer outputs, routing,
generation and perplexity, with a verdict and evidence receipts.

Targets (``configs/hardware_gate.json``): relative L2 error of logits at most
``numeric_rel_l2_max``; identical shapes; identical expert selections; identical greedy
tokens; perplexity at most ``model_quality_relative_perplexity_max`` times the reference's.
"""

from __future__ import annotations

import hashlib
import os
import platform
import time
from typing import Any

import numpy as np

from basp.evidence import PACKAGE_DIR, EvidenceLevel, Receipt, file_identity, sha256_json

from .generation import compare_generation, reference_greedy
from .intermediate_tensors import compare_hidden
from .logits import compare_logits
from .perplexity import perplexity
from .reference_runtime import load_reference, reference_forward
from .routing import compare_routing

RUNTIME_SOURCES = ("runtime", "backends/execution.py", "backends/cpu/backend.py", "backends/cpu/kernels.py",
                   "backends/torch_cpu",
                   "models/quants.py", "models/gguf.py", "models/gguf_tensors.py", "memory/domains.py")  # fmt: skip


def runtime_fingerprint() -> str:
    """Hash of the code that executes a model (runtime, CPU backend, GGUF reading)."""
    h = hashlib.sha256()
    for entry in RUNTIME_SOURCES:
        path = PACKAGE_DIR / entry
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for f in files:
            h.update(f.relative_to(PACKAGE_DIR).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def machine_identity() -> str:
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), cpu)
    except OSError:
        pass
    return f"{platform.system()} {platform.machine()} {cpu} ({os.cpu_count()} logical CPUs)".strip()


def run_parity(path: str | os.PathLike, prompt: list[int], *, model_name: str, eval_tokens: list[int] | None = None,
               n_generate: int = 16, thresholds: dict[str, float] | None = None,
               prefer: tuple[str, ...] = ("cpu",)) -> dict[str, Any]:  # fmt: skip
    from basp.experiments.profiles import load_json
    from basp.runtime import Engine, generate

    gate = thresholds or load_json("hardware_gate")
    max_rel = float(gate["numeric_rel_l2_max"])
    max_ppl = float(gate["model_quality_relative_perplexity_max"])
    start = time.perf_counter()
    engine = Engine(path, prefer=prefer)
    reference, config = load_reference(path)
    result: dict[str, Any] = {"model": model_name, "file": file_identity(path), "config": config.__dict__,
                              "prompt_tokens": len(prompt), "thresholds": {"rel_l2_max": max_rel,
                                                                            "relative_perplexity_max": max_ppl}}  # fmt: skip

    # 1. prompt: every row's logits, every layer's output, every routing decision
    with engine.session(len(prompt) + n_generate + 1) as s:
        out = s.prefill(prompt, all_logits=True, capture=True)
    ref_logits, ref_hidden, ref_routing = reference_forward(reference, prompt, capture_routing=config.moe)
    result["shapes_match"] = out.logits.shape == ref_logits.shape
    result["prefill_logits"] = compare_logits(out.logits, ref_logits)
    result["hidden"] = compare_hidden(out.hidden, ref_hidden)
    result["routing"] = compare_routing(out.experts, ref_routing) if config.moe else None

    # 2. greedy generation through the KV cache, against the reference recomputing everything
    with engine.session(len(prompt) + n_generate + 1) as s:
        gen = generate(s, prompt, n_generate, keep_logits=True)
    ref_tokens, ref_rows = reference_greedy(reference, prompt, n_generate)
    result["generation"] = compare_generation(gen.tokens, np.stack(gen.logits), ref_tokens, ref_rows)

    # 3. perplexity on held-out text, teacher forced
    if eval_tokens:
        with engine.session(len(eval_tokens) + 1) as s:
            rt = s.prefill(eval_tokens, all_logits=True).logits
        ref_eval, _, _ = reference_forward(reference, eval_tokens)
        ppl_rt, ppl_ref = perplexity(rt, np.asarray(eval_tokens)), perplexity(ref_eval, np.asarray(eval_tokens))
        result["perplexity"] = {"tokens": len(eval_tokens), "runtime": ppl_rt, "reference": ppl_ref,
                                "relative": ppl_rt / ppl_ref, "logits": compare_logits(rt, ref_eval)}  # fmt: skip

    # verdict
    reasons = []
    worst = max(result["prefill_logits"]["rel_l2_max"], result["generation"]["decode_logits"]["rel_l2_max"])
    result["numerical_error"] = worst
    if not result["shapes_match"] or not result["hidden"]["shapes_match"]:
        reasons.append("shapes differ")
    if not worst <= max_rel:
        reasons.append(f"logit relative L2 error {worst:.2e} exceeds {max_rel:g}")
    if result["routing"] is not None and not result["routing"]["exact"]:
        reasons.append(f"{result['routing']['mismatches']} expert selections differ")
    if not result["generation"]["identical"]:
        reasons.append(f"greedy tokens diverge at step {result['generation']['first_divergence']}")
    if "perplexity" in result and not result["perplexity"]["relative"] <= max_ppl:
        reasons.append(f"relative perplexity {result['perplexity']['relative']:.4f} exceeds {max_ppl:g}")
    result["verdict"] = "pass" if not reasons else "fail"
    result["reasons"] = reasons
    result["seconds"] = time.perf_counter() - start
    result["telemetry"] = engine.telemetry.to_json()
    result["runtime_fingerprint"] = runtime_fingerprint()
    result["backend"] = sorted(set(engine.plan.assignment.values()))
    result["receipts"] = [r.to_json() for r in parity_receipts(result, model_name=model_name)]
    return result


def parity_receipts(result: dict[str, Any], *, model_name: str) -> list[Receipt]:
    """E3 receipts of what was measured.  They are about the NumPy runtime on this model
    (candidate ``runtime:numpy-cpu``): no BASP architecture or regime is qualified by them."""
    import torch
    import transformers

    bindings = {"machine": machine_identity(), "model": model_name, "weights": result["file"]["sha256"],
                "runtime": result["runtime_fingerprint"],
                "benchmark": sha256_json({"prompt_tokens": result["prompt_tokens"],
                                          "generated": result["generation"]["tokens"],
                                          "perplexity_tokens": (result.get("perplexity") or {}).get("tokens"),
                                          "reference": f"transformers {transformers.__version__}, "
                                                       f"torch {torch.__version__}, float32"})}  # fmt: skip
    regime = f"{model_name}, float32 CPU execution, parity with the transformers reference"
    candidate = "runtime:" + "+".join(result.get("backend", ["numpy-cpu"]))
    out = [Receipt("numerical_error", float(result["numerical_error"]), EvidenceLevel.E3, candidate,
                   regime, "transformers", bindings, note="max relative L2 of logits, prompt and decode")]  # fmt: skip
    if "perplexity" in result:
        out.append(Receipt("model_quality", float(result["perplexity"]["relative"]), EvidenceLevel.E3,
                           candidate, regime, "transformers", bindings,
                           note="perplexity relative to the reference on the same weights"))  # fmt: skip
    return out
