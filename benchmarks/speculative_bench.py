"""Measure speculative execution and grouped expert reads on real GGUF models (this machine).

    python -m benchmarks.speculative_bench dense --target Qwen3-1.7B-Q8_0.gguf \
        --draft Qwen3-0.6B-Q8_0.gguf --tokenizer <HF tokenizer dir> --out evidence/speculative/dense.json
    python -m benchmarks.speculative_bench moe --target Qwen3-30B-A3B-Q4_K_M.gguf \
        --draft Qwen3-0.6B-Q8_0.gguf --tokenizer <dir> --out evidence/speculative/qwen3-30b-a3b.json \
        --trace evidence/traces/qwen3-30b-a3b

``dense``: wall-clock tokens per second of ordinary greedy decoding against speculative decoding
(draft model with fixed and adaptive draft length; prompt lookup) on the same runtime and machine,
and whether the tokens are identical.

``moe``: on a mixture-of-experts target, expert reads per generated token for ordinary decoding
and for speculative verification (every expert chosen by any verified token read once), the
draft model's acceptance rate, the expert-union curve of the model's real routing (experts per
token if ``g`` consecutive tokens are processed together), expert-prediction accuracy, and a
routing trace in the simulator's format.

Everything is measured on this machine with the NumPy float32 runtime (weights dequantised on
use): wall-clock times describe this runtime, not an optimised engine; counts of expert reads and
acceptance rates describe the model and the method.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROMPTS = {
    "prose": "The printing press changed Europe because",
    "code": "def merge_sort(items):\n    \"\"\"Sort a list with merge sort.\"\"\"\n",
    "explain": "Explain step by step how photosynthesis turns light into chemical energy.",
}
EVAL_TEXT = (
    "Python is an easy to learn, powerful programming language. It has efficient high-level data "
    "structures and a simple but effective approach to object-oriented programming. Python's elegant "
    "syntax and dynamic typing, together with its interpreted nature, make it an ideal language for "
    "scripting and rapid application development in many areas on most platforms."
)


def _environment() -> dict[str, Any]:
    from basp.verification.parity_report import machine_identity, runtime_fingerprint

    return {"machine": machine_identity(), "python": platform.python_version(), "numpy": np.__version__,
            "runtime_fingerprint": runtime_fingerprint()}  # fmt: skip


def sequential(engine, prompt: list[int], n: int, *, record: bool = False) -> dict[str, Any]:
    tel = engine.telemetry
    before = (tel.expert_requests, tel.expert_bytes)
    routing: list[tuple[list, list]] = []
    tokens: list[int] = []
    start = time.perf_counter()
    with engine.session(len(prompt) + n + 2) as s:
        out = s.prefill(prompt)
        prefill_s = time.perf_counter() - start
        prefill_routing = [(e.tolist(), w.tolist()) for e, w in zip(out.experts, out.expert_weights)]
        logits = out.logits[-1]
        after_prefill = (tel.expert_requests, tel.expert_bytes)
        for i in range(n):
            token = int(np.argmax(logits))
            tokens.append(token)
            if i + 1 == n:
                break
            out = s.step(token)
            if record:
                routing.append([(e[0].tolist(), w[0].tolist()) for e, w in zip(out.experts, out.expert_weights)])
            logits = out.logits[-1]
    total = time.perf_counter() - start
    decode_requests = tel.expert_requests - after_prefill[0]
    return {"tokens": tokens, "seconds": total, "prefill_seconds": prefill_s, "decode_steps": n - 1,
            "decode_tokens_per_s": (n - 1) / (total - prefill_s) if n > 1 else 0.0,
            "tokens_per_s": n / total, "expert_requests": tel.expert_requests - before[0],
            "expert_bytes": tel.expert_bytes - before[1],
            "decode_expert_requests_per_token": decode_requests / max(n - 1, 1),
            "decode_expert_bytes_per_token": (tel.expert_bytes - after_prefill[1]) / max(n - 1, 1),
            "routing": routing, "prefill_routing": prefill_routing}  # fmt: skip


def speculative(engine, drafter, prompt: list[int], n: int, **kw) -> dict[str, Any]:
    from basp.runtime.speculative import speculative_generate

    with engine.session(len(prompt) + n + 16) as s:
        r = speculative_generate(s, drafter, prompt, n, **kw)
    # the first round also processes the prompt: report decode-only expert reads separately
    later = r.rounds[1:]
    later_tokens = sum(x.emitted for x in later)
    return {**r.summary(), "token_ids": r.tokens,
            "decode_expert_requests_per_token": sum(x.expert_requests for x in later) / max(later_tokens, 1),
            "decode_expert_bytes_per_token": sum(x.expert_bytes for x in later) / max(later_tokens, 1),
            "rows_per_call": float(np.mean([x.rows for x in later])) if later else 0.0}  # fmt: skip


def _same(a: list[int], b: list[int]) -> dict[str, Any]:
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    return {"identical": first is None and len(a) == len(b), "first_divergence": first}


def run_dense(args) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from basp.runtime import Engine
    from basp.runtime.speculative import AdaptiveController, ModelDrafter, NgramDrafter

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    target, draft = Engine(args.target, prefer=args.prefer), Engine(args.draft, prefer=args.prefer)
    warm = tok("Hello")["input_ids"]
    sequential(target, warm, 4)  # first-call effects out of the timings
    out: dict[str, Any] = {"target": Path(args.target).name, "draft": Path(args.draft).name, "tokens": args.tokens,
                           "backend": sorted(set(target.plan.assignment.values())), "prompts": {}}  # fmt: skip
    for name, text in PROMPTS.items():
        prompt = tok(text)["input_ids"]
        base = sequential(target, prompt, args.tokens)
        rows = {"sequential": {k: v for k, v in base.items() if k not in ("routing", "prefill_routing")}}
        configs = [(f"draft model, k={k}", "model", {"k": k}) for k in (2, 4)]
        configs += [("draft model, adaptive k", "model", {"controller": AdaptiveController(k_max=8)}),
                    ("prompt lookup, k=4", "lookup", {"k": 4})]  # fmt: skip
        for label, kind, options in configs:
            drafter = ModelDrafter(draft, 512) if kind == "model" else NgramDrafter()
            r = speculative(target, drafter, prompt, args.tokens, **options)
            if hasattr(drafter, "close"):
                drafter.close()
            r.update(_same(r["token_ids"], base["tokens"]), speedup=base["seconds"] / r["seconds"])
            rows[label] = r
            print(f"{name:8s} {label:26s} {r['tokens_per_s']:6.2f} tok/s  speedup {r['speedup']:.2f}x  "
                  f"acceptance {r['acceptance']:.2f}  identical {r['identical']}", flush=True)
        print(f"{name:8s} {'sequential':26s} {base['tokens_per_s']:6.2f} tok/s  text {tok.decode(base['tokens'])!r}")
        rows["text"] = tok.decode(base["tokens"])
        out["prompts"][name] = rows
    return out


def union_curve(routing: list[list[tuple[list, list]]], sizes=(1, 2, 3, 4, 5, 6, 8)) -> dict[int, float]:
    """Distinct experts per layer per token when ``g`` consecutive tokens are processed together."""
    n_layers = len(routing[0])
    out = {}
    for g in sizes:
        windows = [routing[i:i + g] for i in range(0, len(routing) - g + 1, g)]
        per_token = [np.mean([len({e for step in w for e in step[layer][0]}) for layer in range(n_layers)]) / g
                     for w in windows]  # fmt: skip
        out[g] = float(np.mean(per_token))
    return out


def predictor_accuracy(trace, n_values=(8, 16, 32)) -> dict[str, dict[str, float]]:
    from basp.scheduler.predictor import ExpertPredictor

    out = {}
    for label, kw in (("frequency", {"use_recency": False, "use_transitions": False}),
                      ("recency + frequency", {"use_transitions": False}),
                      ("recency + frequency + transitions", {})):  # fmt: skip
        out[label] = {str(n): ExpertPredictor(trace.n_layers, trace.n_experts, **kw).evaluate(trace, n)
                      for n in n_values}  # fmt: skip
    return out


def run_moe(args) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from basp.runtime import Engine
    from basp.runtime.speculative import ModelDrafter, NgramDrafter
    from basp.verification import perplexity
    from basp.workloads.trace import RoutingStep, RoutingTrace

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    t0 = time.perf_counter()
    target = Engine(args.target, embedding="rows", expert_cache_bytes=args.expert_cache_gib << 30)
    load_s = time.perf_counter() - t0
    cfg = target.config
    print(f"loaded {Path(args.target).name} in {load_s:.0f} s: {cfg.n_layers} layers, {cfg.n_experts} experts, "
          f"top-{cfg.top_k}", flush=True)  # fmt: skip
    out: dict[str, Any] = {"target": Path(args.target).name, "draft": Path(args.draft).name, "load_seconds": load_s,
                           "config": cfg.__dict__,
                           "expert_stored_bytes_mean": float(np.mean(target.weights._expert_stored))}  # fmt: skip
    # 1. sanity: perplexity on a fixed text (a broken MoE path gives nonsense)
    eval_ids = tok(EVAL_TEXT)["input_ids"]
    with target.session(len(eval_ids) + 1) as s:
        t = time.perf_counter()
        logits = s.prefill(eval_ids, all_logits=True).logits
        out["perplexity"] = {"tokens": len(eval_ids), "value": perplexity(logits, np.asarray(eval_ids)),
                             "seconds": time.perf_counter() - t}  # fmt: skip
    print(f"perplexity {out['perplexity']['value']:.2f} on {len(eval_ids)} tokens", flush=True)
    # 2. ordinary greedy decoding, recording the routing of every generated token
    prompt = tok(PROMPTS["explain"])["input_ids"]
    base = sequential(target, prompt, args.tokens, record=True)
    print(f"sequential: {args.tokens} tokens in {base['seconds']:.0f} s; "
          f"{base['decode_expert_requests_per_token']:.0f} expert reads per token; "
          f"text {tok.decode(base['tokens'])!r}", flush=True)  # fmt: skip
    routing, prefill_routing = base.pop("routing"), base.pop("prefill_routing")
    out["sequential"] = base
    out["text"] = tok.decode(base["tokens"])
    steps = [[RoutingStep((tuple(e),), (tuple(float(x) for x in np.asarray(w) / np.sum(w)),)) for e, w in layers]
             for layers in routing]  # fmt: skip
    trace = RoutingTrace(cfg.n_layers, cfg.n_experts, cfg.top_k, 1, steps)
    trace.validate()
    if args.trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        trace.save(f"{args.trace}.decode.json")
        pre = [[RoutingStep(tuple(tuple(r) for r in e), tuple(tuple(float(x) for x in np.asarray(r) / np.sum(r))
                                                                for r in w)) for e, w in prefill_routing]]  # fmt: skip
        RoutingTrace(cfg.n_layers, cfg.n_experts, cfg.top_k, len(prompt), pre).save(f"{args.trace}.prefill.json")
    out["union_curve"] = union_curve(routing)
    out["predictors"] = predictor_accuracy(trace)
    print("experts per token when g tokens share one pass:",
          {g: round(v, 2) for g, v in out["union_curve"].items()}, flush=True)
    # 3. speculative decoding: the 0.6B model drafts, the MoE target verifies in one pass
    n = min(args.spec_tokens, args.tokens)
    draft = Engine(args.draft)
    for label, drafter, k in (("draft model, k=4", ModelDrafter(draft, 1024), 4),
                              ("prompt lookup, k=4", NgramDrafter(), 4)):  # fmt: skip
        r = speculative(target, drafter, prompt, n, k=k)
        if hasattr(drafter, "close"):
            drafter.close()
        r.update(_same(r["token_ids"], base["tokens"][:n]))
        r["decode_expert_reduction"] = base["decode_expert_requests_per_token"] / max(
            r["decode_expert_requests_per_token"], 1e-9)  # fmt: skip
        out[label] = r
        print(f"{label}: acceptance {r['acceptance']:.2f}, {r['tokens_per_target_call']:.2f} tokens per target call, "
              f"{r['decode_expert_requests_per_token']:.0f} expert reads per token "
              f"({r['decode_expert_reduction']:.2f}x fewer), identical {r['identical']}, {r['seconds']:.0f} s",
              flush=True)  # fmt: skip
    return out


SYSTEM_PROMPT = (
    "You are a careful technical assistant. Answer in plain English, state assumptions explicitly, "
    "prefer short sentences, and say when you do not know something. When a question involves numbers, "
    "show the calculation. When it involves code, give a minimal working example and explain each step. "
    "Never invent references. If a request is ambiguous, answer the most likely interpretation and name "
    "the alternative. Keep answers under two hundred words unless asked for more detail. "
) * 3
QUESTIONS = ["What is a mixture-of-experts model?", "How does a KV cache work?",
             "Why is decoding memory-bound?", "What does quantisation trade away?"]  # fmt: skip


def run_prefix(args) -> dict[str, Any]:
    """Time to the first token with and without the shared-prefix cache (same system prompt)."""
    from transformers import AutoTokenizer

    from basp.runtime import Engine
    from basp.runtime.prefix_cache import PrefixCache

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    engine = Engine(args.target, prefer=args.prefer)
    cache = PrefixCache(engine, block=16)
    sequential(engine, tok("Hello")["input_ids"], 2)
    rows = []
    for q in QUESTIONS:
        tokens = tok(SYSTEM_PROMPT + "\n\nQuestion: " + q)["input_ids"]
        t = time.perf_counter()
        with engine.session(len(tokens) + 1) as s:
            full = s.prefill(tokens).logits[-1]
        cold = time.perf_counter() - t
        t = time.perf_counter()
        s, reused = cache.session(tokens, len(tokens) + 1)
        warm_logits = s.prefill(tokens[reused:]).logits[-1]
        warm = time.perf_counter() - t
        cache.store(s, tokens)
        s.close()
        rows.append({"question": q, "tokens": len(tokens), "reused": reused, "full_prefill_s": cold,
                     "with_cache_s": warm, "speedup": cold / warm,
                     "same_next_token": int(np.argmax(full)) == int(np.argmax(warm_logits)),
                     "max_abs_logit_diff": float(np.abs(full - warm_logits).max())})  # fmt: skip
        print(f"{len(tokens)} tokens, reused {reused}: {cold:.2f} s -> {warm:.2f} s", flush=True)
    return {"model": Path(args.target).name, "block": cache.block, "rows": rows,
            "stats": cache.stats.__dict__}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["dense", "moe", "prefix"])
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", help="draft model (dense, moe)")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--spec-tokens", type=int, default=48, help="moe: tokens generated speculatively")
    parser.add_argument("--expert-cache-gib", type=int, default=2, help="moe: dequantised-expert cache")
    parser.add_argument("--trace", help="moe: write <trace>.decode.json and <trace>.prefill.json")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy",
                        help="dense, prefix: execution backend (moe always uses NumPy)")  # fmt: skip
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.prefer = ("torch",) if args.backend == "torch" else ("cpu",)
    result = {"dense": run_dense, "moe": run_moe, "prefix": run_prefix}[args.mode](args)
    result["environment"] = _environment()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True, default=float) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
