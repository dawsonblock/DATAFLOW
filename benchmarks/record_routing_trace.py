"""Record the real expert-routing decisions of a mixture-of-experts model.

    python -m benchmarks.record_routing_trace --prompt-file prompt.txt --tokens 256 \
        --out traces/qwen3_chat

Loads a causal language model with Hugging Face ``transformers`` (default
``Qwen/Qwen3-30B-A3B``), hooks every router -- a module named ``gate`` that scores
``config.num_experts`` experts: an ``nn.Linear`` in transformers 4, a ``Qwen3MoeTopKRouter``
returning ``(logits, scores, indices)`` in transformers 5 (checked against the 5.24 source) --
runs the prompt and greedy generation with the KV cache,
and writes two files in the BASP ``RoutingTrace`` JSON format:

* ``<out>.prefill.json`` -- one step whose batch rows are the prompt tokens;
* ``<out>.decode.json`` -- one step per generated token after the first (batch 1).  The first
  generated token comes out of the prompt pass, so ``--tokens N`` yields ``N - 1`` decode steps.

and ``<out>.stats.json`` with popularity entropy, stickiness and concentration of each trace --
the properties the synthetic trace generator assumes.  The recorded traces replace synthetic
routing in the simulator (``RoutingTrace.load``).

NOT exercised in the development container: loading a model, the hooks and generation (no
PyTorch, no transformers, no model weights there).  Only the array-level collection logic
(:class:`RouterCollector`, :func:`find_routers` on stand-in modules) is covered by tests.  The
module layout is assumed from the Qwen MoE implementations in ``transformers``; if no router is
found the script lists candidate modules so the name can be given with ``--router-name``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from basp.calibration.routing import build_trace, trace_statistics
from basp.workloads.trace import RoutingTrace


class RouterCollector:
    """Collects router logits layer by layer and splits them into forward passes.

    Routers fire in layer order once per forward pass, so a pass ends when the first router
    fires again.  The first pass is the prompt (prefill); every later pass handles one new
    token.  Logits may arrive as ``[tokens, experts]`` or ``[batch, tokens, experts]``.
    """

    def __init__(self, n_routers: int) -> None:
        if n_routers < 1:
            raise ValueError("need at least one router")
        self.n_routers = n_routers
        self.passes: list[list[np.ndarray]] = []
        self.checked = 0  # router rows whose model-selected experts were compared
        self.disagreements = 0  # rows where the recomputed top-k differs (ties, precision)

    def record(self, router_index: int, logits: Any, selected: Any = None) -> None:
        """Record one router call; ``selected`` (the model's own top-k expert ids, if the router
        returns them) is checked against the top-k recomputed from the logits."""
        array = np.asarray(logits, dtype=np.float64)
        if array.ndim < 1:
            raise ValueError("router output must have an expert dimension")
        array = array.reshape(-1, array.shape[-1])
        if selected is not None:
            ids = np.asarray(selected).reshape(array.shape[0], -1)
            k = ids.shape[1]
            ours = np.argsort(-array, axis=1, kind="stable")[:, :k]
            self.disagreements += int(sum(set(a) != set(b) for a, b in zip(ours.tolist(), ids.tolist())))
            self.checked += array.shape[0]
        if not self.passes or len(self.passes[-1]) == self.n_routers:
            self.passes.append([])
        expected = len(self.passes[-1])
        if router_index != expected:
            raise ValueError(f"router {router_index} fired out of order (expected {expected}); "
                             "the hooked modules are not called once per layer per pass")  # fmt: skip
        self.passes[-1].append(array)

    def complete_passes(self) -> list[list[np.ndarray]]:
        return [p for p in self.passes if len(p) == self.n_routers]

    def prefill_trace(self, top_k: int) -> RoutingTrace:
        passes = self.complete_passes()
        if not passes:
            raise ValueError("no complete forward pass was recorded")
        return build_trace([passes[0]], top_k)

    def decode_trace(self, top_k: int) -> RoutingTrace | None:
        """Trace of the single-token passes after the prompt, or ``None`` if there were none."""
        passes = [p for p in self.complete_passes()[1:] if p[0].shape[0] == 1]
        return build_trace(passes, top_k) if passes else None


def router_width(module: Any) -> int | None:
    """Number of experts a router module scores: ``out_features`` of a linear layer, or the first
    dimension of the ``weight`` of a router module (transformers 5: ``Qwen3MoeTopKRouter`` holds a
    ``[num_experts, hidden]`` weight and returns ``(logits, scores, indices)``)."""
    width = getattr(module, "out_features", None)
    if width is not None:
        return int(width)
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    return int(shape[0]) if shape is not None and len(shape) == 2 else None


def router_output(output: Any) -> tuple[Any, Any | None]:
    """(logits, selected expert ids or ``None``) from a router module's forward output."""
    if isinstance(output, (tuple, list)):
        return output[0], (output[2] if len(output) >= 3 else None)
    return output, None


def find_routers(model: Any, num_experts: int, router_name: str = "gate") -> list[tuple[str, Any]]:
    """Router modules in call order: named ``router_name`` and scoring ``num_experts`` experts."""
    found, candidates = [], []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        width = router_width(module)
        if leaf == router_name and width == num_experts:
            found.append((name, module))
        elif width == num_experts or leaf in ("gate", "router"):
            candidates.append(name)
    if not found:
        hint = ", ".join(candidates[:8]) or "none"
        raise SystemExit(
            f"no router named {router_name!r} with {num_experts} outputs found in the model. "
            f"Candidate modules: {hint}. Pass the right leaf name with --router-name."
        )
    return found


def write_outputs(collector: RouterCollector, top_k: int, out: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Write the prefill and decode traces and their statistics; returns the statistics."""
    stats: dict[str, Any] = {"meta": meta}
    prefill = collector.prefill_trace(top_k)
    prefill.save(f"{out}.prefill.json")
    stats["prefill"] = trace_statistics(prefill)
    decode = collector.decode_trace(top_k)
    if decode is not None:
        decode.save(f"{out}.decode.json")
        stats["decode"] = trace_statistics(decode)
    Path(f"{out}.stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B", help="Hugging Face id or path")
    parser.add_argument("--prompt", help="prompt text")
    parser.add_argument("--prompt-file", help="file containing the prompt")
    parser.add_argument("--tokens", type=int, default=128, help="tokens to generate (greedy)")
    parser.add_argument("--out", required=True, help="output path prefix")
    parser.add_argument("--top-k", type=int, help="experts per token (default: from the config)")
    parser.add_argument("--router-name", default="gate", help="leaf name of the router modules")
    parser.add_argument("--device-map", default="auto", help="transformers device_map")
    parser.add_argument("--dtype", default="auto", help="torch dtype name, or 'auto'")
    args = parser.parse_args(argv)
    if bool(args.prompt) == bool(args.prompt_file):
        parser.error("give exactly one of --prompt and --prompt-file")
    prompt = args.prompt or Path(args.prompt_file).read_text()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"recording a trace needs PyTorch and transformers ({exc.name} is "
                         "missing): pip install torch transformers accelerate") from None  # fmt: skip

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                 device_map=args.device_map)  # fmt: skip
    model.eval()
    num_experts = getattr(model.config, "num_experts", None)
    if not num_experts:
        raise SystemExit(f"{args.model} has no config.num_experts: not a mixture-of-experts model "
                         "this recorder understands")  # fmt: skip
    top_k = args.top_k or getattr(model.config, "num_experts_per_tok", None)
    if not top_k:
        raise SystemExit("the config has no num_experts_per_tok; pass --top-k")
    routers = find_routers(model, num_experts, args.router_name)
    collector = RouterCollector(len(routers))
    hooks = [
        module.register_forward_hook(
            lambda _m, _i, output, index=index: collector.record(
                index, *(None if t is None else t.detach().float().cpu().numpy()
                         for t in router_output(output)))
        )
        for index, (_, module) in enumerate(routers)
    ]  # fmt: skip
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.tokens, do_sample=False,
                                       use_cache=True)  # fmt: skip
    finally:
        for hook in hooks:
            hook.remove()

    meta = {
        "model": args.model,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_tokens": int(generated.shape[-1] - inputs["input_ids"].shape[-1]),
        "num_experts": int(num_experts),
        "top_k": int(top_k),
        "routers": len(routers),
        "norm_topk_prob": bool(getattr(model.config, "norm_topk_prob", True)),
        "router_rows_checked": collector.checked,
        "router_rows_disagreeing": collector.disagreements,
    }
    stats = write_outputs(collector, int(top_k), args.out, meta)
    for phase in ("prefill", "decode"):
        if phase in stats:
            s = stats[phase]
            stick = "n/a" if s["stickiness"] is None else f"{s['stickiness']:.2f}"
            print(f"{phase}: {s['n_steps']} step(s) x {s['batch']} row(s), usage entropy "
                  f"{s['usage_entropy_bits_mean']:.2f} of {s['usage_entropy_bits_max']:.2f} bits, "
                  f"stickiness {stick}, top-decile share {s['top_decile_share']:.2f}")  # fmt: skip
    print(f"wrote {args.out}.prefill.json, {args.out}.decode.json (if any), {args.out}.stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
