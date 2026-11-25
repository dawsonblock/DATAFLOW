"""Verify the physical runtime against the transformers reference on a real GGUF model.

    python -m benchmarks.verify_runtime --gguf ~/models/Qwen3-0.6B-Q8_0.gguf --model qwen3-0.6b \
        --tokenizer ~/models/qwen3-0.6b-hf --out evidence/verification/qwen3-0.6b

Runs the model on the NumPy CPU runtime and on transformers (float32, same weights, dequantised
by the reference ``gguf`` package), compares logits, every layer's output, expert routing,
greedy generation and perplexity on a fixed text, and writes ``parity.json`` (all results and
the verdict) and ``receipts.json`` (E3 receipts) to ``--out``.  Needs torch, transformers and
gguf.  With ``--hf-original DIR`` it also reports how far the quantised file's predictions are
from the original checkpoint's (information about the quantisation, not about the runtime).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: fixed evaluation text (the opening of the Python tutorial's introduction, public domain style)
EVAL_TEXT = (
    "Python is an easy to learn, powerful programming language. It has efficient high-level data "
    "structures and a simple but effective approach to object-oriented programming. Python's elegant "
    "syntax and dynamic typing, together with its interpreted nature, make it an ideal language for "
    "scripting and rapid application development in many areas on most platforms."
)
PROMPT = "The capital of France is"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--model", required=True, help="model name the evidence is about (e.g. qwen3-0.6b)")
    parser.add_argument("--tokenizer", required=True, help="directory with the model's Hugging Face tokenizer")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--generate", type=int, default=16)
    parser.add_argument("--hf-original", help="original checkpoint directory (quantisation effect, optional)")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from basp.evidence import Receipt, ReceiptRegistry
    from basp.verification.parity_report import run_parity

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompt = tok(args.prompt)["input_ids"]
    eval_tokens = tok(EVAL_TEXT)["input_ids"]
    prefer = ("torch",) if args.backend == "torch" else ("cpu",)
    result = run_parity(args.gguf, prompt, model_name=args.model, eval_tokens=eval_tokens, n_generate=args.generate,
                        prefer=prefer)  # fmt: skip
    result["prompt_text"] = args.prompt
    result["generated_text"] = {"runtime": tok.decode(result["generation"]["runtime"]),
                                "reference": tok.decode(result["generation"]["reference"])}  # fmt: skip
    if args.hf_original:
        result["quantisation"] = quantisation_effect(args.gguf, args.hf_original, eval_tokens)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "parity.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=float) + "\n")
    ReceiptRegistry(Receipt.from_json(r) for r in result["receipts"]).save(args.out / "receipts.json")
    p = result.get("perplexity", {})
    print(f"{args.model}: {result['verdict'].upper()}  logits rel L2 {result['numerical_error']:.2e}, "
          f"greedy tokens identical: {result['generation']['identical']}, relative perplexity "
          f"{p.get('relative', float('nan')):.6f}")  # fmt: skip
    print(f"runtime: {result['generated_text']['runtime']!r}")
    for reason in result["reasons"]:
        print("  ", reason)
    return 0 if result["verdict"] == "pass" else 1


def quantisation_effect(gguf_path: str, original_dir: str, tokens: list[int]) -> dict:
    """Perplexity of the runtime on the quantised file against the original checkpoint's
    (transformers, float32).  Describes the quantisation, not the runtime."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    from basp.runtime import Engine
    from basp.verification import compare_logits, perplexity

    original = AutoModelForCausalLM.from_pretrained(original_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        ref = original(torch.tensor([tokens])).logits[0].float().numpy()
    engine = Engine(gguf_path)
    with engine.session(len(tokens) + 1) as s:
        rt = s.prefill(tokens, all_logits=True).logits
    a, b = perplexity(rt, np.asarray(tokens)), perplexity(ref, np.asarray(tokens))
    return {"original_perplexity": b, "quantised_runtime_perplexity": a, "relative": a / b,
            "logits": compare_logits(rt, ref)}  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
