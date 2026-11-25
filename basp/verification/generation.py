"""Greedy generation, runtime against reference, token by token."""

from __future__ import annotations

from typing import Any

import numpy as np

from .logits import compare_logits


def reference_greedy(model, prompt: list[int], n: int) -> tuple[list[int], np.ndarray]:
    """Greedy continuation by the reference (full recomputation each step: no cache to trust)."""
    from .reference_runtime import reference_forward

    tokens = list(prompt)
    rows = []
    for _ in range(n):
        logits, _, _ = reference_forward(model, tokens)
        rows.append(logits[-1])
        tokens.append(int(np.argmax(logits[-1])))
    return tokens[len(prompt):], np.stack(rows)


def compare_generation(runtime_tokens: list[int], runtime_logits: np.ndarray, reference_tokens: list[int],
                       reference_logits: np.ndarray) -> dict[str, Any]:  # fmt: skip
    n = min(len(runtime_tokens), len(reference_tokens))
    first = next((i for i in range(n) if runtime_tokens[i] != reference_tokens[i]), None)
    same = first is None and len(runtime_tokens) == len(reference_tokens)
    # logits are comparable up to and including the first divergent step (same inputs so far)
    upto = n if first is None else first + 1
    out = {"tokens": n, "identical": same, "first_divergence": first,
           "runtime": list(map(int, runtime_tokens)), "reference": list(map(int, reference_tokens)),
           "decode_logits": compare_logits(runtime_logits[:upto], reference_logits[:upto])}  # fmt: skip
    if first is not None:
        ref_row = reference_logits[first]
        top2 = np.sort(ref_row)[-2:]
        out["reference_margin_at_divergence"] = float(top2[1] - top2[0])  # a near-tie explains a flip
    return out
