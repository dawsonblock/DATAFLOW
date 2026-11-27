"""Expert-selection comparison: the runtime must select exactly the experts the reference does."""

from __future__ import annotations

from typing import Any

import numpy as np


def compare_routing(runtime: list[np.ndarray], reference: list[np.ndarray]) -> dict[str, Any]:
    """Per layer, per token: is the *set* of selected experts identical?  (Order within the
    top-k does not change the weighted sum.)"""
    if len(runtime) != len(reference):
        return {"layers": len(reference), "exact": False, "reason": f"{len(runtime)} vs {len(reference)} layers"}
    mismatches = []
    tokens = 0
    for layer, (a, b) in enumerate(zip(runtime, reference)):
        a, b = np.atleast_2d(a), np.atleast_2d(b)
        if a.shape != b.shape:
            return {"layers": len(reference), "exact": False, "reason": f"layer {layer}: shape {a.shape} vs {b.shape}"}
        for t, (x, y) in enumerate(zip(a, b)):
            tokens += 1
            if set(map(int, x)) != set(map(int, y)):
                mismatches.append({"layer": layer, "token": t, "runtime": sorted(map(int, x)),
                                   "reference": sorted(map(int, y))})  # fmt: skip
    return {"layers": len(reference), "token_selections": tokens, "mismatches": len(mismatches),
            "exact": not mismatches, "first_mismatches": mismatches[:5]}  # fmt: skip
