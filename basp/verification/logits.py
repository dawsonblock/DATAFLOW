"""Logit comparison between the runtime and the reference."""

from __future__ import annotations

from typing import Any

import numpy as np


def rel_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row relative L2 error of ``a`` against reference ``b``."""
    a, b = np.atleast_2d(a).astype(np.float64), np.atleast_2d(b).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-30)


def compare_logits(runtime: np.ndarray, reference: np.ndarray, *, top: int = 5) -> dict[str, Any]:
    errors = rel_l2(runtime, reference)
    a, b = np.atleast_2d(runtime), np.atleast_2d(reference)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return {"rows": len(a), "finite": False, "rel_l2_max": float("inf"), "rel_l2_mean": float("inf"),
                "max_abs": float("inf"), "argmax_agreement": 0.0, "top_overlap": 0.0}  # fmt: skip
    top_a = np.argsort(-a, axis=-1)[:, :top]
    top_b = np.argsort(-b, axis=-1)[:, :top]
    overlap = np.mean([len(set(x) & set(y)) / top for x, y in zip(top_a, top_b)])
    return {"rows": len(a), "finite": True, "rel_l2_max": float(errors.max()), "rel_l2_mean": float(errors.mean()),
            "max_abs": float(np.abs(a - b).max()),
            "argmax_agreement": float(np.mean(a.argmax(-1) == b.argmax(-1))), "top_overlap": float(overlap)}  # fmt: skip
