"""Golden numerical reference: plain float64 numpy, unaware of quantization or streaming.

Every kernel in this package is verified against these functions. Conventions: a weight
matrix is ``[rows, cols]`` (``[out_features, in_features]``); a single activation is
``[cols]`` and a batch is ``[batch, cols]``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def matvec(w, x) -> np.ndarray:
    """``w @ x`` for ``w`` [rows, cols] and ``x`` [cols] -> float64 [rows]."""
    w, x = _f64(w), _f64(x)
    if w.ndim != 2 or x.shape != (w.shape[1],):
        raise ValueError(f"matvec shape mismatch: w {w.shape}, x {x.shape}")
    return w @ x


def matmul(w, x) -> np.ndarray:
    """``x @ w.T`` for ``w`` [rows, cols] and ``x`` [batch, cols] -> float64 [batch, rows]."""
    w, x = _f64(w), _f64(x)
    if w.ndim != 2 or x.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError(f"matmul shape mismatch: w {w.shape}, x {x.shape}")
    return x @ w.T


def silu(x) -> np.ndarray:
    """``x * sigmoid(x)``, evaluated without overflow for large ``|x|``."""
    x = _f64(x)
    return x * np.exp(-np.logaddexp(0.0, -x))


def swiglu_expert(x, w_gate, w_up, w_down) -> np.ndarray:
    """``W_down @ (silu(W_gate x) * (W_up x))`` for ``x`` [hidden] or [batch, hidden]."""
    x, w_gate, w_up, w_down = _f64(x), _f64(w_gate), _f64(w_up), _f64(w_down)
    return (silu(x @ w_gate.T) * (x @ w_up.T)) @ w_down.T


def rms_norm(x, weight, eps: float = 1e-6) -> np.ndarray:
    """``x / sqrt(mean(x^2) + eps) * weight`` over the last axis."""
    x = _f64(x)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * _f64(weight)


def weighted_reduce(outputs: Sequence, coeffs: Sequence[float]) -> np.ndarray:
    """``sum_i coeffs[i] * outputs[i]``."""
    if len(outputs) != len(coeffs) or len(outputs) == 0:
        raise ValueError("weighted_reduce needs one coefficient per output and at least one")
    acc = np.zeros_like(_f64(outputs[0]))
    for out, coeff in zip(outputs, coeffs):
        acc += float(coeff) * _f64(out)
    return acc


def route_topk(logits, k: int, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Top-k expert routing: softmax over all logits, keep the k largest.

    Returns ``(indices int64, coeffs float64)`` in descending-probability order; ties go to
    the lower index. With ``normalize`` the kept coefficients are rescaled to sum to 1
    (Qwen3 ``norm_topk_prob``). Accepts ``[n]`` or ``[batch, n]`` logits.
    """
    logits = _f64(logits)
    if logits.ndim not in (1, 2):
        raise ValueError(f"logits must be [n] or [batch, n], got shape {logits.shape}")
    n = logits.shape[-1]
    if not 1 <= k <= n:
        raise ValueError(f"k={k} must be in [1, {n}]")
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = shifted / shifted.sum(axis=-1, keepdims=True)
    # Rank on logits (not probabilities) so softmax saturation cannot create spurious ties;
    # a stable sort of the negated logits breaks genuine ties toward the lower index.
    indices = np.argsort(-logits, axis=-1, kind="stable")[..., :k].astype(np.int64)
    coeffs = np.take_along_axis(probs, indices, axis=-1)
    if normalize:
        coeffs = coeffs / coeffs.sum(axis=-1, keepdims=True)
    return indices, coeffs
