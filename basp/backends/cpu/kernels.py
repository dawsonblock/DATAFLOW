"""NumPy float32 kernels of a Qwen3-style decoder (dense and mixture-of-experts).

Conventions: activations are ``(tokens, features)`` float32; weight matrices are stored as
``(out_features, in_features)`` (the GGUF/PyTorch convention), so a projection is
``x @ W.T``.  These are reference kernels: correct and simple, not fast.
"""

from __future__ import annotations

import numpy as np


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)) * weight


def matmul(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``x @ w.T`` for ``w`` of shape (out, in)."""
    return x @ w.T


def rope_neox(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """Rotary embedding, NEOX layout (dimension i pairs with i + d/2), as Qwen3 uses.

    ``x``: (tokens, heads, head_dim); ``positions``: (tokens,).
    """
    d = x.shape[-1]
    half = d // 2
    inv = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / d)
    angles = positions.astype(np.float64)[:, None] * inv[None, :]  # (T, half)
    cos = np.cos(angles).astype(np.float32)[:, None, :]
    sin = np.sin(angles).astype(np.float32)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def attention(q: np.ndarray, k_cache: np.ndarray, v_cache: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Causal grouped-query attention of new queries against the cache.

    ``q``: (T, n_head, d); caches: (L, n_kv, d) holding positions 0..L-1 (L >= last position
    + 1); ``positions``: (T,) absolute positions of the queries.  Returns (T, n_head * d).
    """
    t, n_head, d = q.shape
    n_kv = k_cache.shape[1]
    group = n_head // n_kv
    length = int(positions.max()) + 1
    k = k_cache[:length]  # (L, n_kv, d)
    v = v_cache[:length]
    qg = q.reshape(t, n_kv, group, d)
    scores = np.einsum("tkgd,lkd->tkgl", qg, k) / np.sqrt(np.float32(d))
    mask = np.arange(length)[None, :] > positions[:, None]  # (T, L): future keys
    scores = np.where(mask[:, None, None, :], -np.inf, scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    out = np.einsum("tkgl,lkd->tkgd", p.astype(np.float32), v)
    return out.reshape(t, n_head * d)


def silu(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):  # exp(-x) overflows to inf for very negative x: silu -> -0
        return x / (1.0 + np.exp(-x))


def swiglu(x: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    return matmul(silu(matmul(x, gate)) * matmul(x, up), down)


def moe_router(x: np.ndarray, gate_inp: np.ndarray, top_k: int,
               normalize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Softmax router: experts (T, k), their weights (T, k) and the router logits (T, E).

    Ties are broken towards the lower expert index (a stable sort), as ``torch.topk`` does
    on CPU in practice; a reordered tie never changes the weighted sum.
    """
    logits = matmul(x, gate_inp)  # (T, E)
    z = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=-1, keepdims=True)
    order = np.argsort(-probs, axis=-1, kind="stable")[:, :top_k]
    weights = np.take_along_axis(probs, order, axis=-1)
    if normalize:
        weights = weights / weights.sum(axis=-1, keepdims=True)
    return order, weights.astype(np.float32), logits


def moe_experts(x: np.ndarray, experts: np.ndarray, weights: np.ndarray, expert_weights) -> np.ndarray:
    """Weighted sum of the selected experts' SwiGLU outputs.

    ``expert_weights(e) -> (gate, up, down)`` fetches one expert's matrices (so a caller can
    dequantise or cache experts on demand).  Tokens routed to the same expert are batched.
    """
    out = np.zeros((x.shape[0], x.shape[1]), np.float32)
    for e in np.unique(experts):
        rows, slots = np.nonzero(experts == e)
        gate, up, down = expert_weights(int(e))
        y = swiglu(x[rows], gate, up, down)
        out[rows] += weights[rows, slots][:, None] * y
    return out
