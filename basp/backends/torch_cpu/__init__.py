"""PyTorch CPU execution backend (optional: needs ``torch``).

The same operations as the NumPy backend, computed with PyTorch's CPU kernels, whose small-batch
matrix products read the weights once for a few rows (NumPy's BLAS path costs about four
single-row products for two to nine rows).  That matters for speculative verification, which
multiplies a handful of rows by every weight matrix.  Weights still come from the runtime's own
dequantisation and are shared with NumPy without copying.  Unavailable, and refusing every call,
when PyTorch is not installed.  Verified against transformers like the NumPy backend.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np

from basp.backends.execution import (
    OPS,
    Buffer,
    Capabilities,
    Completion,
    ExecutionBackend,
    Op,
    UnavailableBackend,
)
from basp.models.quants import SUPPORTED

try:  # optional dependency
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - depends on the environment
    torch = None


def _t(a: Any):
    if isinstance(a, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(a))
    return a


class TorchCPUBackend(ExecutionBackend):
    def __init__(self, threads: int | None = None) -> None:
        if torch is None:
            raise ImportError("PyTorch is not installed")
        if threads:
            torch.set_num_threads(threads)
        self._time: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)

    def capabilities(self) -> Capabilities:
        return Capabilities("torch-cpu", "torch", True, "", OPS, frozenset({"float32"}), frozenset(SUPPORTED))

    def allocate(self, nbytes: int, category: str = "workspace") -> Buffer:
        return Buffer("torch-cpu", next(self._ids), int(nbytes), category, None)

    def upload(self, buffer: Buffer, array: np.ndarray) -> None:
        buffer.data = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

    def submit(self, op: Op) -> Completion:
        if op.kind not in OPS:
            raise ValueError(f"unknown operation {op.kind!r}")
        done = Completion(op.kind, "torch-cpu", time.perf_counter())
        done.started_s = time.perf_counter()
        with torch.inference_mode():
            done.result = self._run(op)
        done.finished_s = time.perf_counter()
        done.done = True
        self._time[op.kind] += done.finished_s - done.started_s
        self._count[op.kind] += 1
        return done

    def _run(self, op: Op) -> Any:
        p, a = op.params, op.inputs
        k = op.kind
        if k == "embedding":
            table, ids = a
            return table(ids) if callable(table) else table[ids]
        if k in ("rms_norm", "head_rms_norm"):
            x, w = _t(a[0]), _t(a[1])
            return (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + p["eps"]) * w).numpy()
        if k == "matmul":
            return F.linear(_t(a[0]), _t(a[1])).numpy()
        if k == "rope":
            x, positions, theta = a[0], a[1], p["theta"]
            d = x.shape[-1]
            half = d // 2
            inv = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / d)
            angles = positions.astype(np.float64)[:, None] * inv[None, :]
            cos = torch.from_numpy(np.cos(angles).astype(np.float32))[:, None, :]
            sin = torch.from_numpy(np.sin(angles).astype(np.float32))[:, None, :]
            xt = _t(x)
            x1, x2 = xt[..., :half], xt[..., half:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).numpy()
        if k == "attention":
            q, k_cache, v_cache, positions = a
            t, n_head, d = q.shape
            n_kv = k_cache.shape[1]
            length = int(positions.max()) + 1
            qt = _t(q).reshape(t, n_kv, n_head // n_kv, d).permute(1, 2, 0, 3)  # (kv, g, T, d)
            kt = _t(k_cache[:length]).permute(1, 0, 2)[:, None]  # (kv, 1, L, d)
            vt = _t(v_cache[:length]).permute(1, 0, 2)[:, None]
            scores = qt @ kt.transpose(-1, -2) / float(np.sqrt(d))  # (kv, g, T, L)
            mask = torch.from_numpy(np.arange(length)[None, :] > positions[:, None])
            scores = scores.masked_fill(mask, float("-inf"))
            out = torch.softmax(scores, dim=-1) @ vt  # (kv, g, T, d)
            return out.permute(2, 0, 1, 3).reshape(t, n_head * d).contiguous().numpy()
        if k == "swiglu":
            x, g, u, dn = (_t(v) for v in a)
            return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), dn).numpy()
        if k == "add":
            return a[0] + a[1]
        if k == "moe_router":
            logits = F.linear(_t(a[0]), _t(a[1]))
            probs = torch.softmax(logits, dim=-1)
            order = torch.sort(probs, dim=-1, descending=True, stable=True).indices[:, : p["top_k"]]
            weights = torch.gather(probs, -1, order)
            if p["normalize"]:
                weights = weights / weights.sum(-1, keepdim=True)
            return order.numpy(), weights.numpy().astype(np.float32), logits.numpy()
        if k == "moe_experts":
            x, experts, weights, fetch = a
            xt = _t(x)
            out = torch.zeros_like(xt)
            for e in np.unique(experts):
                rows, slots = np.nonzero(experts == e)
                g, u, dn = (_t(m) for m in fetch(int(e)))
                y = F.linear(F.silu(F.linear(xt[rows], g)) * F.linear(xt[rows], u), dn)
                out[rows] += torch.from_numpy(weights[rows, slots])[:, None] * y
            return out.numpy()
        raise AssertionError(k)

    def synchronize(self) -> None:
        return None

    def release(self, buffer: Buffer) -> None:
        buffer.released = True
        buffer.data = None

    def telemetry(self) -> dict[str, Any]:
        return {"backend": "torch-cpu",
                "ops": {k: {"count": self._count[k], "seconds": self._time[k]} for k in sorted(self._count)}}  # fmt: skip


def torch_backend() -> ExecutionBackend:
    """The PyTorch backend, or an unavailable one saying why."""
    if torch is None:
        backend = UnavailableBackend("PyTorch is not installed (pip install torch)")
        backend.kind = "torch"
        return backend
    return TorchCPUBackend()


__all__ = ["TorchCPUBackend", "torch_backend"]
