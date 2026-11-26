"""The CPU execution backend: NumPy float32 kernels on host memory.

Synchronous: ``submit`` runs the operation and returns a completed handle with its measured
wall-clock time.  Weights arrive already dequantised to float32 (``basp.models.quants``).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import numpy as np

from basp.backends.execution import OPS, Buffer, Capabilities, Completion, ExecutionBackend, Op
from basp.models.quants import SUPPORTED

from . import kernels


def _host_memory_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return 0


class CPUBackend(ExecutionBackend):
    def __init__(self) -> None:
        self._buffers: dict[int, Buffer] = {}
        self._time: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)
        self._allocated = 0

    def capabilities(self) -> Capabilities:
        return Capabilities("numpy-cpu", "cpu", True, "", OPS, frozenset({"float32"}), frozenset(SUPPORTED),
                            _host_memory_bytes())  # fmt: skip

    def allocate(self, nbytes: int, category: str = "workspace") -> Buffer:
        if nbytes < 0:
            raise ValueError("buffer size must be non-negative")
        buf = Buffer("numpy-cpu", next(self._ids), int(nbytes), category, np.empty(nbytes // 4, np.float32))
        self._buffers[buf.buffer_id] = buf
        self._allocated += buf.nbytes
        return buf

    def upload(self, buffer: Buffer, array: np.ndarray) -> None:
        if buffer.released:
            raise ValueError("upload to a released buffer")
        array = np.ascontiguousarray(array, dtype=np.float32)
        if array.nbytes > buffer.nbytes:
            raise ValueError(f"{array.nbytes} bytes do not fit a {buffer.nbytes}-byte buffer")
        buffer.data = array

    def submit(self, op: Op) -> Completion:
        if op.kind not in OPS:
            raise ValueError(f"unknown operation {op.kind!r}")
        done = Completion(op.kind, "numpy-cpu", time.perf_counter())
        done.started_s = time.perf_counter()
        done.result = self._run(op)
        done.finished_s = time.perf_counter()
        done.done = True
        self._time[op.kind] += done.finished_s - done.started_s
        self._count[op.kind] += 1
        return done

    def _run(self, op: Op) -> Any:
        p, a = op.params, op.inputs
        if op.kind == "embedding":
            table, ids = a
            return table(ids) if callable(table) else table[ids]
        if op.kind in ("rms_norm", "head_rms_norm"):
            return kernels.rms_norm(a[0], a[1], p["eps"])
        if op.kind == "matmul":
            return kernels.matmul(a[0], a[1])
        if op.kind == "rope":
            return kernels.rope_neox(a[0], a[1], p["theta"])
        if op.kind == "attention":
            return kernels.attention(*a)
        if op.kind == "swiglu":
            return kernels.swiglu(*a)
        if op.kind == "add":
            return a[0] + a[1]
        if op.kind == "moe_router":
            return kernels.moe_router(a[0], a[1], p["top_k"], p["normalize"])
        if op.kind == "moe_experts":
            return kernels.moe_experts(*a, only=p.get("only"), out=p.get("out"))
        raise AssertionError(op.kind)

    def synchronize(self) -> None:
        return None  # every submission completes before submit returns

    def release(self, buffer: Buffer) -> None:
        if buffer.released:
            raise ValueError("buffer released twice")
        buffer.released = True
        buffer.data = None
        self._allocated -= buffer.nbytes
        self._buffers.pop(buffer.buffer_id, None)

    def telemetry(self) -> dict[str, Any]:
        return {"backend": "numpy-cpu", "allocated_bytes": self._allocated,
                "ops": {k: {"count": self._count[k], "seconds": self._time[k]} for k in sorted(self._count)}}  # fmt: skip
