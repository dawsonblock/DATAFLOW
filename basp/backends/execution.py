"""The physical execution contract: what the runtime (``basp.runtime``) asks of a device backend.

This is separate from the simulator's ``Payload`` contract (``basp.backends.base``): a payload
computes values while the *simulated* system decides timing; an ``ExecutionBackend`` runs
operations on a real device and its timings are measurements.

A backend declares what it can do (``capabilities``) and the runtime plans against that
declaration before running anything: an operation no available backend supports makes the
plan fail with ``UnsupportedOperation``.  A backend that is not available on this machine
(no driver, not built, or not implemented) says so and refuses every call with
``BackendUnavailable`` -- it never pretends to compute.
"""

from __future__ import annotations

import abc
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: operations of a decoder-only transformer, as the runtime issues them
OPS = frozenset({
    "embedding", "rms_norm", "head_rms_norm", "matmul", "rope", "attention", "swiglu", "add",
    "moe_router", "moe_experts",
})  # fmt: skip


class BackendUnavailable(RuntimeError):
    """The backend cannot run on this machine (or is not implemented)."""


class UnsupportedOperation(RuntimeError):
    """No selected backend supports an operation the model needs."""


@dataclass(frozen=True)
class Capabilities:
    name: str
    kind: str  # cpu | cuda | metal | ggml | remote
    available: bool
    reason: str = ""  # why it is not available
    ops: frozenset[str] = frozenset()
    dtypes: frozenset[str] = frozenset()  # dtypes buffers can hold
    weight_formats: frozenset[str] = frozenset()  # ggml formats it executes (after its own dequantisation)
    memory_bytes: int = 0

    def supports(self, op: str) -> bool:
        return self.available and op in self.ops

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "available": self.available, "reason": self.reason,
                "ops": sorted(self.ops), "dtypes": sorted(self.dtypes),
                "weight_formats": sorted(self.weight_formats), "memory_bytes": self.memory_bytes}  # fmt: skip


@dataclass(eq=False)
class Buffer:
    backend: str
    buffer_id: int
    nbytes: int
    category: str
    data: Any = None  # backend-specific handle (a NumPy array on the CPU)
    released: bool = False


@dataclass
class Completion:
    """Handle for a submitted operation.  ``done`` once its outputs are valid."""

    op: str
    backend: str
    submitted_s: float
    done: bool = False
    started_s: float | None = None
    finished_s: float | None = None
    error: str | None = None
    result: Any = None

    @property
    def elapsed_s(self) -> float | None:
        if self.started_s is None or self.finished_s is None:
            return None
        return self.finished_s - self.started_s


@dataclass
class Op:
    """One operation: kind, NumPy-level inputs and parameters."""

    kind: str
    inputs: tuple[Any, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    label: str = ""


class ExecutionBackend(abc.ABC):
    """allocate / upload / submit / synchronize / release / telemetry."""

    _ids = itertools.count()

    @abc.abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abc.abstractmethod
    def allocate(self, nbytes: int, category: str = "workspace") -> Buffer: ...

    @abc.abstractmethod
    def upload(self, buffer: Buffer, array: np.ndarray) -> None: ...

    @abc.abstractmethod
    def submit(self, op: Op) -> Completion: ...

    @abc.abstractmethod
    def synchronize(self) -> None: ...

    @abc.abstractmethod
    def release(self, buffer: Buffer) -> None: ...

    @abc.abstractmethod
    def telemetry(self) -> dict[str, Any]: ...


class UnavailableBackend(ExecutionBackend):
    """A backend that does not run here.  Every call refuses; nothing is fabricated."""

    kind = "unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def capabilities(self) -> Capabilities:
        return Capabilities(self.kind, self.kind, False, self.reason)

    def _refuse(self, *_args, **_kwargs):
        raise BackendUnavailable(f"{self.kind} backend: {self.reason}")

    allocate = upload = submit = synchronize = release = _refuse  # type: ignore[assignment]

    def telemetry(self) -> dict[str, Any]:
        return {"backend": self.kind, "available": False, "reason": self.reason}
