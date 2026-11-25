"""The seam between scheduling and execution: what the runtime asks of a backend.

The virtual runtime (``basp.backends.virtual``) decides *when* and *where* every
operation runs and plays the schedule on simulated devices.  A **payload** computes
the *values*: it is told about every operation as the schedule executes it, in a
dependency-respecting order, together with the plan the scheduler chose.

``NumericPayload`` (``basp.backends.cpu``) is the only implementation today: NumPy INT4
reference kernels on a small random-weight model.  A backend that executes a real model
(GGML/llama.cpp kernels, CUDA, Metal; roadmap v0.5-v0.6) implements the same four calls.
It must not change routing: ``route`` reports the experts the model's own router
selected, and the runtime executes exactly those.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from basp.core.graph import OpNode
from basp.dataflow.base import ResolvedPlan
from basp.workloads.trace import RoutingStep


@runtime_checkable
class Payload(Protocol):
    def begin_step(self, step: int, x: int) -> None:
        """A decoding step starts; ``x`` is the tensor id of its input activation."""

    def route(self, step: int, layer: int, logits: int) -> RoutingStep:
        """The experts the model's router selects in ``layer``, from the router-logits tensor."""

    def execute(self, op: OpNode, plan: ResolvedPlan) -> None:
        """Compute ``op``'s outputs as ``plan`` places them (called once its inputs exist)."""

    def end_step(self, step: int) -> None:
        """The step's last operation has run (the output token is known)."""


__all__ = ["Payload"]
