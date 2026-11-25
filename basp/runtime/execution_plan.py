"""Which backend runs which part of the model -- decided, and checked, before anything runs.

Operations are grouped (embedding, attention, dense FFN, router, experts, output); each group
is assigned the first backend in the preference order that is available and supports every
operation of the group.  If no backend can run a group the plan fails with
``UnsupportedOperation`` naming the group and the missing operations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from basp.backends.execution import UnsupportedOperation

from .config import ModelConfig
from .device_manager import DeviceManager

GROUPS: dict[str, frozenset[str]] = {
    "embedding": frozenset({"embedding"}),
    "attention": frozenset({"rms_norm", "head_rms_norm", "matmul", "rope", "attention", "add"}),
    "ffn": frozenset({"rms_norm", "swiglu", "add"}),
    "router": frozenset({"rms_norm", "moe_router"}),
    "experts": frozenset({"moe_experts", "add"}),
    "output": frozenset({"rms_norm", "matmul"}),
}


def required_groups(config: ModelConfig) -> list[str]:
    groups = ["embedding", "attention"]
    groups += ["router", "experts"] if config.moe else ["ffn"]
    return [*groups, "output"]


@dataclass
class ExecutionPlan:
    assignment: dict[str, str]  # group -> backend name
    rejected: dict[str, list[str]] = field(default_factory=dict)  # backend -> why it was not used

    def backend_for(self, group: str) -> str:
        return self.assignment[group]

    def to_json(self) -> dict[str, Any]:
        return {"assignment": dict(self.assignment), "rejected": dict(self.rejected)}


def make_plan(config: ModelConfig, devices: DeviceManager, *, prefer: Sequence[str] = ("cuda", "metal", "ggml", "cpu"),
              overrides: dict[str, str] | None = None) -> ExecutionPlan:  # fmt: skip
    """Assign every group to a backend; ``prefer`` lists backend kinds, ``overrides`` pins a
    group to a backend kind (and fails if that backend cannot run it)."""
    order = [b for kind in prefer for b in devices.by_kind(kind)]
    rejected: dict[str, list[str]] = {}
    for b in devices.backends.values():
        caps = b.capabilities()
        if not caps.available:
            rejected[caps.name] = [caps.reason]
    assignment = {}
    for group in required_groups(config):
        ops = GROUPS[group]
        pinned = (overrides or {}).get(group)
        candidates = [b for b in order if pinned is None or b.capabilities().kind == pinned]
        chosen = next((b for b in candidates if all(b.capabilities().supports(op) for op in ops)), None)
        if chosen is None:
            detail = "; ".join(f"{b.capabilities().name}: " + (b.capabilities().reason if not b.capabilities().available
                               else "lacks " + ", ".join(sorted(op for op in ops if not b.capabilities().supports(op))))
                               for b in candidates) or "no backend of that kind"  # fmt: skip
            raise UnsupportedOperation(f"no backend can run {group} ({', '.join(sorted(ops))}): {detail}")
        assignment[group] = chosen.capabilities().name
    return ExecutionPlan(assignment, rejected)
