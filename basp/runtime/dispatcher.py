"""Send each operation to the backend the plan assigned to its group, and time it."""

from __future__ import annotations

from typing import Any

from basp.backends.execution import Op

from .command_queue import CommandQueue
from .device_manager import DeviceManager
from .execution_plan import GROUPS, ExecutionPlan
from .telemetry import Telemetry


class Dispatcher:
    def __init__(self, plan: ExecutionPlan, devices: DeviceManager, telemetry: Telemetry) -> None:
        self.plan = plan
        self.telemetry = telemetry
        self.queues = {name: CommandQueue(devices.backends[name]) for name in set(plan.assignment.values())}

    def run(self, group: str, kind: str, *inputs: Any, label: str = "", **params: Any) -> Any:
        if kind not in GROUPS[group]:
            raise ValueError(f"operation {kind!r} is not part of group {group!r}")
        backend = self.plan.backend_for(group)
        completion = self.queues[backend].enqueue(Op(kind, inputs, params, label))
        if not completion.done:
            self.queues[backend].flush()
        if completion.error:
            raise RuntimeError(f"{backend}: {kind} failed: {completion.error}")
        self.telemetry.record(backend, kind, completion.elapsed_s or 0.0)
        return completion.result
