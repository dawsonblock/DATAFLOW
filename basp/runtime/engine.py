"""The physical runtime: load a GGUF model, plan its execution on the available backends,
and run sessions.

Separate from the simulator: nothing here imports ``basp.simulator`` or the virtual backend,
and every timing it reports is measured on this machine.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from basp.memory import AdmissionController, MemoryDomain
from basp.models.gguf_tensors import GGUFTensors

from .config import ModelConfig
from .device_manager import DeviceManager
from .dispatcher import Dispatcher
from .execution_plan import make_plan
from .model import Qwen3Model
from .session import Session
from .telemetry import Telemetry
from .weights import WeightStore


class Engine:
    def __init__(self, path: str | os.PathLike, *, devices: DeviceManager | None = None,
                 prefer: Sequence[str] = ("cuda", "metal", "ggml", "cpu"), overrides: dict[str, str] | None = None,
                 memory_bytes: int | None = None, reserved_fraction: float = 0.1,
                 expert_cache_bytes: int | None = None, embedding: str = "resident",
                 workers: int | None = None) -> None:  # fmt: skip
        self.path = os.fspath(path)
        self.tensors = GGUFTensors(self.path)
        self.config = ModelConfig.from_gguf(self.tensors.file)
        self.devices = devices or DeviceManager.discover()
        self.plan = make_plan(self.config, self.devices, prefer=prefer, overrides=overrides)
        self.telemetry = Telemetry()
        self.dispatcher = Dispatcher(self.plan, self.devices, self.telemetry)
        host = next(b for b in self.devices.available() if b.capabilities().kind == "cpu")
        capacity = memory_bytes or host.capabilities().memory_bytes or 8 << 30
        self.domain = MemoryDomain("host", capacity, devices=["cpu"], reserved=int(capacity * reserved_fraction))
        self.admission = AdmissionController(self.domain)
        self.weights = WeightStore(self.tensors, self.config, self.domain, self.telemetry,
                                   expert_cache_bytes=expert_cache_bytes, embedding=embedding,
                                   workers=workers)  # fmt: skip
        self.model = Qwen3Model(self.config, self.weights, self.dispatcher)

    def close(self) -> None:
        self.weights.close()

    def session(self, max_tokens: int = 512) -> Session:
        return Session(self, max_tokens)

    def report(self) -> dict[str, Any]:
        return {"model": os.path.basename(self.path), "config": self.config.__dict__, "plan": self.plan.to_json(),
                "devices": self.devices.report(), "memory": self.domain.budget(),
                "telemetry": self.telemetry.to_json()}  # fmt: skip
