"""Backend discovery: which execution backends exist on this machine, and which do not."""

from __future__ import annotations

from typing import Any

from basp.backends.cpu import CPUBackend
from basp.backends.cuda import CUDABackend
from basp.backends.execution import ExecutionBackend
from basp.backends.ggml import GGMLBackend
from basp.backends.metal import MetalBackend
from basp.backends.remote import RemoteBackend


class DeviceManager:
    def __init__(self, backends: list[ExecutionBackend]) -> None:
        self.backends = {b.capabilities().name: b for b in backends}

    @classmethod
    def discover(cls) -> DeviceManager:
        """Every backend this release knows; only the CPU backend is implemented."""
        return cls([CPUBackend(), CUDABackend(), MetalBackend(), GGMLBackend(), RemoteBackend()])

    def available(self) -> list[ExecutionBackend]:
        return [b for b in self.backends.values() if b.capabilities().available]

    def by_kind(self, kind: str) -> list[ExecutionBackend]:
        return [b for b in self.backends.values() if b.capabilities().kind == kind]

    def report(self) -> list[dict[str, Any]]:
        return [b.capabilities().to_json() for b in self.backends.values()]
