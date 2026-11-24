"""Initial tensor placement policies.

Placement decides which device owns each weight tensor before execution starts;
the scheduler may later replicate tensors but never moves ownership.  Every
policy returns a :class:`PlacementReport` and sets ``program.trunk_device``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from basp.compiler.lower import ModelProgram
from basp.core.tensor import CompressionFormat
from basp.memory.allocator import OutOfMemory
from basp.memory.residency import ResidencyMap


class PlacementError(RuntimeError):
    pass


@dataclass
class PlacementReport:
    policy: str
    bytes_by_device: dict[int, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, device: int, nbytes: int) -> None:
        self.bytes_by_device[device] = self.bytes_by_device.get(device, 0) + nbytes


def _place(residency: ResidencyMap, report: PlacementReport, tensor: int, device: int, *,
           pinned: bool = False, owner: bool = True, fmt: CompressionFormat | None = None) -> None:  # fmt: skip
    try:
        copy = residency.place(tensor, device, pinned=pinned, owner=owner, fmt=fmt)
    except OutOfMemory as exc:
        name = residency.topology.devices[device].name
        raise PlacementError(f"{residency.registry.spec(tensor).name} does not fit on {name}") from exc
    report.add(device, copy.alloc.size)


def _place_trunk(program: ModelProgram, residency: ResidencyMap, report: PlacementReport,
                 layer: int, device: int) -> None:  # fmt: skip
    program.trunk_device[layer] = device
    _place(residency, report, program.trunk_w[layer], device, pinned=True)
    if program.kv:
        _place(residency, report, program.kv[layer], device, pinned=True)


def place_single(program: ModelProgram, residency: ResidencyMap, device: int) -> PlacementReport:
    """Everything on one device (CPU-only, or GPU-only when the model fits)."""
    report = PlacementReport("single")
    for layer in range(program.spec.n_layers):
        _place_trunk(program, residency, report, layer, device)
        for tensor in program.expert_w[layer]:
            _place(residency, report, tensor, device)
    for tensor in program.head_w:
        _place(residency, report, tensor, device, pinned=True)
    program.sample_device = device
    return report


def place_gpu_host(
    program: ModelProgram,
    residency: ResidencyMap,
    gpu: int,
    host: int,
    *,
    prefill: Sequence[tuple[int, int]] = (),
    cold_format: CompressionFormat | None = None,
) -> PlacementReport:
    """Trunk and output head pinned on the GPU; experts owned by host memory.

    ``prefill`` lists (layer, expert) pairs replicated onto the GPU up front, in
    priority order, until it is full.  ``cold_format`` stores the host copies in
    a different (lossless-equivalent) representation.
    """
    report = PlacementReport("gpu_host")
    for layer in range(program.spec.n_layers):
        _place_trunk(program, residency, report, layer, gpu)
    for tensor in program.head_w:
        _place(residency, report, tensor, gpu, pinned=True)
    for layer in range(program.spec.n_layers):
        for tensor in program.expert_w[layer]:
            _place(residency, report, tensor, host, fmt=cold_format)
    filled = 0
    for layer, expert in prefill:
        tensor = program.expert_w[layer][expert]
        if residency.free_bytes(gpu) < residency.registry.spec(tensor).nbytes + 64:
            break
        _place(residency, report, tensor, gpu, owner=False)
        filled += 1
    if prefill:
        report.notes.append(f"prefilled {filled} experts on the GPU")
    program.sample_device = gpu
    return report


def place_static_split(
    program: ModelProgram,
    residency: ResidencyMap,
    gpu: int,
    host: int,
    n_gpu_layers: int | None = None,
) -> PlacementReport:
    """Conventional layer offloading: the first N whole layers on the GPU, the rest on the host.

    With ``n_gpu_layers=None`` the largest N that fits GPU memory is used.
    """
    spec = program.spec
    registry = residency.registry
    per_layer = [
        registry.spec(program.trunk_w[layer]).nbytes
        + (registry.spec(program.kv[layer]).nbytes if program.kv else 0)
        + sum(registry.spec(t).nbytes for t in program.expert_w[layer])
        + 64 * (spec.n_experts + 2)
        for layer in range(spec.n_layers)
    ]  # fmt: skip
    if n_gpu_layers is None:
        budget = residency.free_bytes(gpu)
        n_gpu_layers = 0
        for nbytes in per_layer:
            if nbytes > budget:
                break
            budget -= nbytes
            n_gpu_layers += 1
    report = PlacementReport("static_split")
    report.notes.append(f"{n_gpu_layers} of {spec.n_layers} layers on the GPU")
    for layer in range(spec.n_layers):
        device = gpu if layer < n_gpu_layers else host
        _place_trunk(program, residency, report, layer, device)
        for tensor in program.expert_w[layer]:
            _place(residency, report, tensor, device)
    last = program.trunk_device[-1]
    head_bytes = sum(registry.spec(t).nbytes for t in program.head_w) + 64 * len(program.head_w)
    head_device = gpu if last == gpu and residency.free_bytes(gpu) >= head_bytes else last
    for tensor in program.head_w:
        _place(residency, report, tensor, head_device, pinned=True)
    program.sample_device = program.trunk_device[0]
    return report


def place_modules(
    program: ModelProgram,
    residency: ResidencyMap,
    trunk: int,
    modules: Sequence[int],
    *,
    policy: str = "balanced",
    frequencies: np.ndarray | None = None,
    head_on_modules: bool = True,
    prefill: Sequence[tuple[int, int]] = (),
    cold_format: CompressionFormat | None = None,
) -> PlacementReport:
    """Complete experts on compute-memory modules; trunk on ``trunk``.

    ``round_robin`` assigns expert ``e`` to module ``e mod M``.  ``balanced``
    assigns the experts of each layer in descending expected frequency to the
    module with the least expected load so far, so frequently co-selected
    experts end up on different modules and can run concurrently.
    """
    if not modules:
        raise PlacementError("no modules to place experts on")
    spec = program.spec
    report = PlacementReport(f"modules/{policy}")
    for layer in range(spec.n_layers):
        _place_trunk(program, residency, report, layer, trunk)
    for layer in range(spec.n_layers):
        if policy == "round_robin" or frequencies is None:
            target = [modules[e % len(modules)] for e in range(spec.n_experts)]
        elif policy == "balanced":
            load = {m: 0.0 for m in modules}
            count = {m: 0 for m in modules}
            target = [modules[0]] * spec.n_experts
            order = sorted(range(spec.n_experts), key=lambda e: (-frequencies[layer][e], e))
            for e in order:
                m = min(modules, key=lambda d: (load[d], count[d], d))
                target[e] = m
                load[m] += float(frequencies[layer][e])
                count[m] += 1
        else:
            raise PlacementError(f"unknown expert distribution policy {policy!r}")
        for e, tensor in enumerate(program.expert_w[layer]):
            _place(residency, report, tensor, target[e], fmt=cold_format)
    if head_on_modules and len(program.head_w) > 1:
        for i, tensor in enumerate(program.head_w):
            _place(residency, report, tensor, modules[i % len(modules)], pinned=True)
    else:
        for tensor in program.head_w:
            _place(residency, report, tensor, trunk, pinned=True)
    filled = 0
    for layer, expert in prefill:
        tensor = program.expert_w[layer][expert]
        if residency.free_bytes(trunk) < residency.registry.spec(tensor).nbytes + 64:
            break
        _place(residency, report, tensor, trunk, owner=False)
        filled += 1
    if prefill:
        report.notes.append(f"prefilled {filled} experts on the trunk device")
    program.sample_device = trunk
    return report


def hottest_experts(frequencies: np.ndarray) -> list[tuple[int, int]]:
    """(layer, expert) pairs in descending selection frequency (ties: layer, expert)."""
    n_layers, n_experts = frequencies.shape
    pairs = [(layer, e) for layer in range(n_layers) for e in range(n_experts)]
    return sorted(pairs, key=lambda p: (-frequencies[p[0]][p[1]], p[0], p[1]))
