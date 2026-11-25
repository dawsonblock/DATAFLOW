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
from basp.workloads.trace import RoutingTrace


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
        for tensor in program.layer_expert_tensors(layer):
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
        for tensor in program.layer_expert_tensors(layer):
            _place(residency, report, tensor, host, fmt=cold_format)
    filled = _prefill(program, residency, report, gpu, prefill)
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
    n_cpu_moe: int = 0,
) -> PlacementReport:
    """Conventional layer offloading: the first N whole layers on the GPU, the rest on the host.

    With ``n_gpu_layers=None`` the largest N that fits GPU memory is used.  ``n_cpu_moe`` keeps
    the expert weights of the first M layers on the host even where the layer is on the GPU
    (llama.cpp ``--n-cpu-moe``): attention runs on the GPU, those experts on the CPU.
    """
    spec = program.spec
    registry = residency.registry
    per_layer = [
        registry.spec(program.trunk_w[layer]).nbytes
        + (registry.spec(program.kv[layer]).nbytes if program.kv else 0)
        + sum(registry.spec(t).nbytes for t in program.layer_expert_tensors(layer))
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
    if n_cpu_moe < 0:
        raise PlacementError("n_cpu_moe must be >= 0")
    report = PlacementReport("static_split")
    report.notes.append(f"{n_gpu_layers} of {spec.n_layers} layers on the GPU")
    if n_cpu_moe:
        report.notes.append(f"experts of the first {min(n_cpu_moe, spec.n_layers)} layers on the host")
    for layer in range(spec.n_layers):
        device = gpu if layer < n_gpu_layers else host
        _place_trunk(program, residency, report, layer, device)
        experts_on = host if layer < n_cpu_moe else device
        for tensor in program.layer_expert_tensors(layer):
            _place(residency, report, tensor, experts_on)
    last = program.trunk_device[-1]
    head_bytes = sum(registry.spec(t).nbytes for t in program.head_w) + 64 * len(program.head_w)
    head_device = gpu if last == gpu and residency.free_bytes(gpu) >= head_bytes else last
    for tensor in program.head_w:
        _place(residency, report, tensor, head_device, pinned=True)
    program.sample_device = program.trunk_device[0]
    return report


def coactivation(trace: RoutingTrace, *, prior_visits: float | None = None) -> np.ndarray:
    """[layers, experts, experts] probability that two experts are selected in the same visit.

    The empirical co-selection rate is shrunk towards independence (the product
    of the two selection frequencies) with the weight of ``prior_visits`` pseudo
    observations.  Pair probabilities are tiny -- about ``(top_k / n_experts)^2``
    -- so they need thousands of visits to estimate; the default prior is 16
    times the reciprocal of that probability.  A short profiling trace therefore
    yields essentially a frequency-balanced placement, and only well-supported
    co-activation structure changes where experts go.
    """
    counts = np.zeros((trace.n_layers, trace.n_experts, trace.n_experts))
    visits = 0
    for step in trace.steps:
        for layer, routing in enumerate(step):
            for row in routing.experts:
                idx = np.asarray(row)
                counts[layer][np.ix_(idx, idx)] += 1.0
        visits += trace.batch
    visits = max(visits, 1)
    if prior_visits is None:
        prior_visits = 16.0 * (trace.n_experts / trace.top_k) ** 2
    empirical = counts / visits
    freq = np.einsum("lii->li", empirical)
    independent = freq[:, :, None] * freq[:, None, :]
    weight = visits / (visits + prior_visits)
    return weight * empirical + (1.0 - weight) * independent


def _assign_experts(n_experts: int, modules: Sequence[int], policy: str, freq, co, copies: Sequence[int]):
    """Devices for every expert of one layer: a list of device tuples, primary first."""
    if policy == "round_robin" or freq is None:
        primary = [modules[e % len(modules)] for e in range(n_experts)]
        order = list(range(n_experts))
    elif policy in ("balanced", "coactivation"):
        order = sorted(range(n_experts), key=lambda e: (-freq[e], e))
        primary = [modules[0]] * n_experts
    else:
        raise PlacementError(f"unknown expert distribution policy {policy!r}")
    load = dict.fromkeys(modules, 0.0)
    count = dict.fromkeys(modules, 0)
    members: dict[int, list[int]] = {m: [] for m in modules}

    def clash(e: int, m: int) -> float:
        # expected number of experts already on m that are selected together with e
        return float(co[e, members[m]].sum()) if co is not None and members[m] else 0.0

    placed: list[tuple[int, ...]] = [()] * n_experts
    for e in order:
        if policy == "round_robin" or freq is None:
            first = primary[e]
        elif policy == "coactivation" and co is not None:
            first = min(modules, key=lambda d: (clash(e, d), load[d], count[d], d))
        else:
            first = min(modules, key=lambda d: (load[d], count[d], d))
        chosen = [first]
        share = float(freq[e]) / copies[e] if freq is not None else 1.0 / copies[e]
        for _ in range(copies[e] - 1):
            rest = [m for m in modules if m not in chosen]
            if not rest:
                break
            chosen.append(min(rest, key=lambda d: (clash(e, d), load[d], count[d], d)))
        for m in chosen:
            members[m].append(e)
            load[m] += share
            count[m] += 1
        placed[e] = tuple(chosen)
    return placed


def place_modules(
    program: ModelProgram,
    residency: ResidencyMap,
    trunk: int | Sequence[int],
    modules: Sequence[int],
    *,
    policy: str = "balanced",
    frequencies: np.ndarray | None = None,
    coactivation_matrix: np.ndarray | None = None,
    replicas: int = 1,
    replicate_top: float = 1.0,
    head: str = "auto",
    head_on_modules: bool | None = None,
    prefill: Sequence[tuple[int, int]] = (),
    cold_format: CompressionFormat | None = None,
) -> PlacementReport:
    """Complete experts on compute-memory modules; trunk blocks on ``trunk``.

    Expert distribution policies:

    * ``round_robin`` -- expert ``e`` on module ``e mod M``.
    * ``balanced`` -- experts in descending frequency to the least-loaded module.
    * ``coactivation`` -- experts in descending frequency to the module where they
      collide least with experts already placed there (experts that are selected
      together should not share a module), then by load.

    ``replicas`` copies of the hottest ``replicate_top`` share of each layer's
    experts are placed on distinct modules, which lets the scheduler send each
    selected expert to the less loaded of its holders.

    ``trunk`` is one device for every layer or one device per layer.  ``head``
    selects where the output projection lives: ``"modules"`` (row shards only),
    ``"trunk"``, or ``"auto"``: shards on the modules *and* a full copy on the
    trunk device when it fits, so the scheduler picks the location per token.
    """
    if not modules:
        raise PlacementError("no modules to place experts on")
    if replicas < 1 or not 0.0 <= replicate_top <= 1.0:
        raise PlacementError("replicas must be >= 1 and replicate_top within [0, 1]")
    if policy not in ("round_robin", "balanced", "coactivation"):
        raise PlacementError(f"unknown expert distribution policy {policy!r}")
    if head_on_modules is not None:  # backwards-compatible spelling
        head = "modules" if head_on_modules else "trunk"
    spec = program.spec
    trunks = [trunk] * spec.n_layers if isinstance(trunk, int) else list(trunk)
    if len(trunks) != spec.n_layers:
        raise PlacementError("need one trunk device or one per layer")
    report = PlacementReport(f"modules/{policy}" + (f"/x{replicas}" if replicas > 1 else ""))
    for layer in range(spec.n_layers):
        _place_trunk(program, residency, report, layer, trunks[layer])
    extra = 0
    for layer in range(spec.n_layers):
        freq = None if frequencies is None else frequencies[layer]
        co = None if coactivation_matrix is None else coactivation_matrix[layer]
        copies = [1] * spec.n_experts
        if replicas > 1:
            ranked = sorted(range(spec.n_experts),
                            key=lambda e: (-(freq[e] if freq is not None else 0.0), e))  # fmt: skip
            for e in ranked[: round(replicate_top * spec.n_experts)]:
                copies[e] = min(replicas, len(modules))
        order = list(modules)
        if spec.n_experts < len(modules):
            # fewer experts than modules (dense models): rotate so layers spread over modules
            shift = layer % len(modules)
            order = order[shift:] + order[:shift]
        placed = _assign_experts(spec.n_experts, order, policy, freq, co, copies)
        for e, parts in enumerate(program.expert_parts[layer]):
            for i, device in enumerate(placed[e]):
                for tensor in parts:
                    _place(residency, report, tensor, device, owner=i == 0, fmt=cold_format)
                extra += i > 0
    if extra:
        report.notes.append(f"{extra} expert replicas across modules")
    _place_head(program, residency, report, trunks[-1], modules, head)
    if prefill:
        filled = _prefill(program, residency, report, trunks[0], prefill)
        report.notes.append(f"prefilled {filled} experts on the trunk device")
    program.sample_device = trunks[0]
    return report


def place_tensor_parallel(
    program: ModelProgram,
    residency: ResidencyMap,
    trunk: int,
    modules: Sequence[int],
    *,
    head: str = "auto",
) -> PlacementReport:
    """Every expert partitioned across all modules: shard ``j`` of each expert on module ``j``.

    Each module then holds an equal slice of every expert, so every selected
    expert uses all modules at once and the load is balanced by construction.
    The program must have been built with ``expert_shards == len(modules)``.
    """
    parts = len(program.shard_bounds)
    if parts != len(modules):
        raise PlacementError(f"program has {parts} expert shards for {len(modules)} modules")
    report = PlacementReport("tensor_parallel")
    for layer in range(program.spec.n_layers):
        _place_trunk(program, residency, report, layer, trunk)
        for shards in program.expert_parts[layer]:
            for tensor, device in zip(shards, modules):
                _place(residency, report, tensor, device)
    _place_head(program, residency, report, trunk, modules, head)
    program.sample_device = trunk
    return report


def _prefill(program: ModelProgram, residency: ResidencyMap, report: PlacementReport, device: int,
             prefill: Sequence[tuple[int, int]]) -> int:  # fmt: skip
    """Replicate whole experts onto ``device`` in priority order until it is full."""
    registry = residency.registry
    filled = 0
    for layer, expert in prefill:
        parts = program.expert_parts[layer][expert]
        if all(residency.copy(t, device) is not None for t in parts):
            continue
        if residency.free_bytes(device) < sum(registry.spec(t).nbytes + 64 for t in parts):
            break
        for tensor in parts:
            _place(residency, report, tensor, device, owner=False)
        filled += 1
    return filled


def _place_head(program: ModelProgram, residency: ResidencyMap, report: PlacementReport, trunk: int,
                modules: Sequence[int], head: str) -> None:  # fmt: skip
    """Output projection: row shards on the modules, a full copy on the trunk device, or both."""
    if head not in ("auto", "modules", "trunk"):
        raise PlacementError(f"unknown head placement {head!r}")
    if head != "trunk" and len(program.head_w) > 1:
        for i, tensor in enumerate(program.head_w):
            _place(residency, report, tensor, modules[i % len(modules)], pinned=True)
        need = sum(residency.registry.spec(t).nbytes + 64 for t in program.head_w)
        if head == "auto" and trunk not in modules and residency.free_bytes(trunk) >= need:
            for tensor in program.head_w:
                _place(residency, report, tensor, trunk, pinned=True, owner=False)
            report.notes.append("output projection on the modules and on the trunk device")
    else:
        for tensor in program.head_w:
            _place(residency, report, tensor, trunk, pinned=True)


def hottest_experts(frequencies: np.ndarray) -> list[tuple[int, int]]:
    """(layer, expert) pairs in descending selection frequency (ties: layer, expert)."""
    n_layers, n_experts = frequencies.shape
    pairs = [(layer, e) for layer in range(n_layers) for e in range(n_experts)]
    return sorted(pairs, key=lambda p: (-frequencies[p[0]][p[1]], p[0], p[1]))
