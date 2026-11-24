"""The eight execution configurations compared before any hardware is built.

| key          | research purpose                    |
|--------------|-------------------------------------|
| gpu_only     | conventional reference              |
| cpu_only     | high-capacity memory reference      |
| static_split | standard layer offloading           |
| adaptive     | dynamic scheduling benefit          |
| distributed  | near-memory benefit                 |
| predictive   | latency-hiding benefit              |
| compression  | reduced memory traffic              |
| complete     | combined BASP dataflow              |

Configurations 1-4, 6 and 7 use the CPU+GPU profile; 5 and 8 add compute-memory
modules.  Reports must state that the module configurations have more memory
and compute installed: this is a comparison of dataflows on their natural
hardware, not an equal-cost comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from basp.core.capabilities import DeviceKind
from basp.core.tensor import CompressionFormat, FormatTable
from basp.core.topology import Topology
from basp.dataflow.base import (
    ALL_MODES,
    CachePolicy,
    DataflowMode,
    ExecutionPolicy,
    PrefetchPolicy,
)
from basp.experiments.harness import Placer, Setup, build
from basp.scheduler import placement
from basp.workloads.model import MoEModelSpec
from basp.workloads.trace import RoutingTrace

M = DataflowMode


@dataclass
class Configuration:
    key: str
    title: str
    purpose: str
    topology: Topology
    policy: ExecutionPolicy
    place: Placer
    head_shards: int = 1
    baseline: str | None = None  # configuration this one is meant to be compared with
    formats: FormatTable | None = None

    def build(self, spec: MoEModelSpec, prior: RoutingTrace | None, **kwargs) -> Setup:
        return build(self.topology, spec, self.policy, self.place, head_shards=self.head_shards,
                     formats=self.formats, prior=prior, **kwargs)  # fmt: skip


def _one(topology: Topology, kind: DeviceKind) -> int:
    found = topology.of_kind(kind)
    if not found:
        raise ValueError(f"topology {topology.name!r} has no {kind.name} device")
    return found[0].device_id


def standard_configurations(
    desktop: Topology,
    basp: Topology,
    prior: RoutingTrace,
    *,
    formats: FormatTable | None = None,
    joint_rounds: int = 2,
) -> list[Configuration]:
    """Build the eight configurations.  ``prior`` supplies profiling statistics
    (expert frequencies) for placement and cache pre-fill; evaluate on a
    different trace."""
    freq = prior.expert_frequencies()
    hot = placement.hottest_experts(freq)
    natural = [(layer, e) for layer in range(prior.n_layers) for e in range(prior.n_experts)]

    host, gpu = _one(desktop, DeviceKind.CPU), _one(desktop, DeviceKind.GPU)
    b_host, b_gpu = _one(basp, DeviceKind.CPU), _one(basp, DeviceKind.GPU)
    modules = [d.device_id for d in basp.of_kind(DeviceKind.NEAR_MEMORY)]
    if not modules:
        raise ValueError(f"topology {basp.name!r} has no compute-memory modules")

    lru = CachePolicy("lru", (gpu,), background=False)
    benefit = CachePolicy("benefit", (gpu,))
    b_benefit = CachePolicy("benefit", (b_gpu,))

    def policy(name, modes, devices, **kw) -> ExecutionPolicy:
        return ExecutionPolicy(name, frozenset(modes), frozenset(devices), joint_rounds, **kw)

    gpu_only = Configuration(
        "gpu_only", "GPU-only", "conventional reference", desktop,
        policy("gpu_only", {M.ACTIVATION_STATIONARY}, {gpu}, cache=lru),
        lambda p, r: placement.place_gpu_host(p, r, gpu, host, prefill=natural),
        formats=formats,
    )  # fmt: skip
    cpu_only = Configuration(
        "cpu_only", "CPU-only", "high-capacity memory reference", desktop,
        policy("cpu_only", {M.WEIGHT_STATIONARY}, {host}),
        lambda p, r: placement.place_single(p, r, host),
        formats=formats,
    )  # fmt: skip
    static_split = Configuration(
        "static_split", "Static CPU/GPU", "standard layer offloading", desktop,
        policy("static_split", {M.WEIGHT_STATIONARY}, {host, gpu}),
        lambda p, r: placement.place_static_split(p, r, gpu, host),
        formats=formats,
    )  # fmt: skip
    adaptive = Configuration(
        "adaptive", "Adaptive CPU/GPU", "dynamic scheduling benefit", desktop,
        policy("adaptive", {M.WEIGHT_STATIONARY, M.ACTIVATION_STATIONARY}, {host, gpu}, cache=benefit),
        lambda p, r: placement.place_gpu_host(p, r, gpu, host, prefill=hot),
        baseline="static_split", formats=formats,
    )  # fmt: skip
    distributed = Configuration(
        "distributed", "Distributed memory compute", "near-memory benefit", basp,
        policy("distributed", {M.WEIGHT_STATIONARY, M.DISTRIBUTED}, {b_gpu, *modules}),
        lambda p, r: placement.place_modules(p, r, b_gpu, modules, frequencies=freq),
        head_shards=len(modules), baseline="static_split", formats=formats,
    )  # fmt: skip
    predictive = Configuration(
        "predictive", "Predictive dataflow", "latency-hiding benefit", desktop,
        policy("predictive", {M.WEIGHT_STATIONARY, M.ACTIVATION_STATIONARY, M.PREDICTIVE},
               {host, gpu}, cache=benefit, prefetch=PrefetchPolicy()),
        lambda p, r: placement.place_gpu_host(p, r, gpu, host, prefill=hot),
        baseline="adaptive", formats=formats,
    )  # fmt: skip
    compression = Configuration(
        "compression", "Compression-aware dataflow", "reduced memory traffic", desktop,
        policy("compression", {M.ACTIVATION_STATIONARY}, {gpu}, cache=lru),
        lambda p, r: placement.place_gpu_host(p, r, gpu, host, prefill=natural,
                                              cold_format=CompressionFormat.ENTROPY),
        baseline="gpu_only", formats=formats,
    )  # fmt: skip
    complete = Configuration(
        "complete", "Complete BASP", "combined performance", basp,
        policy("complete", ALL_MODES, {b_host, b_gpu, *modules}, cache=b_benefit,
               prefetch=PrefetchPolicy()),
        lambda p, r: placement.place_modules(p, r, b_gpu, modules, frequencies=freq, prefill=hot),
        head_shards=len(modules), baseline="adaptive", formats=formats,
    )  # fmt: skip
    return [gpu_only, cpu_only, static_split, adaptive, distributed, predictive, compression,
            complete]  # fmt: skip


def with_modes(config: Configuration, modes, name: str | None = None) -> Configuration:
    """Copy of ``config`` restricted to (or extended with) the given dataflow modes."""
    policy = replace(config.policy, modes=frozenset(modes), name=name or config.policy.name)
    return replace(config, policy=policy, key=name or config.key)
