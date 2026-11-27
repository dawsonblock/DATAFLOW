"""Eight competing dataflow architectures in one simulator.

Each architecture is a rule for placing a model on a topology plus an execution
policy.  None is privileged: they run the same workload on the same simulator
and are compared on speed, energy and cost.

| key | architecture     | core concept                                                        |
|-----|------------------|---------------------------------------------------------------------|
| A   | Centralized      | conventional CPU/GPU execution with per-expert caching              |
| B   | Distributed      | independent compute-memory modules, no GPU in the loop              |
| C   | Pipeline         | consecutive layers assigned to sequential processing stages         |
| D   | Expert-local     | whole experts stay on their modules; trunk on the GPU               |
| E   | Tensor-parallel  | every expert partitioned across all modules                         |
| F   | Adaptive hybrid  | all execution strategies combined dynamically                       |
| G   | Spatial dataflow | persistent operators and activation streams, no host in the loop    |
| H   | Hierarchical     | adaptive hybrid on a fast local fabric with a slow path to the host |
"""

from __future__ import annotations

from dataclasses import dataclass

from basp.core.capabilities import DeviceKind
from basp.core.topology import Topology
from basp.dataflow.base import (
    ALL_MODES,
    CachePolicy,
    DataflowMode,
    ExecutionPolicy,
    PrefetchPolicy,
)
from basp.experiments.configurations import Configuration
from basp.scheduler import placement
from basp.workloads.trace import RoutingTrace

M = DataflowMode


@dataclass(frozen=True)
class Architecture:
    key: str
    name: str
    concept: str
    needs_modules: bool
    uses_gpu: bool


ARCHITECTURES: dict[str, Architecture] = {
    a.key: a
    for a in (
        Architecture("A", "Centralized", "conventional CPU/GPU execution", False, True),
        Architecture("B", "Distributed", "independent compute-memory modules", True, False),
        Architecture("C", "Pipeline", "layers assigned to sequential processing stages", True, False),
        Architecture("D", "Expert-local", "whole experts stay on their assigned modules", True, True),
        Architecture("E", "Tensor-parallel", "individual tensors partitioned across modules", True, True),
        Architecture("F", "Adaptive hybrid", "dynamically combines execution strategies", True, True),
        Architecture("G", "Spatial dataflow", "persistent operators connected by activation streams",
                     True, True),
        Architecture("H", "Hierarchical", "local clusters with fast and slow interconnects", True, True),
    )
}  # fmt: skip


class ArchitectureError(ValueError):
    """The architecture cannot be built on the given topology."""


def _devices(topology: Topology):
    host = topology.host
    gpus = [d.device_id for d in topology.of_kind(DeviceKind.GPU)]
    modules = [d.device_id for d in topology.of_kind(DeviceKind.NEAR_MEMORY) if d.caps.can_compute]
    return host, (gpus[0] if gpus else None), modules


def place_pipeline(program, residency, stages) -> placement.PlacementReport:
    """Consecutive layers on consecutive stages; a layer's trunk and experts stay together."""
    spec = program.spec
    report = placement.PlacementReport("pipeline")
    for layer in range(spec.n_layers):
        stage = stages[layer * len(stages) // spec.n_layers]
        placement._place_trunk(program, residency, report, layer, stage)
        for tensor in program.layer_expert_tensors(layer):
            placement._place(residency, report, tensor, stage)
    for tensor in program.head_w:
        placement._place(residency, report, tensor, stages[-1], pinned=True)
    program.sample_device = stages[0]
    return report


def configure(
    key: str,
    topology: Topology,
    prior: RoutingTrace,
    *,
    replicas: int = 1,
    joint_rounds: int = 2,
    energy_weight: float = 0.0,
) -> Configuration:
    """Build architecture ``key`` on ``topology``.

    ``prior`` supplies profiling statistics (expert frequencies and
    co-activation) for placement.  ``replicas`` > 1 keeps that many copies of
    every expert on distinct modules where the architecture allows it.
    """
    if key not in ARCHITECTURES:
        raise ArchitectureError(f"unknown architecture {key!r}")
    arch = ARCHITECTURES[key]
    host, gpu, modules = _devices(topology)
    if arch.needs_modules and not modules:
        raise ArchitectureError(f"architecture {key} needs compute-memory modules")
    if key == "A" and gpu is None:
        raise ArchitectureError("architecture A needs a GPU")
    if key == "H" and not any(d.kind is DeviceKind.SWITCH for d in topology.devices.values()):
        raise ArchitectureError("architecture H needs a local fabric")
    freq = prior.expert_frequencies()
    co = placement.coactivation(prior)
    hot = placement.hottest_experts(freq)
    trunk = gpu if gpu is not None else host
    copies = max(1, min(replicas, len(modules)))
    common = {"joint_rounds": joint_rounds, "energy_weight_s_per_j": energy_weight}
    head_shards = max(len(modules), 1)
    expert_shards = 1

    def on_modules(**kw):
        return lambda p, r: placement.place_modules(
            p, r, trunk, modules, policy="coactivation", frequencies=freq, coactivation_matrix=co,
            replicas=copies, **kw,
        )  # fmt: skip

    if key == "A":
        policy = ExecutionPolicy("A", frozenset({M.WEIGHT_STATIONARY, M.ACTIVATION_STATIONARY}),
                                 frozenset({host, gpu}), cache=CachePolicy("benefit", (gpu,)), **common)  # fmt: skip
        place = lambda p, r: placement.place_gpu_host(p, r, gpu, host, prefill=hot)
        head_shards = 1
    elif key == "B":
        policy = ExecutionPolicy("B", frozenset({M.WEIGHT_STATIONARY, M.DISTRIBUTED, M.OUTPUT_STATIONARY}),
                                 frozenset(modules), **common)  # fmt: skip
        trunks = [modules[layer % len(modules)] for layer in range(prior.n_layers)]
        place = lambda p, r: placement.place_modules(
            p, r, trunks, modules, policy="coactivation", frequencies=freq, coactivation_matrix=co,
            replicas=copies, head="modules",
        )  # fmt: skip
    elif key == "C":
        policy = ExecutionPolicy("C", frozenset({M.WEIGHT_STATIONARY}), frozenset(modules),
                                 host_in_loop=False, **common)  # fmt: skip
        place = lambda p, r: place_pipeline(p, r, modules)
        head_shards = 1
    elif key == "D":
        policy = ExecutionPolicy("D", frozenset({M.WEIGHT_STATIONARY, M.DISTRIBUTED}),
                                 frozenset({trunk, *modules}), **common)  # fmt: skip
        place = on_modules()
    elif key == "E":
        policy = ExecutionPolicy("E", frozenset({M.WEIGHT_STATIONARY, M.DISTRIBUTED, M.OUTPUT_STATIONARY}),
                                 frozenset({trunk, *modules}), **common)  # fmt: skip
        place = lambda p, r: placement.place_tensor_parallel(p, r, trunk, modules)
        expert_shards = len(modules)
    elif key in ("F", "H"):
        if gpu is None:
            raise ArchitectureError(f"architecture {key} caches experts on a GPU")
        policy = ExecutionPolicy(key, ALL_MODES, frozenset({host, gpu, *modules}),
                                 cache=CachePolicy("benefit", (gpu,)), prefetch=PrefetchPolicy(),
                                 **common)  # fmt: skip
        place = on_modules(prefill=hot)
    else:  # G
        policy = ExecutionPolicy(
            "G", frozenset({M.WEIGHT_STATIONARY, M.ROW_STATIONARY, M.OUTPUT_STATIONARY, M.DISTRIBUTED}),
            frozenset({trunk, *modules}), host_in_loop=False, eager_activations=True, **common,
        )  # fmt: skip
        place = on_modules()
    return Configuration(key, arch.name, arch.concept, topology, policy, place,
                         head_shards=head_shards, expert_shards=expert_shards, baseline="A")  # fmt: skip
