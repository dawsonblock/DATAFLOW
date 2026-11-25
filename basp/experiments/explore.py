"""Architectural design-space exploration with the simulator in the loop.

Candidate systems are generated from a :class:`DesignSpace` (architecture,
GPU memory, number of modules and their memory, channels, MAC lanes and link
bandwidth, expert replication), built into a topology, simulated on a workload
and scored on three objectives: throughput, energy efficiency and estimated
cost.  The search returns the Pareto front; nothing favours the architecture
originally proposed -- conventional CPU/GPU systems are candidates like any other.

Cost and all module parameters are estimates (see ``configs/design_estimates.json``).
The search runs at the nominal corner; designs on the front are re-evaluated at
the pessimistic and optimistic corners so every figure carries a range.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from basp.core.tensor import CompressionFormat, FormatTable
from basp.dse.hardware import CORNERS, DesignError, HardwareDesign
from basp.dse.pareto import Objective
from basp.dse.space import DesignSpace, Param
from basp.experiments.architectures import ArchitectureError, configure
from basp.experiments.profiles import load_json, load_topology
from basp.experiments.variants import _recorded, trace_matches
from basp.scheduler.placement import PlacementError
from basp.scheduler.scheduler import SchedulingError
from basp.workloads.trace import RoutingTrace, synth_trace
from basp.workloads.zoo import MODELS

OBJECTIVES = (
    Objective("tokens_per_s", maximize=True),
    Objective("tokens_per_joule", maximize=True),
    Objective("cost_usd", maximize=False),
)

DEFAULT_WORKLOAD: dict[str, Any] = {
    "model": "qwen3-30b-a3b", "tokens": 4, "warmup_tokens": 2, "context_len": 2048,
    "trace": {"zipf_s": 0.8, "stickiness": 0.6, "seed": 7, "prior_steps": 24},
}  # fmt: skip


def design_space(model_gib: float, *, gpu_sizes=(0, 12, 24, 32), module_counts=(0, 2, 4, 8),
                 module_sizes=(16, 32, 64), channels=(1, 2, 4), lanes=(256, 512, 1024, 2048),
                 links=(8.0, 16.0, 32.0), host_gib: int = 64) -> DesignSpace:  # fmt: skip
    """Design space with the constraints that keep candidates physically meaningful."""

    def consistent(c: Mapping[str, Any]) -> bool:
        arch, modules = c["arch"], c["modules"]
        if arch == "A":
            # one canonical point per GPU size: module parameters are meaningless here
            return (modules == 0 and c["gpu_gib"] > 0 and c["replicas"] == 1
                    and c["module_gib"] == module_sizes[0] and c["channels"] == channels[0]
                    and c["lanes"] == lanes[0] and c["link_gbps"] == links[0]
                    and model_gib * 1.05 <= host_gib)  # fmt: skip
        if modules == 0 or c["replicas"] > modules:
            return False
        if arch in ("B", "C") and c["gpu_gib"] != 0:
            return False  # these never use a GPU: do not pay for one
        if arch in ("F", "H") and c["gpu_gib"] == 0:
            return False  # these cache experts on a GPU
        if arch in ("C", "E") and c["replicas"] != 1:
            return False
        if c["module_gib"] > 128 * c["channels"]:
            return False
        # every expert copy must fit on the modules (with allocator and scratch head-room)
        return modules * c["module_gib"] >= model_gib * c["replicas"] * 1.1

    params = [
        Param("arch", ("A", "B", "C", "D", "E", "F", "G", "H"), ordered=False),
        Param("gpu_gib", tuple(gpu_sizes)),
        Param("modules", tuple(module_counts)),
        Param("module_gib", tuple(module_sizes)),
        Param("channels", tuple(channels)),
        Param("lanes", tuple(lanes)),
        Param("link_gbps", tuple(links)),
        Param("replicas", (1, 2)),
    ]
    return DesignSpace(params, [consistent])


def seed_designs(space: DesignSpace) -> list[dict[str, Any]]:
    """Starting designs: every conventional system and one standard system per architecture.

    A search that starts from random designs can miss the conventional baseline
    entirely (it is a handful of points in thousands); seeding guarantees every
    architecture is represented and judged on the same footing.
    """
    values = {p.name: p.values for p in space.params}
    seeds: list[dict[str, Any]] = []
    for gpu in values["gpu_gib"]:
        seeds.append({"arch": "A", "gpu_gib": gpu, "modules": 0, "module_gib": values["module_gib"][0],
                      "channels": values["channels"][0], "lanes": values["lanes"][0],
                      "link_gbps": values["link_gbps"][0], "replicas": 1})  # fmt: skip
    for arch in ("B", "C", "D", "E", "F", "G", "H"):
        for gpu in values["gpu_gib"]:
            for replicas in (1, 2):
                seeds.append({"arch": arch, "gpu_gib": gpu, "modules": 4, "module_gib": 32,
                              "channels": 2, "lanes": 1024, "link_gbps": 16.0, "replicas": replicas})  # fmt: skip
    return [space.canonical(s) for s in seeds if space.feasible(s)]


def to_design(config: Mapping[str, Any], host_gib: int = 64) -> HardwareDesign:
    arch = config["arch"]
    return HardwareDesign(
        gpu_gib=int(config["gpu_gib"]), modules=int(config["modules"]),
        module_gib=int(config["module_gib"]), channels=int(config["channels"]),
        lanes=int(config["lanes"]), link_gbps=float(config["link_gbps"]),
        fabric="local" if arch == "H" else "star", persistent=arch == "G", host_gib=host_gib,
    )  # fmt: skip


@functools.lru_cache(maxsize=8)
def _traces(model: str, tokens: int, zipf_s: float, stickiness: float, seed: int, prior_steps: int,
            batch: int, path: str | None = None):  # fmt: skip
    spec = MODELS[model]
    if path is not None:
        recorded = RoutingTrace.load(path)
        if not trace_matches(recorded, spec):
            raise ValueError(f"trace {path} ({recorded.n_layers} layers, {recorded.n_experts} experts, "
                             f"top-{recorded.top_k}) does not match {spec.name}")  # fmt: skip
        cut = _recorded(recorded, prior_steps, tokens, batch)
        if cut is None:
            raise ValueError(f"trace {path} has {len(recorded.steps)} steps; {prior_steps + batch * tokens} needed")
        return cut
    trace = synth_trace(prior_steps + tokens, spec.n_layers, spec.n_experts, spec.top_k, batch=batch,
                        zipf_s=zipf_s, stickiness=stickiness, seed=seed)  # fmt: skip
    return trace.split(prior_steps)


@functools.lru_cache(maxsize=4)
def _base(profile: str):
    return load_topology(profile), load_json("design_estimates")


def evaluate_design(config: Mapping[str, Any], *, workload: Mapping[str, Any] | None = None,
                    base_profile: str = "desktop_gpu12", corner: str = "nominal",
                    host_gib: int = 64) -> dict[str, Any] | None:  # fmt: skip
    """Simulate one design; ``None`` when it cannot hold or run the model."""
    workload = workload or DEFAULT_WORKLOAD
    spec = MODELS[workload["model"]]
    t = workload["trace"]
    batch = int(workload.get("batch", 1))
    prior, held = _traces(workload["model"], workload["tokens"], t["zipf_s"], t["stickiness"],
                          t["seed"], t["prior_steps"], batch, t.get("path"))  # fmt: skip
    formats = {CompressionFormat[k]: float(v) for k, v in workload.get("formats", {}).items()}
    base, estimates = _base(base_profile)
    try:
        design = to_design(config, host_gib)
        topology = design.topology(base, estimates, corner)
        built = configure(config["arch"], topology, prior, replicas=int(config.get("replicas", 1)))
        if formats:
            built = replace(built, formats=FormatTable(formats))
        setup = built.build(spec, prior, batch=batch, context_len=workload.get("context_len", 0),
                            max_new_tokens=workload["tokens"])  # fmt: skip
        result = setup.run(held, tokens=workload["tokens"], warmup_tokens=workload.get("warmup_tokens", 0))
    except (PlacementError, ArchitectureError, DesignError, SchedulingError):
        return None
    if not result.ok:
        return None
    cost = design.cost_usd(estimates, corner)
    return {
        "tokens_per_s": result.tokens_per_s,
        "tokens_per_joule": result.tokens_per_joule,
        "cost_usd": cost,
        "tokens_per_s_per_kusd": 1e3 * result.tokens_per_s / cost,
        "average_power_w": result.average_power_w,
        "token_latency_ms": 1e3 * result.token_latency_s,
        "link_mb_per_token": result.link_bytes_per_token / 1e6,
    }


def evaluate_corners(config: Mapping[str, Any], **kwargs) -> dict[str, dict[str, Any] | None]:
    """The design at the pessimistic, nominal and optimistic corners of every estimate."""
    return {corner: evaluate_design(config, corner=corner, **kwargs) for corner in CORNERS}
