"""Shared helpers for the v0.2 experiments: running one architecture on one design."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from basp.backends.virtual.runtime import RunResult
from basp.core.topology import Topology
from basp.dse.hardware import HardwareDesign
from basp.experiments.architectures import configure
from basp.experiments.configurations import Configuration
from basp.experiments.harness import Setup
from basp.experiments.profiles import load_json, load_topology
from basp.verification import check_accounting
from basp.workloads.model import MoEModelSpec
from basp.workloads.trace import RoutingTrace, synth_trace

GiB = 1024**3
_BASE_PROFILE = "desktop_gpu12"


def set_base_profile(name: str) -> str:
    """Select the host/GPU profile every design is built on; returns the previous selection.

    The default is an illustrative desktop.  After calibrating a real machine,
    select its profile so the comparison runs on measured host and GPU parameters.
    """
    global _BASE_PROFILE
    load_topology(name)  # fail now, not in the middle of a report
    previous, _BASE_PROFILE = _BASE_PROFILE, name
    return previous


def base_profile() -> str:
    return _BASE_PROFILE


@dataclass
class Outcome:
    result: RunResult
    setup: Setup
    design: HardwareDesign
    cost_usd: float

    @property
    def tokens_per_s(self) -> float:
        return self.result.tokens_per_s

    @property
    def per_kusd(self) -> float:
        return 1e3 * self.result.tokens_per_s / self.cost_usd


def traces(spec: MoEModelSpec, *, tokens: int, prior_steps: int = 32, batch: int = 1,
           zipf_s: float = 0.8, stickiness: float = 0.6, seed: int = 7) -> tuple[RoutingTrace, RoutingTrace]:  # fmt: skip
    """(profiling trace, held-out evaluation trace) from one seeded synthetic generator run."""
    trace = synth_trace(prior_steps + tokens, spec.n_layers, spec.n_experts, spec.top_k, batch=batch,
                        zipf_s=zipf_s, stickiness=stickiness, seed=seed)  # fmt: skip
    return trace.split(prior_steps)


def scale_latency(topology: Topology, scale: float) -> Topology:
    """Every link latency, kernel start-up time and the host sync latency multiplied by ``scale``."""
    data = topology.to_dict()
    data["sync_latency_s"] *= scale
    for link in data["links"]:
        link["latency_s"] *= scale
    for dev in data["devices"]:
        dev["kernel_startup_s"] *= scale
        if dev.get("warm_startup_s") is not None:
            dev["warm_startup_s"] *= scale
    return Topology.from_dict(data)


def edit_devices(topology: Topology, edit: Callable[[dict[str, Any]], None]) -> Topology:
    """Copy of ``topology`` with ``edit`` applied to every device dictionary."""
    data = topology.to_dict()
    for dev in data["devices"]:
        edit(dev)
    return Topology.from_dict(data)


def run_architecture(
    key: str,
    design: HardwareDesign,
    spec: MoEModelSpec,
    prior: RoutingTrace,
    held: RoutingTrace,
    *,
    tokens: int = 6,
    warmup_tokens: int = 2,
    context_len: int = 2048,
    batch: int = 1,
    concurrent: int = 0,
    replicas: int = 1,
    corner: str = "nominal",
    base_profile: str | None = None,
    keep_log: bool = False,
    topology_edit: Callable[[Topology], Topology] | None = None,
    config_edit: Callable[[Configuration], Configuration] | None = None,
    estimates: Mapping[str, Any] | None = None,
) -> Outcome:
    """Simulate architecture ``key`` on ``design`` and verify the run's accounting."""
    estimates = estimates or load_json("design_estimates")
    topology = design.topology(load_topology(base_profile or _BASE_PROFILE), estimates, corner)
    if topology_edit is not None:
        topology = topology_edit(topology)
    config = configure(key, topology, prior, replicas=replicas)
    if config_edit is not None:
        config = config_edit(config)
    setup = config.build(spec, prior, batch=batch, context_len=context_len, max_new_tokens=tokens)
    if concurrent:
        result = setup.run_concurrent(held, streams=concurrent, tokens=tokens,
                                      warmup_tokens=warmup_tokens, keep_log=keep_log)  # fmt: skip
    else:
        result = setup.run(held, tokens=tokens, warmup_tokens=warmup_tokens, keep_log=keep_log)
    if not result.ok:
        raise RuntimeError(f"architecture {key}: {result.failures[:2]} {result.violations[:2]}")
    problems = check_accounting(setup.system)
    if problems:
        raise RuntimeError(f"architecture {key}: accounting check failed: {problems[:2]}")
    return Outcome(result, setup, design, design.cost_usd(estimates, corner))


def with_policy(**changes) -> Callable[[Configuration], Configuration]:
    """Configuration edit that replaces fields of the execution policy."""
    return lambda config: replace(config, policy=replace(config.policy, **changes))


def design_for(spec: MoEModelSpec, *, modules: int = 4, gpu_gib: int = 12, replicas: int = 1,
               bits_per_weight: float = 4.5, **overrides) -> HardwareDesign:  # fmt: skip
    """Smallest standard design whose host and modules can hold ``spec``."""
    model_gib = spec.bytes_at(bits_per_weight) / GiB
    host_gib = max(64, 64 * -(-int(model_gib * 1.15 + 1) // 64))
    params: dict[str, Any] = {"gpu_gib": gpu_gib, "modules": modules, "host_gib": host_gib}
    if modules:
        need = model_gib * replicas * 1.15 / modules
        size = next(s for s in (16, 32, 64, 128, 256, 512, 1024) if s >= need)
        params.update(module_gib=size, channels=max(2, -(-size // 128)))
    params.update(overrides)
    return HardwareDesign(**params)
