"""Shared helpers for the v0.2 experiments: running one architecture on one design."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from basp.backends.virtual.runtime import RunResult
from basp.core.tensor import CompressionFormat, FormatTable
from basp.core.topology import Topology
from basp.dse.hardware import HardwareDesign
from basp.experiments.architectures import configure
from basp.experiments.configurations import Configuration
from basp.experiments.harness import Setup
from basp.experiments.profiles import load_json, load_topology
from basp.verification import check_accounting
from basp.workloads.model import MoEModelSpec
from basp.workloads.trace import RoutingStep, RoutingTrace, synth_trace

GiB = 1024**3


@dataclass
class Context:
    """What every experiment runs on, beyond the hardware design itself.

    * ``base_profile`` -- the host/GPU profile designs are built on (a calibrated machine
      profile once one exists; an illustrative desktop until then);
    * ``trace`` -- recorded routing (``RoutingTrace``) used instead of synthetic routing for
      every model whose shape it matches;
    * ``formats`` -- measured bits per weight for stream formats (from a compression
      profile), replacing the assumed ratios.
    """

    base_profile: str = "desktop_gpu12"
    trace: RoutingTrace | None = None
    trace_path: str | None = None
    formats: dict[CompressionFormat, float] = field(default_factory=dict)
    compression_path: str | None = None


_CONTEXT = Context()


def context() -> Context:
    return _CONTEXT


def set_context(new: Context) -> Context:
    """Replace the experiment context; returns the previous one (restore it when done)."""
    global _CONTEXT
    load_topology(new.base_profile)  # fail now, not in the middle of a report
    previous, _CONTEXT = _CONTEXT, new
    return previous


def set_base_profile(name: str) -> str:
    """Select the host/GPU profile every design is built on; returns the previous selection."""
    previous = _CONTEXT.base_profile
    set_context(replace(_CONTEXT, base_profile=name))
    return previous


def base_profile() -> str:
    return _CONTEXT.base_profile


def load_compression_profile(path: str | Path) -> dict[CompressionFormat, float]:
    """Bits per weight of the ENTROPY format as measured by ``benchmarks.profile_compression``.

    The best lossless codec actually achieved is used (not the order-0 entropy bound), since
    only a working codec can stream weights.
    """
    data = json.loads(Path(path).read_text())
    if data.get("format") != "basp.compression_profile":
        raise ValueError(f"{path} is not a compression profile written by benchmarks.profile_compression")
    codecs = data["summary"]["codecs"]
    best = min(c["bits_per_weight"] for c in codecs.values())
    raw = data["summary"]["raw_bits_per_weight"]
    return {CompressionFormat.ENTROPY: min(best, raw)}


def trace_matches(trace: RoutingTrace, spec: MoEModelSpec) -> bool:
    return (trace.n_layers, trace.n_experts, trace.top_k) == (spec.n_layers, spec.n_experts, spec.top_k)


def routing_source(spec: MoEModelSpec) -> str:
    ctx = _CONTEXT
    if ctx.trace is not None and trace_matches(ctx.trace, spec):
        return f"recorded ({Path(ctx.trace_path).name if ctx.trace_path else 'trace'})"
    return "synthetic"


def _recorded(trace: RoutingTrace, prior_steps: int, tokens: int, batch: int) -> tuple[RoutingTrace, RoutingTrace] | None:
    """(profiling, evaluation) traces cut from a recorded single-stream trace, or ``None``.

    Batch row ``r`` follows its own disjoint window of the recording, so rows do not share
    routing.  The profiling part precedes every evaluation window.
    """
    if trace.batch != 1:
        return None
    need = prior_steps + batch * tokens
    if len(trace.steps) < need:
        return None
    prior = RoutingTrace(trace.n_layers, trace.n_experts, trace.top_k, 1, trace.steps[:prior_steps])
    if batch > 1:
        prior = RoutingTrace(trace.n_layers, trace.n_experts, trace.top_k, batch,
                             [[RoutingStep(st.experts * batch, st.coeffs * batch) for st in step]
                              for step in prior.steps])  # fmt: skip
    held_steps = []
    for t in range(tokens):
        layers = []
        for layer in range(trace.n_layers):
            rows = [trace.steps[prior_steps + r * tokens + t][layer] for r in range(batch)]
            layers.append(RoutingStep(tuple(x.experts[0] for x in rows), tuple(x.coeffs[0] for x in rows)))
        held_steps.append(layers)
    held = RoutingTrace(trace.n_layers, trace.n_experts, trace.top_k, batch, held_steps)
    prior.validate()
    held.validate()
    return prior, held


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
    """(profiling trace, held-out evaluation trace).

    Cut from the recorded routing in the experiment context when it matches the model and is
    long enough; otherwise from one seeded run of the synthetic generator.
    """
    ctx = _CONTEXT
    if ctx.trace is not None and trace_matches(ctx.trace, spec):
        recorded = _recorded(ctx.trace, prior_steps, tokens, batch)
        if recorded is not None:
            return recorded
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
    formats: Mapping[CompressionFormat, float] | None = None,
    prompt: bool = False,
    topology_edit: Callable[[Topology], Topology] | None = None,
    config_edit: Callable[[Configuration], Configuration] | None = None,
    estimates: Mapping[str, Any] | None = None,
) -> Outcome:
    """Simulate architecture ``key`` on ``design`` and verify the run's accounting."""
    estimates = estimates or load_json("design_estimates")
    topology = design.topology(load_topology(base_profile or _CONTEXT.base_profile), estimates, corner)
    if topology_edit is not None:
        topology = topology_edit(topology)
    config = configure(key, topology, prior, replicas=replicas)
    if config_edit is not None:
        config = config_edit(config)
    overrides = dict(_CONTEXT.formats) if formats is None else dict(formats)
    if overrides:
        if config.formats is not None:
            raise ValueError("measured stream formats cannot be combined with a configuration's own formats")
        config = replace(config, formats=FormatTable(overrides))
    setup = config.build(spec, prior, batch=batch, context_len=context_len, max_new_tokens=tokens,
                         prompt=prompt)  # fmt: skip
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
