"""Phase 2 investigations: is each weakness an implementation problem or a limit?

Every function runs a controlled A/B experiment in the simulator on the 30B MoE
workload and returns plain rows (dictionaries) for the report.  All figures are
simulated under the illustrative profile and estimates; the point of each
experiment is the *direction and size* of an effect, not an absolute number.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from basp.analysis import analyse_critical_path, scaling_limit
from basp.dse.hardware import HardwareDesign
from basp.experiments.variants import (
    Outcome,
    edit_devices,
    run_architecture,
    scale_latency,
    traces,
    with_policy,
)
from basp.kernels import quantize_int4, symbol_entropy_bits
from basp.scheduler import placement
from basp.workloads.model import MoEModelSpec

MODULES = HardwareDesign(modules=4)
CENTRAL = HardwareDesign(modules=0)


class Lab:
    """Fixed workload shared by all investigations (one trace, one set of defaults)."""

    def __init__(self, spec: MoEModelSpec, *, tokens: int = 6, warmup_tokens: int = 2,
                 prior_steps: int = 32, context_len: int = 2048) -> None:  # fmt: skip
        self.spec = spec
        self.tokens = tokens
        self.warmup = warmup_tokens
        self.context_len = context_len
        self.prior, self.held = traces(spec, tokens=tokens, prior_steps=prior_steps)

    def run(self, key: str, design: HardwareDesign, **kw) -> Outcome:
        kw.setdefault("tokens", self.tokens)
        kw.setdefault("warmup_tokens", self.warmup)
        kw.setdefault("context_len", self.context_len)
        return run_architecture(key, design, self.spec, self.prior, self.held, **kw)

    def multi(self, key: str, design: HardwareDesign, n: int, *, concurrent: bool, tokens: int = 4,
              **kw) -> Outcome:  # fmt: skip
        prior, held = traces(self.spec, tokens=tokens, prior_steps=16, batch=n, fallback="synthetic")
        if concurrent:
            return run_architecture(key, design, self.spec, prior, held, tokens=tokens,
                                    warmup_tokens=1, context_len=self.context_len, concurrent=n, **kw)  # fmt: skip
        return run_architecture(key, design, self.spec, prior, held, tokens=tokens, warmup_tokens=1,
                                context_len=self.context_len, batch=n, **kw)  # fmt: skip


def _row(investigation: str, variant: str, outcome: Outcome, baseline: Outcome | None = None,
         **extra: Any) -> dict[str, Any]:  # fmt: skip
    row = {
        "investigation": investigation,
        "variant": variant,
        "tokens_per_s": outcome.result.tokens_per_s,
        "tokens_per_joule": outcome.result.tokens_per_joule,
        "average_power_w": outcome.result.average_power_w,
        "link_mb_per_token": outcome.result.link_bytes_per_token / 1e6,
    }
    if baseline is not None:
        row["speedup"] = outcome.result.tokens_per_s / baseline.result.tokens_per_s
    row.update(extra)
    return row


def critical_path(lab: Lab) -> dict[str, Any]:
    """Where a token's time goes in the expert-local architecture, and what scaling can buy."""
    out = lab.run("D", MODULES, keep_log=True)
    report = analyse_critical_path(out.result.op_log, skip_steps=lab.warmup)
    layers = lab.spec.n_layers
    limits = {n: scaling_limit(report, 4, n, layers=layers) for n in (2, 4, 8)}
    return {
        "tokens_per_s": out.result.tokens_per_s,
        "token_time_ms": report.token_time_s * 1e3,
        "by_component_ms": {k: v * 1e3 for k, v in report.by_component.items()},
        "by_kind_ms": {k: v * 1e3 for k, v in report.by_kind.items()},
        "serial_ms": report.serial_s * 1e3,
        "expert_phase_ms": report.expert_phase_s * 1e3,
        "expert_work_ms": report.expert_work_s * 1e3,
        "expert_ops_per_token": report.expert_ops,
        "ideal_efficiency": {str(n): limits[n]["ideal_efficiency"] for n in limits},
        "ideal_tokens_per_s": {str(n): 1.0 / limits[n]["ideal_token_time_s"] for n in limits},
        "residual_ms": report.residual_s * 1e3,
    }


def imbalance(lab: Lab) -> list[dict[str, Any]]:
    """Expert placement and replication against the one-module baseline."""
    one = lab.run("D", HardwareDesign(modules=1))
    freq = lab.prior.expert_frequencies()

    def placer(policy: str):
        def edit(config):
            trunk = config.topology.by_name("gpu").device_id
            mods = [d.device_id for d in config.topology.devices.values()
                    if d.name.startswith("mod") and d.caps.can_compute]  # fmt: skip
            co = placement.coactivation(lab.prior) if policy == "coactivation" else None
            return replace(config, place=lambda p, r: placement.place_modules(
                p, r, trunk, mods, policy=policy, frequencies=freq, coactivation_matrix=co))  # fmt: skip
        return edit

    variants = [
        ("round-robin placement", lab.run("D", MODULES, config_edit=placer("round_robin"))),
        ("frequency-balanced placement", lab.run("D", MODULES, config_edit=placer("balanced"))),
        ("co-activation-aware placement", lab.run("D", MODULES)),
        ("co-activation-aware, every expert on 2 modules", lab.run("D", MODULES, replicas=2)),
        ("tensor-parallel (every expert on all modules)", lab.run("E", MODULES)),
    ]
    rows = [_row("expert imbalance", "one module (baseline)", one, modules=1, efficiency=1.0)]
    for name, out in variants:
        speedup = out.result.tokens_per_s / one.result.tokens_per_s
        rows.append(_row("expert imbalance", name, out, one, modules=4, efficiency=speedup / 4))
    return rows


def scaling(lab: Lab, counts=(1, 2, 4, 8)) -> list[dict[str, Any]]:
    """Throughput against module count for the three module-centric architectures."""
    rows = []
    base: dict[str, float] = {}
    for n in counts:
        for key, replicas, label in (("D", 1, "D expert-local"), ("D", 2, "D with 2 replicas"),
                                     ("E", 1, "E tensor-parallel")):  # fmt: skip
            if replicas > n:
                continue
            out = lab.run(key, HardwareDesign(modules=n), replicas=replicas)
            if n == counts[0]:
                base.setdefault(key, out.result.tokens_per_s)  # the architecture's smallest system
            ref = base.get(key, out.result.tokens_per_s)
            rows.append({"architecture": label, "modules": n, "tokens_per_s": out.result.tokens_per_s,
                         "speedup": out.result.tokens_per_s / ref,
                         "efficiency": out.result.tokens_per_s / ref / (n / counts[0])})  # fmt: skip
    return rows


def synchronisation(lab: Lab) -> list[dict[str, Any]]:
    """Host round trips versus dependency events, and the lock-step batch barrier."""
    rows = []
    for scale in (1.0, 8.0):
        edit = (lambda t, s=scale: scale_latency(t, s)) if scale != 1.0 else None
        host = lab.run("D", MODULES, topology_edit=edit)
        free = lab.run("D", MODULES, topology_edit=edit, config_edit=with_policy(host_in_loop=False))
        tag = f"latencies x{scale:g}"
        rows.append(_row("synchronisation", f"host in the loop ({tag})", host))
        rows.append(_row("synchronisation", f"dependency events only ({tag})", free, host))
    batch = lab.multi("D", MODULES, 4, concurrent=False, replicas=2)
    streams = lab.multi("D", MODULES, 4, concurrent=True, replicas=2)
    rows.append(_row("synchronisation", "4 sequences in lock step (one barrier per layer)", batch))
    rows.append(_row("synchronisation", "4 independent streams (no barrier)", streams, batch))
    return rows


def launch_overhead(lab: Lab) -> list[dict[str, Any]]:
    """Per-kernel launch cost on the GPU: one launch per expert, batched, or resident kernels."""
    def unbatched(dev):
        dev["warm_startup_s"] = None

    def resident(dev):
        if dev["kind"] == "GPU":
            dev["persistent"] = True

    single = lab.run("A", CENTRAL, topology_edit=lambda t: edit_devices(t, unbatched))
    batched = lab.run("A", CENTRAL)
    persistent = lab.run("A", CENTRAL, topology_edit=lambda t: edit_devices(t, resident))
    return [
        _row("GPU launch overhead", "one launch per expert", single),
        _row("GPU launch overhead", "experts batched into one launch", batched, single),
        _row("GPU launch overhead", "persistent kernels", persistent, single),
    ]


def memory_channels(lab: Lab) -> list[dict[str, Any]]:
    """Pooled memory channels versus one independent engine per channel."""
    rows = []
    for key in ("D", "E"):
        pooled = lab.run(key, MODULES)
        split = lab.run(key, replace(MODULES, split_channels=True))
        rows.append(_row("memory channels", f"{key}: channels pooled per module", pooled))
        rows.append(_row("memory channels", f"{key}: independent channel engines", split, pooled))
    return rows


def communication(lab: Lab) -> list[dict[str, Any]]:
    """Streaming the hidden state to the modules before the routing decision is known."""
    rows = []
    for scale in (1.0, 8.0):
        edit = (lambda t, s=scale: scale_latency(t, s)) if scale != 1.0 else None
        lazy = lab.run("D", MODULES, topology_edit=edit)
        eager = lab.run("D", MODULES, topology_edit=edit, config_edit=with_policy(eager_activations=True))
        tag = f"latencies x{scale:g}"
        rows.append(_row("communication", f"activations sent after routing ({tag})", lazy))
        rows.append(_row("communication", f"activations streamed eagerly ({tag})", eager, lazy))
    return rows


def output_projection(lab: Lab) -> list[dict[str, Any]]:
    """Where the output projection runs: module shards, the trunk device, or chosen per token."""
    freq = lab.prior.expert_frequencies()
    co = placement.coactivation(lab.prior)

    def head(where: str):
        def edit(config):
            trunk = config.topology.by_name("gpu").device_id
            mods = [d.device_id for d in config.topology.devices.values()
                    if d.name.startswith("mod") and d.caps.can_compute]  # fmt: skip
            return replace(config, place=lambda p, r: placement.place_modules(
                p, r, trunk, mods, policy="coactivation", frequencies=freq, coactivation_matrix=co,
                head=where))  # fmt: skip
        return edit

    on_modules = lab.run("D", MODULES, config_edit=head("modules"))
    on_trunk = lab.run("D", MODULES, config_edit=head("trunk"))
    dynamic = lab.run("D", MODULES, config_edit=head("auto"))
    return [
        _row("output projection", "row shards on the modules", on_modules),
        _row("output projection", "on the trunk device", on_trunk, on_modules),
        _row("output projection", "both; location chosen per token", dynamic, on_modules),
    ]


def energy(lab: Lab) -> list[dict[str, Any]]:
    """Idle power, gating and energy-aware scheduling."""
    def no_gating(dev):
        dev["power"]["sleep_w"] = None

    rows = []
    hybrid = HardwareDesign(modules=4)
    always_on = lab.run("F", hybrid, replicas=2, topology_edit=lambda t: edit_devices(t, no_gating))
    gated = lab.run("F", hybrid, replicas=2)
    rows.append(_row("energy", "F: no power gating", always_on))
    rows.append(_row("energy", "F: idle devices gated", gated, always_on))
    for weight in (1e-3, 1e-2):
        aware = lab.run("F", hybrid, replicas=2, config_edit=with_policy(energy_weight_s_per_j=weight))
        rows.append(_row("energy", f"F: energy-aware scheduling ({weight:g} s per joule)", aware, always_on))
    central = lab.run("A", CENTRAL)
    aware = lab.run("A", CENTRAL, config_edit=with_policy(energy_weight_s_per_j=1e-2))
    rows.append(_row("energy", "A: time-only scheduling", central))
    rows.append(_row("energy", "A: energy-aware scheduling (0.01 s per joule)", aware, central))
    return rows


def scheduler_overhead(lab: Lab, eval_costs_s=(50e-9, 200e-9, 1e-6)) -> list[dict[str, Any]]:
    """Greedy versus bounded graph-level optimisation, and what the decisions would cost."""
    rows = []
    greedy = None
    for rounds, label in ((0, "greedy earliest-finish"), (1, "one refinement pass"),
                          (2, "two refinement passes"), (4, "four refinement passes")):  # fmt: skip
        out = lab.run("F", MODULES, replicas=2, config_edit=with_policy(joint_rounds=rounds))
        greedy = greedy or out
        sched = out.result.scheduler
        per_batch = sched["plans_evaluated"] / max(sched["batches"], 1)
        batch_s = out.result.elapsed_s / max(sched["batches"], 1)
        row = _row("scheduler", label, out, greedy, plans_per_batch=per_batch,
                   joint_improvements=sched["joint_improvements"])  # fmt: skip
        row["overhead"] = {f"{c * 1e9:.0f} ns per plan evaluation": per_batch * c / batch_s
                           for c in eval_costs_s}  # fmt: skip
        rows.append(row)
    return rows


def compression(seed: int = 0) -> dict[str, Any]:
    """What lossless coding of INT4 codes can save -- on synthetic weights only."""
    rng = np.random.default_rng(seed)
    out = {}
    samples = {
        "gaussian": rng.standard_normal((256, 4096)),
        "laplace": rng.laplace(size=(256, 4096)),
        "student-t (3 d.o.f.)": rng.standard_t(3, size=(256, 4096)),
    }
    for name, weights in samples.items():
        bits = symbol_entropy_bits(quantize_int4(weights)) + 0.5
        out[name] = {"entropy_bits_per_weight": bits, "saving_vs_int4": 1 - bits / 4.5}
    return out


def run_all(lab: Lab) -> dict[str, Any]:
    return {
        "critical_path": critical_path(lab),
        "imbalance": imbalance(lab),
        "scaling": scaling(lab),
        "synchronisation": synchronisation(lab),
        "launch_overhead": launch_overhead(lab),
        "memory_channels": memory_channels(lab),
        "communication": communication(lab),
        "output_projection": output_projection(lab),
        "energy": energy(lab),
        "scheduler": scheduler_overhead(lab),
        "compression": compression(),
    }
