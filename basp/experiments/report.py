"""Reproducible research reports.

Every experiment is deterministic: the same code, profiles and seeds produce
byte-identical report files (wall-clock measurements are printed by the runner
but kept out of the files).  All performance figures are *simulated* under the
illustrative hardware profiles in ``configs/``; they are not measurements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from basp.backends.cpu import NumericPayload
from basp.backends.virtual.runtime import RunResult
from basp.core.capabilities import DeviceKind
from basp.core.tensor import CompressionFormat, FormatTable
from basp.core.topology import Topology
from basp.dataflow.base import ALL_MODES, CachePolicy, DataflowMode, ExecutionPolicy, PrefetchPolicy
from basp.experiments.analytic import compare_paths, size_arithmetic
from basp.experiments.configurations import Configuration, standard_configurations
from basp.experiments.harness import build
from basp.experiments.modes import Scenario, expected_winners, mode_study
from basp.experiments.profiles import MODELS, load_json, load_topology, load_workload
from basp.kernels import quantize_int4, symbol_entropy_bits
from basp.scheduler import placement
from basp.verification import check_accounting, check_dependencies, compare_to_reference
from basp.workloads.model import MoEModelSpec, tiny_spec
from basp.workloads.tiny import TinyModel
from basp.workloads.trace import RoutingTrace, cyclic_trace, synth_trace

GiB = 1024**3


# -- helpers ---------------------------------------------------------------------------
def table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:,.3g}" if abs(value) < 1000 else f"{value:,.0f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def vary_topology(topology: Topology, *, gpu_bytes: int | None = None, modules: int | None = None,
                  latency_scale: float | None = None) -> Topology:  # fmt: skip
    """Copy of a profile with a different GPU memory size, module count or latency level.

    ``latency_scale`` multiplies every link latency, kernel start-up time and the
    host synchronisation latency.
    """
    data = topology.to_dict()
    if latency_scale is not None:
        data["sync_latency_s"] *= latency_scale
        for link in data["links"]:
            link["latency_s"] *= latency_scale
        for dev in data["devices"]:
            dev["kernel_startup_s"] *= latency_scale
    if gpu_bytes is not None:
        for dev in data["devices"]:
            if dev["kind"] == "GPU":
                dev["mem_capacity"] = gpu_bytes
    if modules is not None:
        names = [d["name"] for d in data["devices"] if d["kind"] == "NEAR_MEMORY"]
        drop = set(names[modules:])
        data["devices"] = [d for d in data["devices"] if d["name"] not in drop]
        data["links"] = [k for k in data["links"] if k["a"] not in drop and k["b"] not in drop]
    return Topology.from_dict(data)


def make_traces(spec: MoEModelSpec, workload: dict) -> tuple[RoutingTrace, RoutingTrace]:
    """(profiling trace, held-out evaluation trace) from one seeded generator run."""
    t = workload["trace"]
    steps = t["prior_steps"] + workload["tokens"]
    trace = synth_trace(steps, spec.n_layers, spec.n_experts, spec.top_k, batch=workload.get("batch", 1),
                        zipf_s=t["zipf_s"], stickiness=t["stickiness"], seed=t["seed"])  # fmt: skip
    return trace.split(t["prior_steps"])


def run_config(config: Configuration, spec: MoEModelSpec, prior: RoutingTrace, held: RoutingTrace,
               workload: dict) -> RunResult:  # fmt: skip
    setup = config.build(spec, prior, batch=workload.get("batch", 1),
                         context_len=workload.get("context_len", 0),
                         max_new_tokens=workload["tokens"])  # fmt: skip
    result = setup.run(held, tokens=workload["tokens"], warmup_tokens=workload.get("warmup_tokens", 0))
    if not result.ok:
        raise RuntimeError(f"{config.key}: {result.failures[:3]} {result.violations[:3]}")
    problems = check_accounting(setup.system)
    if problems:
        raise RuntimeError(f"{config.key}: accounting check failed: {problems[:3]}")
    setup.residency.check_invariants()
    return result


def summarise(result: RunResult) -> dict[str, Any]:
    """Deterministic, JSON-friendly summary of a run."""
    tokens = result.tokens * result.batch
    return {
        "policy": result.policy,
        "tokens": result.tokens,
        "batch": result.batch,
        "tokens_per_s": result.tokens_per_s,
        "token_latency_ms": result.token_latency_s * 1e3,
        "link_mb_per_token": result.link_bytes_per_token / 1e6,
        "link_mb_by_kind": {k: v / 1e6 / tokens for k, v in sorted(result.link_bytes.items())},
        "joules_per_token": result.energy.total_j / tokens if tokens else 0.0,
        "tokens_per_joule": result.tokens_per_joule,
        "cache_hit_rate": result.cache["hit_rate"],
        "stall_ms_per_token": result.stall_s * 1e3 / tokens if tokens else 0.0,
        "estimate_error": result.estimate_error["mean_abs_rel"],
        "mode_counts": result.mode_counts,
        "prefetch": result.prefetch,
        "plans_per_op": result.scheduler["plans_evaluated"] / max(result.scheduler["ops"], 1),
        "joint_improvements": result.scheduler["joint_improvements"],
        "devices": {
            name: {k: row[k] for k in ("busy_fraction", "mem_utilisation_busy", "kernels") if k in row}
            for name, row in result.devices.items()
        },
    }


# -- experiments -------------------------------------------------------------------------
@dataclass
class ConfigRow:
    config: Configuration
    summary: dict[str, Any]


def configurations_experiment(spec, desktop, basp, workload) -> list[ConfigRow]:
    prior, held = make_traces(spec, workload)
    return [
        ConfigRow(config, summarise(run_config(config, spec, prior, held, workload)))
        for config in standard_configurations(desktop, basp, prior)
    ]


def gpu_memory_experiment(spec, desktop, basp, workload, sizes_gib) -> list[dict[str, Any]]:
    """Where does each dataflow pay off as the GPU holds less of the model?"""
    prior, held = make_traces(spec, workload)
    rows = []
    for gib in sizes_gib:
        configs = {
            c.key: c for c in standard_configurations(
                vary_topology(desktop, gpu_bytes=gib * GiB), vary_topology(basp, gpu_bytes=gib * GiB), prior
            )
        }  # fmt: skip
        row: dict[str, Any] = {"gpu_gib": gib}
        for key in ("static_split", "adaptive", "distributed", "complete"):
            row[key] = summarise(run_config(configs[key], spec, prior, held, workload))["tokens_per_s"]
        rows.append(row)
    return rows


def latency_experiment(spec, desktop, basp, workload, scales=(0.25, 1.0, 4.0, 16.0)) -> list[dict[str, Any]]:
    """How sensitive is each dataflow to fixed per-message and per-kernel overheads?"""
    prior, held = make_traces(spec, workload)
    rows = []
    for scale in scales:
        configs = {
            c.key: c for c in standard_configurations(
                vary_topology(desktop, latency_scale=scale), vary_topology(basp, latency_scale=scale),
                prior,
            )
        }  # fmt: skip
        row: dict[str, Any] = {"latency_scale": scale}
        for key in ("adaptive", "distributed", "complete"):
            row[key] = summarise(run_config(configs[key], spec, prior, held, workload))["tokens_per_s"]
        rows.append(row)
    return rows


def scaling_experiment(spec, basp, workload, counts=(1, 2, 4)) -> list[dict[str, Any]]:
    """Throughput of the distributed-memory configuration versus module count."""
    prior, held = make_traces(spec, workload)
    rows = []
    base = None
    for n in counts:
        topo = vary_topology(basp, modules=n)
        config = next(c for c in standard_configurations(topo, topo, prior) if c.key == "distributed")
        tps = summarise(run_config(config, spec, prior, held, workload))["tokens_per_s"]
        base = base or tps
        rows.append({"modules": n, "tokens_per_s": tps, "speedup": tps / base,
                     "efficiency": tps / base / (n / counts[0])})  # fmt: skip
    return rows


def multi_user_experiment(spec, desktop, basp, workload, batch: int) -> list[dict[str, Any]]:
    """Single-stream latency and aggregate throughput are separate questions."""
    rows = []
    for b in (1, batch):
        w = dict(workload, batch=b)
        prior, held = make_traces(spec, w)
        configs = {c.key: c for c in standard_configurations(desktop, basp, prior)}
        for key in ("adaptive", "complete"):
            s = summarise(run_config(configs[key], spec, prior, held, w))
            rows.append({"configuration": key, "sequences": b, "aggregate_tokens_per_s": s["tokens_per_s"],
                         "per_sequence_latency_ms": s["token_latency_ms"]})  # fmt: skip
    return rows


def predictive_experiment(desktop: Topology, tokens: int = 48, prior_steps: int = 96) -> list[dict[str, Any]]:
    """Prefetching on held-out routing, with and without transitional structure.

    A mid-sized model on the desktop profile with a GPU that holds only part of
    the experts.  ``cyclic`` routing changes completely every token in a
    learnable order (best case for prediction); ``sticky`` routing is the
    default synthetic trace, whose structure a cache already exploits.
    """
    spec = MoEModelSpec("mid-moe", hidden=1024, n_layers=8, n_experts=64, top_k=4, expert_inter=1024,
                        n_heads=8, n_kv_heads=2, head_dim=128, vocab=8192)  # fmt: skip
    expert_bytes = spec.expert_params * 4.5 / 8
    fixed = spec.n_layers * spec.trunk_params * 4.5 / 8 + spec.head_params * 4.5 / 8
    gpu_bytes = int((fixed + 0.3 * spec.n_layers * spec.n_experts * expert_bytes) * 16 / 15)
    topo = vary_topology(desktop, gpu_bytes=gpu_bytes)
    host = topo.of_kind(DeviceKind.CPU)[0].device_id
    gpu = topo.of_kind(DeviceKind.GPU)[0].device_id
    traces = {
        "cyclic": cyclic_trace(prior_steps + tokens, spec.n_layers, spec.n_experts, spec.top_k,
                               period=8, seed=5),
        "sticky": synth_trace(prior_steps + tokens, spec.n_layers, spec.n_experts, spec.top_k,
                              zipf_s=0.8, stickiness=0.6, seed=5),
    }  # fmt: skip
    rows = []
    for name, trace in traces.items():
        prior, held = trace.split(prior_steps)
        hot = placement.hottest_experts(prior.expert_frequencies())
        for label, prefetch in (("reactive cache", None), ("with prefetch", PrefetchPolicy())):
            policy = ExecutionPolicy(
                label, ALL_MODES, frozenset({gpu}), cache=CachePolicy("lru", (gpu,), background=False),
                prefetch=prefetch,
            )  # fmt: skip
            setup = build(topo, spec, policy,
                          lambda p, r, hot=hot: placement.place_gpu_host(p, r, gpu, host, prefill=hot),
                          context_len=512, max_new_tokens=tokens, prior=prior)  # fmt: skip
            result = setup.run(held, tokens=tokens, warmup_tokens=tokens // 4)
            if not result.ok:
                raise RuntimeError(f"predictive/{name}/{label}: {result.failures[:3]}")
            s = summarise(result)
            rows.append({
                "routing": name, "policy": label, "tokens_per_s": s["tokens_per_s"],
                "stall_ms_per_token": s["stall_ms_per_token"], "cache_hit_rate": s["cache_hit_rate"],
                "prefetch_issued": result.prefetch["issued"], "prefetch_hits": result.prefetch["hits"],
                "wasted_mb": result.prefetch["wasted_bytes"] / 1e6,
            })  # fmt: skip
    return rows


def correctness_experiment(basp: Topology) -> dict[str, Any]:
    """Numerical co-simulation of a small real-valued MoE model under every policy family."""
    spec = tiny_spec()
    model = TinyModel.random(spec, seed=3)
    gpu = basp.of_kind(DeviceKind.GPU)[0].device_id
    modules = [d.device_id for d in basp.of_kind(DeviceKind.NEAR_MEMORY)]
    out: dict[str, Any] = {"runs": []}
    logits = []
    for label, fused, batch in (("fused", True, 3), ("unfused", False, 3)):
        policy = ExecutionPolicy(label, ALL_MODES, cache=CachePolicy("benefit", (gpu,)))
        setup = build(basp, spec, policy,
                      lambda p, r: placement.place_modules(p, r, gpu, modules, policy="round_robin"),
                      batch=batch, head_shards=len(modules), fused=fused)  # fmt: skip
        payload = NumericPayload(model, setup.program, model.initial_input(batch))
        result = setup.run(None, tokens=6, payload=payload, keep_log=True)
        numeric = compare_to_reference(model, payload)
        deps = check_dependencies(result.op_log, setup.system)
        out["runs"].append({
            "graph": label, "max_rel_l2": numeric.max_rel_l2, "routing_matches": numeric.routing_matches,
            "tokens_match": numeric.tokens_match, "dependency_violations": len(deps),
            "accounting_violations": len(check_accounting(setup.system)), "ok": result.ok and numeric.ok,
        })  # fmt: skip
        logits.append(np.stack(payload.logits))
    # Bit-identity is a property of the graph form only while the reduction order is fixed:
    # device-local partial sums (output-stationary) reorder FP32 additions.
    exact = []
    for fused in (True, False):
        policy = ExecutionPolicy("fixed", ALL_MODES - {DataflowMode.OUTPUT_STATIONARY},
                                 cache=CachePolicy("benefit", (gpu,)))  # fmt: skip
        setup = build(basp, spec, policy,
                      lambda p, r: placement.place_modules(p, r, gpu, modules, policy="round_robin"),
                      batch=3, head_shards=len(modules), fused=fused)  # fmt: skip
        payload = NumericPayload(model, setup.program, model.initial_input(3))
        setup.run(None, tokens=6, payload=payload)
        exact.append(np.stack(payload.logits))
    out["fused_equals_unfused"] = bool(np.array_equal(exact[0], exact[1]))
    out["max_abs_diff_between_graphs"] = float(np.abs(logits[0] - logits[1]).max())
    rng = np.random.default_rng(0)
    sample = quantize_int4(rng.standard_normal((256, 4096)))
    out["gaussian_int4_entropy_bits"] = symbol_entropy_bits(sample)
    return out


# -- rendering ---------------------------------------------------------------------------
def render_profile(topology: Topology) -> str:
    rows = []
    for dev in topology.devices.values():
        rows.append([dev.name, dev.kind.name, f"{dev.mem_capacity / GiB:.0f} GiB",
                     f"{dev.mem_bandwidth / 1e9:.0f} GB/s",
                     f"{dev.compute_rate / 1e9:,.0f} GMAC/s" if dev.compute_rate else "-",
                     f"{dev.kernel_startup_s * 1e6:.1f} us", f"{dev.sram_bytes / 2**20:.0f} MiB",
                     f"{dev.power.idle_w:.0f} W"])  # fmt: skip
    links = [[k.name, f"{k.bandwidth_ab / 1e9:.0f} GB/s", f"{k.latency_s * 1e6:.0f} us"]
             for k in topology.links]  # fmt: skip
    return (
        table(["device", "kind", "memory", "sustained bandwidth", "compute", "kernel start-up",
               "tile SRAM", "idle power"], rows)
        + "\n\n" + table(["link", "bandwidth per direction", "latency"], links)
    )  # fmt: skip


def render_configurations(rows: list[ConfigRow]) -> str:
    by_key = {r.config.key: r.summary for r in rows}
    out = []
    for r in rows:
        s = r.summary
        base = by_key.get(r.config.baseline or "")
        versus = f"{s['tokens_per_s'] / base['tokens_per_s']:.2f}x {r.config.baseline}" if base else "-"
        out.append([r.config.title, r.config.purpose, s["tokens_per_s"], s["token_latency_ms"],
                    s["link_mb_per_token"], s["tokens_per_joule"], f"{s['cache_hit_rate']:.0%}",
                    s["stall_ms_per_token"], versus])  # fmt: skip
    return table(["configuration", "purpose", "tokens/s", "ms/token", "link MB/token", "tokens/J",
                  "expert runs on GPU-resident weights", "stall ms/token", "vs. baseline"], out)  # fmt: skip


def render_modes(scenarios: list[Scenario]) -> str:
    rows = []
    for sc in scenarios:
        values = ", ".join(f"{k.replace('_', '-')}: {v:,.1f}" for k, v in sc.results.items())
        ok = "yes" if sc.winner == expected_winners()[sc.key] else "NO"
        rows.append([sc.title, f"{sc.metric} ({sc.unit})", values, sc.winner.replace("_", "-"), ok])
    return table(["scenario", "metric", "simulated result per mode", "best", "as predicted"], rows)


def render_acceptance(configs: list[ConfigRow], scaling, correctness) -> str:
    by = {r.config.key: r.summary for r in configs}
    complete, adaptive, paging = by["complete"], by["adaptive"], by["gpu_only"]
    module_util = max(
        (row.get("mem_utilisation_busy", 0.0) for name, row in by["distributed"]["devices"].items()
         if name.startswith("mod")), default=0.0,
    )  # fmt: skip
    worst = max(run["max_rel_l2"] for run in correctness["runs"])
    all_ok = all(run["ok"] for run in correctness["runs"]) and correctness["fused_equals_unfused"]
    last = scaling[-1]

    def verdict(value: float, target: float) -> str:
        return "simulated: meets target" if value >= target else "simulated: below target"

    def ratio(a: float, b: float) -> float:
        return a / b if b else float("inf") if a else 1.0

    traffic = 1 - ratio(complete["link_mb_per_token"], paging["link_mb_per_token"])
    traffic_adaptive = 1 - ratio(complete["link_mb_per_token"], adaptive["link_mb_per_token"])
    speed = ratio(complete["tokens_per_s"], adaptive["tokens_per_s"])
    energy = ratio(complete["tokens_per_joule"], adaptive["tokens_per_joule"])
    rows = [
        ["Correctness", "match independent numerical reference",
         f"max relative error {worst:.1e}; fused = unfused bit-exact: {correctness['fused_equals_unfused']}",
         "met on the small numeric model" if all_ok else "NOT met"],
        ["Interconnect traffic", "50% reduction on offloading workloads",
         f"{traffic:.0%} vs GPU paging, {traffic_adaptive:.0%} vs adaptive CPU/GPU",
         verdict(min(traffic, traffic_adaptive), 0.5)],
        ["Scheduler overhead", "below 5% of execution time",
         (f"{complete['plans_per_op']:.1f} plan evaluations per operation; the Python prototype's "
          "decision time is not representative"), "not measurable yet"],
        ["Memory utilisation", "above 60% of sustainable bandwidth",
         f"{module_util:.0%} of module bandwidth while a kernel runs", verdict(module_util, 0.6)],
        ["Inference throughput", "2x optimized comparable baseline",
         f"{speed:.2f}x adaptive CPU/GPU", verdict(speed, 2.0)],
        ["Energy efficiency", "2x tokens per joule",
         f"{energy:.2f}x adaptive CPU/GPU", verdict(energy, 2.0)],
        ["Scaling efficiency", "70% from one to four modules",
         f"{last['efficiency']:.0%} at {last['modules']} modules", verdict(last["efficiency"], 0.7)],
        ["Model quality", "no material degradation",
         "routing never altered; only lossless-equivalent formats exchanged", "by construction"],
    ]  # fmt: skip
    return table(["criterion", "target", "result in simulation", "status"], rows)


def build_report(workload_name: str = "workload_qwen3_decode", *, quick: bool = False,
                 progress=None) -> tuple[str, dict]:  # fmt: skip
    """Run every experiment and return (markdown report, results dictionary).

    ``progress`` is an optional callable receiving the name of each finished stage.
    """
    done = progress or (lambda _stage: None)
    workload = load_workload(workload_name)
    spec = MODELS[workload["model"]]
    if quick:
        workload = dict(workload, tokens=4, warmup_tokens=1,
                        trace=dict(workload["trace"], prior_steps=8))  # fmt: skip
    desktop, basp = load_topology("desktop_gpu12"), load_topology("basp_4module")
    section6, sizing = compare_paths(), size_arithmetic()
    configs = configurations_experiment(spec, desktop, basp, workload)
    done("configurations")
    modes = mode_study(basp)
    sizes = (8, 12) if quick else (6, 8, 12, 16, 24)
    memory = gpu_memory_experiment(spec, desktop, basp, workload, sizes)
    done("gpu memory sweep")
    latency = latency_experiment(spec, desktop, basp, workload, (1.0, 4.0) if quick else
                                 (0.25, 1.0, 4.0, 16.0))  # fmt: skip
    done("latency sweep")
    scaling = scaling_experiment(spec, basp, workload)
    done("module scaling")
    users = multi_user_experiment(spec, desktop, basp,
                                  dict(workload, tokens=3, warmup_tokens=1) if quick else
                                  dict(workload, tokens=8, warmup_tokens=3), batch=4 if quick else 8)  # fmt: skip
    predictive = predictive_experiment(desktop, tokens=12 if quick else 48,
                                       prior_steps=24 if quick else 96)  # fmt: skip
    done("multi-user and predictive")
    correctness = correctness_experiment(basp)
    done("correctness")
    formats = FormatTable()

    results = {
        "workload": workload,
        "model": {
            "name": spec.name,
            "total_params": spec.total_params,
            "active_params": spec.active_params,
        },
        "section6": dict(section6.as_rows()),
        "configurations": {r.config.key: r.summary for r in configs},
        "modes": {sc.key: {"results": sc.results, "winner": sc.winner} for sc in modes},
        "gpu_memory": memory,
        "latency": latency,
        "scaling": scaling,
        "multi_user": users,
        "predictive": predictive,
        "correctness": correctness,
    }
    t = workload["trace"]
    md = f"""# BASP Dataflow v0.1 -- simulation report

> **Read this first.** Everything below is the output of a discrete-event *simulation* driven by
> the illustrative hardware profiles in `configs/` and a *synthetic* routing trace.  No number here
> is a measurement of real hardware or of a real model's router.  The report shows what the dataflow
> engine concludes *given those assumptions*, and how sensitive the conclusions are to them.
> Regenerate with `python -m benchmarks.run_report` (deterministic).

## 1. Analytical dataflow comparison (specification section 6)

One expert with hidden size 4096 and intermediate size 14336 at 4.5 bits per weight; 24 GB/s
interconnect, 54 GB/s local memory, 128 GMAC/s local compute.

{table(["quantity", "value"], [list(r) for r in section6.as_rows()])}

Arithmetic sizing (section 10): two DDR5-5600 channels deliver {sizing.memory_bandwidth / 1e9:.1f} GB/s
in theory; consuming that stream at 4.5 bits per weight needs {sizing.required_mac_rate / 1e9:.0f} GMAC/s,
and 1024 lanes at 250 MHz provide {sizing.lane_mac_rate / 1e9:.0f} GMAC/s ({sizing.headroom:.2f}x headroom).
The simulator reproduces both paths of the table on an idle system to within floating-point error
(`tests/test_scheduler.py::test_section_6_numbers_are_reproduced_by_the_simulator`).

## 2. Assumptions

**Workload.** {spec.name}: {spec.total_params / 1e9:.1f} B parameters, {spec.active_params / 1e9:.1f} B
activated per token, {spec.n_experts} experts per layer of which {spec.top_k} are selected, {spec.n_layers}
layers; INT4 weights with one FP16 scale per 32 weights (4.5 bits per weight, {spec.total_params * 4.5 / 8 / GiB:.1f} GiB).
Single-sequence decoding at context length {workload.get("context_len", 0)}, {workload["tokens"]} tokens of which
the first {workload.get("warmup_tokens", 0)} are warm-up.  Routing is synthetic: Zipf({t["zipf_s"]}) expert
popularity per layer and per-slot stickiness {t["stickiness"]}; placement and the predictor see
{t["prior_steps"]} profiling steps, evaluation uses held-out steps of the same generator.

**Hardware profile `desktop_gpu12`** (configurations 1-4, 6, 7):

{render_profile(desktop)}

**Hardware profile `basp_4module`** (configurations 5 and 8) adds four compute-memory modules:

{render_profile(basp)}

Format assumptions: lossless entropy coding of INT4 codes is taken as {formats.bits(CompressionFormat.ENTROPY):.2f}
bits per weight (the order-0 entropy of INT4-quantized Gaussian weights measured by the reference
kernels is {correctness["gaussian_int4_entropy_bits"]:.2f} bits per code, plus 0.5 bit of scales); real
weights will differ.

## 3. Eight execution configurations

{render_configurations(configs)}

"Link MB/token" counts bytes carried by links, once per hop.  "Stall" is time operations spent
waiting for synchronous weight movement.  "Expert runs on GPU-resident weights" is the share of
expert executions that found their weights already on the GPU (the rest ran elsewhere or waited
for a transfer).  The module configurations have additional memory and
compute installed, so rows 5 and 8 compare dataflows on their natural hardware, not at equal cost.

Dataflow modes chosen by the complete configuration:
{", ".join(f"{k.replace('_', '-')} {v}" for k, v in sorted(by_mode(configs).items()))}.

## 4. No dataflow is universally best

Each scenario runs the same operations under single-mode policies on `basp_4module`.

{render_modes(modes)}

## 5. Sensitivity: GPU memory and fixed overheads

Tokens per second as GPU memory varies (model size {spec.total_params * 4.5 / 8 / GiB:.1f} GiB):

{table(["GPU memory (GiB)", "static CPU/GPU", "adaptive CPU/GPU", "distributed memory", "complete BASP"],
       [[r["gpu_gib"], r["static_split"], r["adaptive"], r["distributed"], r["complete"]] for r in memory])}

Tokens per second as every link latency, kernel start-up time and the host synchronisation
latency are scaled together (1 = the profile values):

{table(["latency scale", "adaptive CPU/GPU", "distributed memory", "complete BASP"],
       [[f"{r['latency_scale']:g}x", r["adaptive"], r["distributed"], r["complete"]] for r in latency])}

## 6. Scaling with module count

Distributed-memory configuration, trunk on the GPU:

{table(["modules", "tokens/s", "speed-up", "scaling efficiency"],
       [[r["modules"], r["tokens_per_s"], f"{r['speedup']:.2f}x", f"{r['efficiency']:.0%}"] for r in scaling])}

## 7. Single-user latency versus multi-user throughput

{table(["configuration", "concurrent sequences", "aggregate tokens/s", "per-sequence ms/token"],
       [[r["configuration"], r["sequences"], r["aggregate_tokens_per_s"], r["per_sequence_latency_ms"]]
        for r in users])}

## 8. Predictive prefetching on held-out routing

A mid-sized model whose GPU holds about 30% of the experts; all expert computation on the GPU, so
every miss is a synchronous weight transfer.

{table(["routing", "policy", "tokens/s", "stall ms/token", "GPU-resident runs", "prefetches", "hits",
        "wasted MB"],
       [[r["routing"], r["policy"], r["tokens_per_s"], r["stall_ms_per_token"],
         f"{r['cache_hit_rate']:.0%}", r["prefetch_issued"], r["prefetch_hits"], r["wasted_mb"]]
        for r in predictive])}

`cyclic` routing is a constructed best case in which the selection changes completely every token
in a learnable order; `sticky` is the default synthetic trace, whose structure a reactive cache
already captures.  Whether real routers have exploitable structure beyond recency must be measured
on recorded traces (`RoutingTrace.load`).

## 9. Correctness

{table(["graph", "max relative error vs float64 reference", "routing identical", "tokens identical",
        "dependency violations", "accounting violations"],
       [[r["graph"], f"{r['max_rel_l2']:.2e}", r["routing_matches"], r["tokens_match"],
         r["dependency_violations"], r["accounting_violations"]] for r in correctness["runs"]])}

A small attention-free MoE language model with real INT4 weights is decoded autoregressively by the
streaming kernels *following the scheduled plans* (sharded output projection, fused partial sums,
replication) while the simulator provides the execution order.  Fused and unfused graphs produce
bit-identical logits: {correctness["fused_equals_unfused"]}.

## 10. Acceptance targets (specification section 17)

Targets are goals, not achievements.  "Simulated" means the figure comes from this model under the
assumptions of section 2 and says nothing yet about real hardware.

{render_acceptance(configs, scaling, correctness)}
"""
    return md, results


def by_mode(configs: list[ConfigRow]) -> dict[str, int]:
    return next(r.summary["mode_counts"] for r in configs if r.config.key == "complete")


def write_report(outdir: str | Path, workload_name: str = "workload_qwen3_decode", *,
                 quick: bool = False, progress=None) -> tuple[Path, Path]:  # fmt: skip
    outdir = Path(outdir)
    markdown, results = build_report(workload_name, quick=quick, progress=progress)
    outdir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = outdir / "report.md", outdir / "results.json"
    md_path.write_text(markdown)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=float) + "\n")
    return md_path, json_path


__all__ = ["build_report", "load_json", "vary_topology", "write_report"]
