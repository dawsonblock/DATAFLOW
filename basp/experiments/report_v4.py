"""The v0.4 report: architecture discovery, investigations, the physical runtime's verification,
expert-cache policies, and the evidence-gated decision (decided from receipts).

Deterministic: the same code, profiles, estimates, inputs and seeds give byte-identical
``report.md`` and ``results.json`` (the software environment and git state go to
``provenance.json``).  Every performance figure is simulated unless section 2 shows passing
held-out measurements; claims carry evidence levels E0-E3 (``basp.evidence``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from basp.backends.cpu import NumericPayload
from basp.calibration import coverage
from basp.calibration.plan import plan_runs
from basp.core.capabilities import DeviceKind
from basp.core.tensor import CompressionFormat
from basp.dataflow.base import ALL_MODES, CachePolicy, DataflowMode, ExecutionPolicy
from basp.dse.hardware import CORNERS, HardwareDesign
from basp.evidence import (
    CONFIG_DIR,
    EvidenceLevel,
    Receipt,
    ReceiptRegistry,
    config_path,
    profile_identity,
    provenance,
    sha256_file,
    sha256_json,
    source_fingerprint,
    stale_reasons,
)
from basp.experiments import investigations as inv
from basp.experiments.architectures import ARCHITECTURES
from basp.experiments.calibrate import (
    MACHINE_PROFILES,
    calibration_status,
    predict,
    profile_qualification,
)
from basp.experiments.gate import (
    Candidate,
    GateDecision,
    evaluate_gate,
    scheduler_receipt,
    simulation_receipts,
)
from basp.experiments.harness import build
from basp.experiments.profiles import load_json, load_topology
from basp.experiments.report import table, vary_topology
from basp.experiments.suite import (
    REPLICAS,
    architecture_table,
    default_design,
    model_suite,
    suite_spec,
    workload_suite,
)
from basp.experiments.variants import (
    Context,
    base_profile,
    context,
    design_for,
    load_compression_profile,
    routing_source,
    run_architecture,
    set_context,
    traces,
)
from basp.memory import ExpertCost, compare, trace_steps
from basp.runtime import DeviceManager
from basp.scheduler import placement
from basp.verification import check_accounting, check_dependencies, compare_to_reference
from basp.workloads.model import tiny_spec
from basp.workloads.tiny import TinyModel
from basp.workloads.trace import RoutingTrace, synth_trace
from basp.workloads.zoo import MODELS, SUITE

GiB = 1024**3
M = DataflowMode


# -- experiments that live only in the report ----------------------------------------------
def correctness() -> dict[str, Any]:
    """Numeric co-simulation of a small real-valued model under three dataflow families.

    Dataflows that accumulate partial sums on devices, or partition experts, add the same FP32
    terms in a different order, so their logits agree within rounding rather than bit for bit.
    With the reduction order fixed, the graph form must not change a single bit.
    """
    spec = tiny_spec()
    model = TinyModel.random(spec, seed=3)
    topology = vary_topology(load_topology("basp_4module"), modules=2)
    gpu = topology.of_kind(DeviceKind.GPU)[0].device_id
    modules = [d.device_id for d in topology.of_kind(DeviceKind.NEAR_MEMORY)]
    cases = [
        ("adaptive hybrid, fused graph", {"fused": True},
         ExecutionPolicy("f", ALL_MODES, cache=CachePolicy("benefit", (gpu,))),
         lambda p, r: placement.place_modules(p, r, gpu, modules, policy="round_robin")),
        ("adaptive hybrid, unfused graph", {"fused": False},
         ExecutionPolicy("u", ALL_MODES, cache=CachePolicy("benefit", (gpu,))),
         lambda p, r: placement.place_modules(p, r, gpu, modules, policy="round_robin")),
        ("tensor-parallel with partial sums, no host in the loop", {"expert_shards": 2},
         ExecutionPolicy("e", frozenset({M.DISTRIBUTED, M.OUTPUT_STATIONARY, M.WEIGHT_STATIONARY}),
                         host_in_loop=False, eager_activations=True),
         lambda p, r: placement.place_tensor_parallel(p, r, gpu, modules)),
    ]  # fmt: skip
    runs, logits = [], []
    for label, options, policy, place in cases:
        setup = build(topology, spec, policy, place, batch=3, head_shards=2, **options)
        payload = NumericPayload(model, setup.program, model.initial_input(3))
        result = setup.run(None, tokens=6, payload=payload, keep_log=True)
        numeric = compare_to_reference(model, payload)
        runs.append({
            "dataflow": label, "max_rel_l2": numeric.max_rel_l2,
            "routing_matches": numeric.routing_matches, "tokens_match": numeric.tokens_match,
            "dependency_violations": len(check_dependencies(result.op_log, setup.system)),
            "accounting_violations": len(check_accounting(setup.system)),
            "ok": bool(result.ok and numeric.ok),
        })  # fmt: skip
        logits.append(np.stack(payload.logits))
    scale = float(np.abs(logits[0]).max())
    fixed_order = ExecutionPolicy("fixed", ALL_MODES - {M.OUTPUT_STATIONARY},
                                  cache=CachePolicy("benefit", (gpu,)))  # fmt: skip
    exact = []
    for fused in (True, False):
        setup = build(topology, spec, fixed_order, cases[0][3], batch=3, head_shards=2, fused=fused)
        payload = NumericPayload(model, setup.program, model.initial_input(3))
        setup.run(None, tokens=6, payload=payload)
        exact.append(np.stack(payload.logits))
    return {
        "runs": runs,
        "fused_equals_unfused": bool(np.array_equal(exact[0], exact[1])),
        "max_abs_diff_between_dataflows": float(max(np.abs(x - logits[0]).max() for x in logits[1:])),
        "logit_scale": scale,
        "max_rel_l2": max(r["max_rel_l2"] for r in runs),
        "all_ok": all(r["ok"] for r in runs),
    }


def candidate_scaling(arch: str, spec, prior, held, design_of, *, corner: str, tokens: int,
                      warmup_tokens: int, context_len: int = 2048) -> dict[str, Any]:  # fmt: skip
    """One-to-four-module scaling of ``arch`` itself, on the given model, at one corner.

    ``design_of(modules, replicas)`` returns the system for a module count.  The gate accepts
    scaling evidence only for the candidate it judges, in the regime it judges.
    """
    speeds = {}
    for n in (1, 4):
        replicas = min(REPLICAS[arch], n)
        out = run_architecture(arch, design_of(n, replicas), spec, prior, held, tokens=tokens,
                               warmup_tokens=warmup_tokens, context_len=context_len, replicas=replicas,
                               corner=corner)  # fmt: skip
        speeds[n] = out.result.tokens_per_s
    return {"arch": arch, "tokens_per_s": speeds, "efficiency": speeds[4] / speeds[1] / 4}


def regime_ratios(key: str, *, candidates=("D", "E", "F", "H"), corner: str = "pessimistic",
                  gpu_gib: int = 12) -> dict[str, Any]:  # fmt: skip
    """Best module architecture against the conventional baseline for one model, at one corner,
    with that architecture's own scaling efficiency in the same regime."""
    spec, bits = suite_spec(key)
    prior, held = traces(spec, tokens=4, prior_steps=8)

    def design_of(arch: str):
        return lambda n, replicas: default_design(arch, design_for(
            spec, modules=n, bits_per_weight=bits, gpu_gib=gpu_gib, replicas=replicas))  # fmt: skip

    def run(arch: str):
        return run_architecture(arch, design_of(arch)(4, REPLICAS[arch]), spec, prior, held, tokens=4,
                                warmup_tokens=1, replicas=REPLICAS[arch], corner=corner)  # fmt: skip

    base = run("A")
    best, best_out = None, None
    for arch in candidates:
        out = run(arch)
        if best_out is None or out.result.tokens_per_s > best_out.result.tokens_per_s:
            best, best_out = arch, out
    return {
        "model": spec.name, "candidate": best, "corner": corner,
        "throughput": best_out.result.tokens_per_s / base.result.tokens_per_s,
        "energy": best_out.result.tokens_per_joule / base.result.tokens_per_joule,
        "cost_efficiency": best_out.per_kusd / base.per_kusd,
        "baseline_tokens_per_s": base.result.tokens_per_s,
        "candidate_tokens_per_s": best_out.result.tokens_per_s,
        "scaling": candidate_scaling(best, spec, prior, held, design_of(best), corner=corner, tokens=4,
                                     warmup_tokens=1),
    }


def expert_cache_policies(spec, profile_name: str, *, steps: int = 128,
                          capacities_gib: tuple[int, ...] = (2, 4, 8)) -> dict[str, Any]:  # fmt: skip
    """Expert-cache policies on a GPU with too little memory for every expert (E1: costs from the
    profile, routing synthetic).  Misses execute on the CPU or migrate over the GPU link."""
    topo = load_topology(profile_name)
    host = topo.devices[topo.host]
    gpu = topo.of_kind(DeviceKind.GPU)[0]
    link = next(k for k in topo.links if {k.a, k.b} == {host.device_id, gpu.device_id})
    expert_bytes = int(spec.expert_params * 4.5 / 8)
    cost = ExpertCost.from_rates(expert_bytes, gpu_bandwidth=gpu.mem_bandwidth, cpu_bandwidth=host.mem_bandwidth,
                                 link_bandwidth=link.bandwidth_ab, link_latency_s=link.latency_s)  # fmt: skip
    trace = synth_trace(steps, spec.n_layers, spec.n_experts, spec.top_k, zipf_s=0.8, stickiness=0.6, seed=7)
    sequence = trace_steps(trace)
    rows = []
    for gib in capacities_gib:
        for r in compare(sequence, gib * GiB, lambda _key: cost):
            rows.append({"capacity_gib": gib, **r.to_json(), "ms_per_token": 1e3 * r.time_s / steps,
                         "mb_migrated_per_token": r.bytes_migrated / steps / 1e6,
                         "cpu_share": r.cpu_executions / r.activations if r.activations else 0.0})  # fmt: skip
    return {"profile": profile_name, "expert_mib": expert_bytes / 2**20, "steps": steps,
            "all_experts_gib": spec.n_layers * spec.n_experts * expert_bytes / GiB,
            "cost": {"t_gpu_us": cost.t_gpu * 1e6, "t_cpu_us": cost.t_cpu * 1e6,
                     "t_migration_us": cost.t_migration * 1e6}, "rows": rows}  # fmt: skip


def runtime_verification(evidence: str | Path | None) -> list[dict[str, Any]]:
    """Recorded runtime-parity results (``benchmarks.verify_runtime``), each checked against
    the current runtime code: a result produced by different runtime code is stale."""
    from basp.verification.parity_report import runtime_fingerprint

    out = []
    if evidence is None or not Path(evidence).is_dir():
        return out
    now = runtime_fingerprint()
    for file in sorted(Path(evidence).glob("verification/*/parity.json")):
        r = json.loads(file.read_text())
        out.append({"model": r["model"], "file": r["file"]["name"], "verdict": r["verdict"],
                    "reasons": r["reasons"], "numerical_error": r["numerical_error"],
                    "hidden_rel_l2_max": r["hidden"]["rel_l2_max"], "layers": r["hidden"]["layers_compared"],
                    "routing": r["routing"], "generation_identical": r["generation"]["identical"],
                    "generated_tokens": r["generation"]["tokens"], "perplexity": r.get("perplexity"),
                    "quantisation": r.get("quantisation"), "prompt": r.get("prompt_text"),
                    "generated_text": (r.get("generated_text") or {}).get("runtime"),
                    "decode_tokens_per_s": r["telemetry"]["decode"]["tokens_per_s"],
                    "stale": r["runtime_fingerprint"] != now})  # fmt: skip
    return out


def calibration_plan(profile_key: str = "i9-11900kf-3080ti") -> list[dict[str, Any]]:
    """Uncalibrated predictions for the benchmarks the user is asked to run."""
    from basp.calibration import Measurement

    plan = [("qwen3-4b", 99), ("qwen3-8b", 99), ("qwen3-8b", 0), ("qwen3-14b", 99), ("qwen3-14b", 20),
            ("qwen3-30b-a3b", 0), ("qwen3-30b-a3b", 16), ("qwen3-30b-a3b", 32)]  # fmt: skip
    rows = []
    for model, ngl in plan:
        m = Measurement(id=f"{model}-ngl{ngl}", machine=profile_key, split="validation", kind="decode",
                        model=model, tokens_per_s=1.0, bits_per_weight=4.85, n_gpu_layers=ngl)  # fmt: skip
        rows.append({"model": MODELS[model].name, "gpu_layers": min(ngl, MODELS[model].n_layers),
                     "predicted_tokens_per_s": predict(m)})  # fmt: skip
    return rows


# -- design-space exploration results ------------------------------------------------------
def summarise_dse(data: dict[str, Any]) -> dict[str, Any]:
    feasible = [e for e in data["evaluations"] if e["feasible"]]
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for e in feasible:
        by_arch[e["config"]["arch"]].append(e)

    def best(items, metric):
        # ties (e.g. a model that fits the GPU, so modules add nothing) go to the cheaper design
        return max(items, key=lambda e: (round(e["metrics"][metric], 9), -e["metrics"]["cost_usd"],
                                         json.dumps(e["config"], sort_keys=True)))  # fmt: skip

    per_arch = {}
    for arch in sorted(by_arch):
        items = by_arch[arch]
        per_arch[arch] = {
            "evaluated": len(items),
            "fastest": best(items, "tokens_per_s"),
            "most_efficient": best(items, "tokens_per_joule"),
            "best_per_dollar": best(items, "tokens_per_s_per_kusd"),
        }
    budgets = {}
    for budget in (2000, 5000, 10000, 20000):
        within = [e for e in feasible if e["metrics"]["cost_usd"] <= budget]
        if within:
            top = best(within, "tokens_per_s")
            conventional = [e for e in within if e["config"]["arch"] == "A"]
            budgets[str(budget)] = {"best": top,
                                    "best_conventional": best(conventional, "tokens_per_s") if conventional else None}  # fmt: skip
    powers = {}
    for watts in (130, 160, 200):
        within = [e for e in feasible if e["metrics"]["average_power_w"] <= watts]
        if within:
            top = best(within, "tokens_per_s")
            conventional = [e for e in within if e["config"]["arch"] == "A"]
            powers[str(watts)] = {"best": top,
                                  "best_conventional": best(conventional, "tokens_per_s") if conventional else None}  # fmt: skip
    # Designs that differ only in a parameter without effect (for example a second replica that
    # is never used) have identical objectives: keep the simplest of each such group.
    distinct: dict[tuple, dict] = {}
    for e in sorted(data["front"], key=lambda e: (e["metrics"]["cost_usd"],
                                                  json.dumps(e["config"], sort_keys=True))):  # fmt: skip
        key = tuple(round(e["metrics"][m], 9) for m in ("tokens_per_s", "tokens_per_joule", "cost_usd"))
        distinct.setdefault(key, e)
    front = list(distinct.values())
    return {
        "evaluated": data["evaluated"], "feasible_evaluated": len(feasible),
        "feasible_designs": data["feasible_designs"], "space_size": data["space_size"],
        "method": data["method"], "front_size": len(front), "front": front, "per_arch": per_arch,
        "equal_cost": budgets, "equal_power": powers,
        "base_profile": data.get("base_profile", "desktop_gpu12"),
        "workload": data.get("workload", {}),
        "provenance": data.get("provenance"),
        "front_archs": {a: sum(1 for e in front if e["config"]["arch"] == a) for a in sorted(by_arch)},
    }  # fmt: skip


def _design_text(config: dict[str, Any]) -> str:
    arch = config["arch"]
    if arch == "A":
        return f"A, {config['gpu_gib']} GiB GPU"
    gpu = f"{config['gpu_gib']} GiB GPU + " if config["gpu_gib"] else ""
    rep = ", 2 replicas" if config.get("replicas", 1) > 1 else ""
    return (f"{arch}, {gpu}{config['modules']} x {config['module_gib']} GiB modules, "
            f"{config['channels']} ch, {config['lanes']} lanes, {config['link_gbps']:g} GB/s{rep}")  # fmt: skip


# -- rendering -----------------------------------------------------------------------------
def _range(row: dict[str, Any], metric: str, fmt: str = "{:.0f}") -> str:
    lo, mid, hi = (row[c][metric] for c in CORNERS)
    if abs(hi - lo) <= 1e-9 * max(abs(mid), 1.0):
        return fmt.format(mid)
    low, high = min(lo, hi), max(lo, hi)
    return f"{fmt.format(mid)} ({fmt.format(low)}-{fmt.format(high)})"


def render_architectures(rows: list[dict[str, Any]]) -> str:
    out = []
    for r in rows:
        nominal, worst = r["nominal"]["vs_a"], r["pessimistic"]["vs_a"]
        out.append([
            f"{r['key']} {r['name']}", _range(r, "tokens_per_s"), _range(r, "tokens_per_joule", "{:.2f}"),
            _range(r, "cost_usd", "{:,.0f}"), _range(r, "tokens_per_s_per_kusd", "{:.1f}"),
            f"{nominal['throughput']:.2f}x / {worst['throughput']:.2f}x",
            f"{nominal['energy']:.2f}x / {worst['energy']:.2f}x",
            f"{nominal['cost_efficiency']:.2f}x / {worst['cost_efficiency']:.2f}x",
        ])  # fmt: skip
    return table(["architecture", "tokens/s", "tokens/J", "est. price (USD)", "tokens/s per k$",
                  "throughput vs A", "energy vs A", "per-dollar vs A"], out)  # fmt: skip


def _ab(rows: list[dict[str, Any]]) -> str:
    out = []
    for r in rows:
        change = f"{r['speedup'] - 1:+.1%}" if "speedup" in r else "baseline"
        out.append([r["variant"], r["tokens_per_s"], r["tokens_per_joule"], change])
    return table(["variant", "tokens/s", "tokens/J", "throughput change"], out)


def _effect(rows: list[dict[str, Any]], variant: str) -> float:
    return next(r["speedup"] - 1 for r in rows if r["variant"].startswith(variant))


def _by_variant(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["variant"]: r for r in rows}


def render_gate(decision: GateDecision) -> str:
    rows = [[c.name, c.threshold, c.result, c.status.replace("_", " "), c.evidence, c.required.name]
            for c in decision.criteria]  # fmt: skip
    return table(["criterion", "threshold", "result", "status", "evidence", "required"], rows)


def validated_profiles(calibration, evidence: str | Path | None = "evidence") -> set[str]:
    """Machine profiles that passed validation *and* are still the profile that was validated
    (``profile_qualification``): a profile edited after validation does not count."""
    return {name for name, q in profile_qualification(calibration, evidence).items() if q["qualified"]}


def stored_receipts(evidence: str | Path | None) -> ReceiptRegistry:
    """Receipts recorded by measurement tools (``evidence/**/receipts.json``)."""
    registry = ReceiptRegistry()
    if evidence is not None and Path(evidence).is_dir():
        for file in sorted(Path(evidence).rglob("receipts.json")):
            for receipt in ReceiptRegistry.load(file):
                registry.add(receipt)
    return registry


def current_identities(profile_name: str, calibration, evidence: str | Path | None) -> dict[str, str | None]:
    """The identities simulated receipts bind to now.  ``validation`` is set only if the
    comparison's profile is the one a passing validation was recorded for."""
    configs = {name: sha256_file(config_path(name)) for name in (profile_name, "design_estimates")}
    qualified = profile_qualification(calibration, evidence).get(profile_name, {})
    validation = None
    if qualified.get("qualified"):
        validation = sha256_json(calibration.by_machine[qualified["machine"]].to_json())
    return {"source": source_fingerprint(), "config": sha256_json(configs),
            "profile": profile_identity(load_json(profile_name)), "validation": validation}  # fmt: skip


def numerical_receipts(correct: dict[str, Any], regimes: list[str], candidates: list[str]) -> list[Receipt]:
    """Direct measurement of the dataflow runtime's numerical error -- on the small random-weight
    model, so it is bound to that model and cannot qualify a regime about another."""
    bindings = {"machine": "numpy (deterministic)", "model": "tiny-random", "runtime": source_fingerprint(),
                "benchmark": sha256_json(asdict(tiny_spec()))}  # fmt: skip
    return [Receipt("numerical_error", float(correct["max_rel_l2"]), EvidenceLevel.E3, arch, regime, None, bindings,
                    repetitions=len(correct["runs"]), note="numeric co-simulation against the reference model")
            for arch in candidates for regime in regimes]  # fmt: skip


def dse_provenance_now(dse: dict[str, Any]) -> dict[str, Any]:
    """The provenance the stored exploration would have if it were produced now."""
    workload = dse.get("workload", {})
    trace = workload.get("trace", {}).get("path")
    recorded = dse.get("provenance") or {}
    return provenance(configs=[dse["base_profile"], "design_estimates"], inputs={"trace": trace},
                      parameters=recorded.get("parameters_value", {}))  # fmt: skip


def build_report(*, quick: bool = False, dse_path: str | Path | None = "reports/v0.4/dse.json",
                 measurements_path: str | Path | None = "evidence", base: str | None = None,
                 trace_path: str | Path | None = None, compression_path: str | Path | None = None,
                 progress=None) -> tuple[str, dict[str, Any]]:  # fmt: skip
    """Run every experiment and return (markdown, results).

    ``base`` names the host/GPU profile the comparison is built on (default: the
    illustrative desktop).  Evidence counts as calibrated only if the validation
    measurements pass *and* the comparison runs on a profile they validated.
    ``trace_path`` (a recorded ``RoutingTrace``) replaces synthetic routing for every model it
    matches; ``compression_path`` (from ``benchmarks.profile_compression``) replaces the assumed
    entropy-coding ratio.
    """
    ctx = Context(
        base_profile=base or "desktop_gpu12",
        trace=RoutingTrace.load(trace_path) if trace_path else None,
        trace_path=str(trace_path) if trace_path else None,
        formats=load_compression_profile(compression_path) if compression_path else {},
        compression_path=str(compression_path) if compression_path else None,
    )  # fmt: skip
    previous = set_context(ctx)
    try:
        md, results = _build_report(quick, dse_path, measurements_path, progress)
    finally:
        set_context(previous)
    inputs = {"measurements": measurements_path, "trace": trace_path, "compression": compression_path,
              "dse": dse_path}  # fmt: skip
    results["provenance"] = provenance(
        configs=[ctx.base_profile, "design_estimates", "hardware_gate", *sorted(set(MACHINE_PROFILES.values()))],
        inputs={k: v for k, v in inputs.items()}, parameters={"quick": quick},
    )  # fmt: skip
    return md, results


def _build_report(quick: bool, dse_path, measurements_path, progress) -> tuple[str, dict[str, Any]]:
    done = progress or (lambda _stage: None)
    profile_name = base_profile()
    ctx = context()
    spec = MODELS["qwen3-30b-a3b"]
    thresholds = load_json("hardware_gate")
    lab = inv.Lab(spec, tokens=4, warmup_tokens=1, prior_steps=8) if quick else inv.Lab(spec)

    archs = architecture_table(lab)
    done("architectures")
    cp = inv.critical_path(lab)
    imbalance = inv.imbalance(lab)
    scaling = inv.scaling(lab, counts=(1, 4) if quick else (1, 2, 4, 8))
    sync = inv.synchronisation(lab)
    launch = inv.launch_overhead(lab)
    channels = inv.memory_channels(lab)
    comm = inv.communication(lab)
    head = inv.output_projection(lab)
    energy = inv.energy(lab)
    sched = inv.scheduler_overhead(lab)
    compression = inv.compression()
    done("investigations")
    suite_keys = ("qwen3-8b", "qwen3-30b-a3b", "llama-3.3-70b") if quick else None
    models = model_suite(keys=suite_keys)
    done("model-size suite")
    workloads = workload_suite(spec, quick=quick)
    done("workload types")
    correct = correctness()
    caching = expert_cache_policies(spec, "desktop_gpu12", steps=32 if quick else 128)
    verified = runtime_verification(measurements_path)
    backends = DeviceManager.discover().report()
    routing = routing_source(spec)
    measured_formats = dict(ctx.formats)
    calibration = calibration_status(
        measurements_path, median_max=thresholds["calibration_median_max"],
        p95_max=thresholds["calibration_p95_max"],
        min_validation=thresholds["min_validation_measurements"],
    )  # fmt: skip
    plan = calibration_plan()
    plan_data = json.loads((CONFIG_DIR / "calibration_plan.json").read_text())
    covered = {machine: coverage(plan_runs(plan_data, machine)) for machine in sorted(plan_data["machines"])}
    qualification = profile_qualification(calibration, measurements_path)
    on_validated = profile_name in validated_profiles(calibration, measurements_path)
    calibrated = on_validated
    current = current_identities(profile_name, calibration, measurements_path)
    sim_level = EvidenceLevel.E2 if calibrated else EvidenceLevel.E1
    sim_bindings = {k: v for k, v in current.items() if v is not None}
    done("correctness and calibration")

    # -- gate: the project's first workload, and the most favourable regime found ---------
    corner = thresholds["decision_corner"]
    challengers = [r for r in archs if r["key"] != "A"]
    primary = max(challengers, key=lambda r: (r[corner]["vs_a"]["throughput"], r["key"]))
    greedy = sched[0]
    plans_per_batch = greedy["plans_per_batch"]
    batch_time = 1.0 / greedy["tokens_per_s"] / (spec.n_layers + 1)
    receipts = stored_receipts(measurements_path)
    common = {"calibration": calibration, "comparison_on_calibrated_profile": on_validated,
              "receipts": receipts, "current": current}  # fmt: skip
    regime_primary = f"{spec.name}, single-user decoding, 12 GiB GPU"
    primary_scaling = candidate_scaling(
        primary["key"], spec, lab.prior, lab.held,
        lambda n, replicas: default_design(primary["key"], HardwareDesign(modules=n)),
        corner=corner, tokens=lab.tokens, warmup_tokens=lab.warmup, context_len=lab.context_len,
    )  # fmt: skip
    cand_primary = Candidate(primary["key"], primary["name"], regime_primary, model="qwen3-30b-a3b", corner=corner)
    receipts.extend(simulation_receipts(cand_primary, level=sim_level, bindings=sim_bindings, values={
        "throughput_ratio": primary[corner]["vs_a"]["throughput"], "energy_ratio": primary[corner]["vs_a"]["energy"],
        "cost_efficiency_ratio": primary[corner]["vs_a"]["cost_efficiency"]}))  # fmt: skip
    receipts.extend(simulation_receipts(Candidate(primary_scaling["arch"], "", regime_primary, corner=corner),
                                        level=sim_level, bindings=sim_bindings,
                                        values={"scaling_efficiency": primary_scaling["efficiency"]},
                                        raw={"tokens_per_s": primary_scaling["tokens_per_s"]}, baseline=None))  # fmt: skip
    # a hypothetical model shape cannot justify hardware: only published shapes are eligible
    favourable = max((m for m in models if "speedup_vs_a" in m and m["source"].startswith("published")),
                     key=lambda m: m["speedup_vs_a"])  # fmt: skip
    ratios = regime_ratios(favourable["key"], corner=corner, gpu_gib=favourable["gpu_gib"])
    regime_best = (f"{favourable['model']}, single-user decoding, {favourable['gpu_gib']} GiB GPU "
                   "(most favourable regime found among published model shapes)")  # fmt: skip
    best_key = ratios["candidate"]
    cand_best = Candidate(best_key, ARCHITECTURES[best_key].name, regime_best, model=favourable["key"],
                          corner=corner)  # fmt: skip
    receipts.extend(simulation_receipts(cand_best, level=sim_level, bindings=sim_bindings, values={
        "throughput_ratio": ratios["throughput"], "energy_ratio": ratios["energy"],
        "cost_efficiency_ratio": ratios["cost_efficiency"]}))  # fmt: skip
    # filed under the architecture it was measured for: another candidate's scaling never counts
    receipts.extend(simulation_receipts(Candidate(ratios["scaling"]["arch"], "", regime_best, corner=corner),
                                        level=sim_level, bindings=sim_bindings,
                                        values={"scaling_efficiency": ratios["scaling"]["efficiency"]},
                                        raw={"tokens_per_s": ratios["scaling"]["tokens_per_s"]}, baseline=None))  # fmt: skip
    costs = thresholds["plan_evaluation_cost_s"]
    for cand in (cand_primary, cand_best):
        receipts.add(scheduler_receipt(cand, plans_per_batch=plans_per_batch, batch_time_s=batch_time, costs=costs,
                                       limit=thresholds["scheduler_overhead_max"]))  # fmt: skip
    receipts.extend(numerical_receipts(correct, [regime_primary, regime_best], [cand_primary.arch, cand_best.arch]))
    gate_primary = evaluate_gate(cand_primary, thresholds, **common)
    gate_best = evaluate_gate(cand_best, thresholds, **common)
    done("hardware gate")

    dse = None
    if dse_path is not None and Path(dse_path).exists():
        dse = summarise_dse(json.loads(Path(dse_path).read_text()))
    dse_stale: list[str] = []
    if dse is not None:
        dse_stale = stale_reasons(dse["provenance"], dse_provenance_now(dse))
        if dse["base_profile"] != profile_name:
            dse_stale.append(f"explored on `{dse['base_profile']}`, this report uses `{profile_name}`")

    results = {
        "architectures": archs, "critical_path": cp, "imbalance": imbalance, "scaling": scaling,
        "synchronisation": sync, "launch_overhead": launch, "memory_channels": channels,
        "communication": comm, "output_projection": head, "energy": energy, "scheduler": sched,
        "compression": compression, "model_suite": models, "workloads": workloads,
        "correctness": correct, "calibration": calibration.to_json(), "calibration_plan": plan,
        "gate": [gate_primary.to_json(), gate_best.to_json()], "regime_ratios": ratios,
        "receipts": [r.to_json() for r in receipts], "profile_qualification": qualification,
        "calibration_coverage": covered,
        "identities": current,
        "base_profile": profile_name, "calibrated": calibrated, "routing": routing,
        "measured_formats": {f.name: b for f, b in measured_formats.items()},
        "candidate_scaling": {"primary": primary_scaling, "favourable": ratios["scaling"]},
        "dse_stale": dse_stale, "expert_cache": caching, "runtime_verification": verified,
        "backends": [{k: b[k] for k in ("name", "kind", "available", "reason")} for b in backends],
        "dse": None if dse is None else {k: v for k, v in dse.items() if k != "front"},
    }  # fmt: skip

    # -- text ------------------------------------------------------------------------------
    fastest = max(archs, key=lambda r: r["nominal"]["tokens_per_s"])
    cheapest = max(archs, key=lambda r: r["nominal"]["tokens_per_s_per_kusd"])
    d4 = next(r for r in scaling if r["architecture"] == "D expert-local" and r["modules"] == 4)
    e4 = next(r for r in scaling if r["architecture"].startswith("E") and r["modules"] == 4)
    comp = cp["by_component_ms"]
    kinds = cp["by_kind_ms"]
    joint_gain = sched[-1]["speedup"] - 1
    lines_models = [
        [m["model"], f"{m['params_b']:,.1f} B", f"{m['weights_gib']:,.1f}", m["source"], m["gpu_gib"],
         *[f"{m['tokens_per_s'][a]:.3g}" if a in m["tokens_per_s"] else "-" for a in ("A", "D", "E", "F", "H")],
         ("does not fit" if not m["tokens_per_s"]
          else f"{m['speedup_vs_a']:.2f}x ({m['best_module_arch']})" if m.get("speedup_vs_a", 0) > 1.005
          else "none faster"),
         f"{m['per_dollar_vs_a']:.2f}x" if "per_dollar_vs_a" in m else "-"]
        for m in models
    ]  # fmt: skip
    # Commentary is derived from the numbers, so it stays true when the report is regenerated
    # on another profile or with other estimates.
    target = thresholds["scaling_efficiency_min"]
    ideal4, ideal8 = cp["ideal_efficiency"]["4"], cp["ideal_efficiency"]["8"]
    if ideal8 >= target:
        reach = f"the {target:.0%} target is within reach at four and at eight modules if the expert work is balanced"
    elif ideal4 >= target:
        reach = (f"the {target:.0%} target is reachable at four modules only with near-perfect balance, "
                 "and not at eight")  # fmt: skip
    else:
        reach = f"the {target:.0%} target is out of reach even at four modules"
    imb = _by_variant(imbalance)
    rr = imb["round-robin placement"]["tokens_per_s"]
    imb_gain = {k: v["tokens_per_s"] / rr - 1 for k, v in imb.items()}
    slow = {r["key"]: r["nominal"]["vs_a"]["throughput"] for r in archs if r["key"] in ("B", "C")}
    if e4["efficiency"] > d4["efficiency"] + 0.05:
        scaling_note = ("**The low efficiency measured in v0.1 (47%) was a property of expert-local "
                        "placement, not of distributed execution as such.**")  # fmt: skip
    else:
        scaling_note = "Partitioning experts across modules does not scale better than keeping them whole here."
    ch_d, ch_e = _effect(channels, "D: independent"), _effect(channels, "E: independent")
    if max(ch_d, ch_e) <= 0.005:
        channel_note = (
            f"Running each memory channel as an independent engine does not help ({ch_d:+.1%} for D, "
            f"{ch_e:+.1%} for E): a kernel already uses the module's full bandwidth, and splitting it halves "
            "each engine's rate while worsening imbalance. Memory contention is not the bottleneck; memory "
            "bandwidth is.")  # fmt: skip
    else:
        channel_note = (f"Independent channel engines change throughput by {ch_d:+.1%} for D and "
                        f"{ch_e:+.1%} for E.")  # fmt: skip
    en = _by_variant(energy)

    def tpj_change(variant: str, base: str) -> float:
        return en[variant]["tokens_per_joule"] / en[base]["tokens_per_joule"] - 1

    gating_tpj = tpj_change("F: idle devices gated", "F: no power gating")
    energy_note = (
        f"Power gating changes tokens per joule by {gating_tpj:+.1%} for single-user decoding"
        + (": no device idles long enough to be gated, so the modules' static power is paid in full. "
           if abs(gating_tpj) < 0.005 else ". ")
        + f"Energy-aware scheduling (0.01 s per joule) changes the hybrid's tokens per joule by "
        f"{tpj_change('F: energy-aware scheduling (0.01 s per joule)', 'F: no power gating'):+.1%} and the "
        f"conventional system's by "
        f"{tpj_change('A: energy-aware scheduling (0.01 s per joule)', 'A: time-only scheduling'):+.1%}, at "
        f"{_effect(energy, 'A: energy-aware'):+.1%} throughput for the latter.")  # fmt: skip
    plans_ratio = sched[-1]["plans_per_batch"] / sched[0]["plans_per_batch"]
    sched_note = (
        f"Bounded graph-level refinement changed throughput by {joint_gain:+.1%} relative to greedy "
        f"earliest-finish scheduling on this workload while evaluating {plans_ratio:.1f}x as many plans. "
        + ("It does fix constructed collisions (see the test suite), but mixture layers present few of "
           "them; greedy scheduling is the sensible default. " if joint_gain < 0.01 else "")
        + "The overhead of either depends on a per-evaluation cost that only a production implementation "
        "can establish.")  # fmt: skip
    measured_compression = ""
    if measured_formats:
        measured_compression = (f" (a measured profile of real weights, `{Path(ctx.compression_path).name}`, "
                                f"gives {measured_formats[CompressionFormat.ENTROPY]:.2f} bits per weight with the "
                                "best working codec, and is used for the entropy-coded format throughout)")  # fmt: skip
    ran = [m for m in models if "speedup_vs_a" in m]
    no_gain = [m for m in ran if m["speedup_vs_a"] <= 1.005]
    gains = [m["speedup_vs_a"] for m in ran if m["speedup_vs_a"] > 1.005]
    suite_note = ""
    if no_gain:
        fits = all(m["weights_gib"] <= m["gpu_gib"] for m in no_gain)
        suite_note += (f"No module-based architecture is faster than A for {', '.join(m['model'] for m in no_gain)}"
                       + (" (each fits on the GPU). " if fits else ". "))  # fmt: skip
    if gains:
        suite_note += (f"Where one is faster, the best is {min(gains):.2f}x to {max(gains):.2f}x A; per estimated "
                       f"dollar the best ratio anywhere is {max(m['per_dollar_vs_a'] for m in ran):.2f}x.")  # fmt: skip
    unfit = [m["model"] for m in models if not m["tokens_per_s"]]
    if unfit:
        suite_note += f" {', '.join(unfit)} could not be placed on any standard system tried."
    favourable_note = ""
    if favourable["weights_gib"] > favourable["gpu_gib"]:
        favourable_note = (
            f" In the favourable regime the conventional system runs most of a {favourable['weights_gib']:.0f} GiB "
            f"model on its CPU, so those ratios depend directly on the host's assumed arithmetic rate and "
            f"memory bandwidth -- the parameters calibration replaces first.")  # fmt: skip
    lines_work = [
        [w["workload"], w["metric"],
         *[f"{w['tokens_per_s'][a]:,.1f}" if a in w["tokens_per_s"] else "-" for a in ("A", "E", "F", "H")],
         f"{w['speedup_vs_a']:.2f}x ({w['fastest']})"]
        for w in workloads
    ]  # fmt: skip
    machines = ", ".join(f"`{k}` -> `configs/{v}.json`" for k, v in MACHINE_PROFILES.items())
    v_stats = calibration.validation
    if calibrated:
        status_line = (
            f"Every performance number below is simulated. The host and GPU parameters come from "
            f"`configs/{profile_name}.json`, which reproduces {v_stats.n} held-out measurements with a "
            f"median error of {v_stats.median:.1%} (95th percentile {v_stats.p95:.1%}); that validates "
            f"the model of the existing machine only.")  # fmt: skip
    else:
        status_line = (
            f"Every performance number below is the output of a simulator that has **not been "
            f"validated against measurements** (section 2). Host and GPU parameters come from "
            f"`configs/{profile_name}.json`, "
            + ("a profile that has not passed validation." if profile_name in MACHINE_PROFILES.values()
               else "an illustrative profile."))  # fmt: skip
    if gate_primary.go and gate_best.go:
        decision_line = "Every criterion is met in both regimes."
    else:
        blockers = [g.blockers[0] for g in (gate_primary, gate_best) if not g.go]
        decision_line = ("Hardware development is not justified by the evidence so far. First blocker: "
                         + blockers[0].rstrip(".") + ".")  # fmt: skip
    if v_stats.n == 0 and not calibration.calibration.n:
        measured_text = "Nothing has been measured yet."
    else:
        measured_rows = [[r["id"], r["split"], r["measured"],
                          r["predicted"] if r["predicted"] is not None else "-",
                          f"{r['rel_error']:.1%}" if r["rel_error"] is not None else "-"]
                         for r in calibration.per_measurement[:40]]  # fmt: skip
        measured_text = (
            table(["measurement", "split", "measured tokens/s", "predicted tokens/s", "error"], measured_rows)
            + "\n\n" + " ".join([*calibration.reasons, *calibration.notes]))  # fmt: skip
        if calibration.status == "pass" and not on_validated:
            measured_text += (f"\n\nThe validated profiles are not the one this report's comparison runs "
                              f"on (`{profile_name}`); rerun with `--base-profile` set to a validated profile "
                              f"for the evidence to count as calibrated.")  # fmt: skip

    coverage_text = "; ".join(
        f"`{m}` " + ("covers every regime" if c["complete"] else "lacks " + ", ".join(c["missing"]))
        for m, c in covered.items())  # fmt: skip
    if routing.startswith("recorded"):
        routing_line = f"routing for {spec.name} is {routing}; other models use synthetic routing"
    else:
        routing_line = "routing is synthetic (no recorded trace was given)"
    if measured_formats:
        compression_line = (f"the entropy-coded format uses the measured "
                            f"{measured_formats[CompressionFormat.ENTROPY]:.2f} bits per weight")  # fmt: skip
    else:
        compression_line = "compression ratios are assumed"
    md = f"""# BASP Dataflow v0.4 -- evidence-gated heterogeneous execution: simulation and runtime report

> **Read this first.** {status_line} Everything about
> compute-memory modules (performance, power, price) is an estimate with a
> pessimistic-nominal-optimistic range (`configs/design_estimates.json`) that no measurement on
> existing hardware can narrow; {routing_line}; {compression_line}. Figures written `x (a-b)` give
> the nominal value and the range over those estimates. Evidence levels: E0 estimate or argument,
> E1 uncalibrated simulation, E2 simulation validated on existing hardware, E3 measurement.
> Regenerate with `python -m benchmarks.run_report` (deterministic; see `provenance.json`).

## 1. Hardware development decision

**{gate_primary.verdict}** for the project's first workload, and **{gate_best.verdict}** in the
most favourable regime found among published model shapes. {decision_line}

Regime: {gate_primary.regime}

{render_gate(gate_primary)}

Regime: {gate_best.regime}

{render_gate(gate_best)}

Ratios compare the best module-based architecture with architecture A (conventional CPU/GPU with
per-expert caching) at the {corner} corner of every estimate. The thresholds are proposed
acceptance criteria (`configs/hardware_gate.json`), not results. A criterion that cannot be
evaluated counts against the decision.{favourable_note}

## 2. Calibration status

Validation measurements available: **{calibration.validation.n}** (at least
{thresholds["min_validation_measurements"]} required); status: **{calibration.status.replace("_", " ")}**.

{measured_text}

The procedure for the calibration machines is in `docs/calibration.md`; machine names map to
profiles as {machines}. Until `python -m benchmarks.calibrate` has fitted them, these profiles are
priors built from public specifications and assumptions. For orientation, this is what the simulator
currently predicts for conventional layer offloading on the primary machine (a 4.85 bits-per-weight
file is assumed); measurements will show how wrong these are:

{table(["model", "layers on GPU", "predicted tokens/s"],
       [[p["model"], p["gpu_layers"], p["predicted_tokens_per_s"]] for p in plan])}

Validation coverage of the pre-registered plan (regimes the held-out benchmarks exercise; the
simulator is trusted only there): {coverage_text}.

Each machine is judged on its own validation benchmarks; repeated runs of one benchmark count once,
and measurements made with an executable or model file that has since changed are excluded. A profile
counts as validated only while its content is the one the calibration run recorded.

Calibration and validation measurements are separate sets, declared when they are recorded; a
benchmark used to tune a profile cannot also validate it. Calibrating against an existing CPU and
GPU does not narrow any estimate for hardware that does not exist.

## 3. Eight architectures in one simulator

{spec.name} ({spec.total_params / 1e9:.1f} B parameters, {spec.bytes_at() / GiB:.1f} GiB at 4.5 bits per weight),
single-user decoding, context {lab.context_len}; a 12 GiB GPU and, where used, four modules with two DDR5
channels and 1024 MAC lanes each. Ratios are given at the nominal / pessimistic corner.

{render_architectures(archs)}

* Fastest at nominal estimates: **{fastest["key"]} {fastest["name"]}**, {fastest["nominal"]["vs_a"]["throughput"]:.2f}x architecture A.
* Most throughput per dollar: **{cheapest["key"]} {cheapest["name"]}** ({cheapest["nominal"]["tokens_per_s_per_kusd"]:.0f} tokens/s per 1000 USD;
  the best module-based design reaches {max(r["nominal"]["tokens_per_s_per_kusd"] for r in challengers):.0f}).
* B and C run the trunk blocks on memory-bound modules instead of a GPU and reach
  {slow["B"]:.2f}x and {slow["C"]:.2f}x of A.

Prices are estimates for ratio purposes only, and the module-based systems contain more memory and
compute than the conventional one: this compares architectures on their natural hardware.

## 4. Phase 2 investigations

Each experiment changes one thing. The question is whether a weakness is an implementation problem
or a property of the architecture.

### 4.1 Sequential execution: the critical path

Expert-local architecture (D), four modules, per token ({cp["token_time_ms"]:.2f} ms):

{table(["on the critical path", "ms per token", "share"],
       [[k, v, f"{v / cp['token_time_ms']:.0%}"] for k, v in
        [("computing (kernels)", comp["kernel"]), ("waiting for a busy device", comp["queue"]),
         ("returning results", comp["output"]), ("waiting for inputs", comp["input"]),
         ("host round trips", comp["host"])]])}

By operation: experts {kinds.get("expert", 0.0):.2f} ms, trunk blocks {kinds.get("trunk", 0.0):.2f} ms, output projection
{kinds.get("head", 0.0):.2f} ms, reductions {kinds.get("reduce", 0.0):.2f} ms. Only the expert phase
({cp["expert_phase_ms"]:.2f} ms on the path; {cp["expert_work_ms"]:.2f} ms of kernel work in total) can be divided among
modules; the remaining {cp["serial_ms"]:.2f} ms is a serial dependency chain. If the expert work divided
perfectly, four modules would reach {ideal4:.0%} scaling efficiency and eight
{ideal8:.0%}: **{reach}** while the trunk stays serial.

### 4.2 Expert imbalance

{table(["placement", "modules", "tokens/s", "scaling efficiency"],
       [[r["variant"], r["modules"], r["tokens_per_s"], f"{r['efficiency']:.0%}"] for r in imbalance])}

With whole experts on modules, the experts selected for a token land unevenly (a balls-in-bins
effect), and the layer waits for the fullest module. Relative to round-robin placement:
frequency-balanced {imb_gain["frequency-balanced placement"]:+.1%}, co-activation-aware {imb_gain["co-activation-aware placement"]:+.1%}, two copies of every expert
{imb_gain["co-activation-aware, every expert on 2 modules"]:+.1%} (the scheduler picks the emptier holder), tensor-parallel {imb_gain["tensor-parallel (every expert on all modules)"]:+.1%} (no
imbalance by construction). Co-activation-aware placement needs far more profiling data than a
short trace provides and is regularised towards load balancing until then.

### 4.3 Scaling with module count

{table(["architecture", "modules", "tokens/s", "speed-up", "scaling efficiency"],
       [[r["architecture"], r["modules"], r["tokens_per_s"], f"{r['speedup']:.2f}x", f"{r['efficiency']:.0%}"]
        for r in scaling])}

Four-module efficiency: expert-local {d4["efficiency"]:.0%}, tensor-parallel {e4["efficiency"]:.0%}.
{scaling_note}

### 4.4 Synchronisation

{_ab(sync)}

Removing the host round trip gains {_effect(sync, "dependency events only (latencies x1)"):+.1%} at the profile's latencies and
{_effect(sync, "dependency events only (latencies x8)"):+.1%} when they are eight times larger. A lock-step batch waits for its slowest
module at every layer; independent streams avoid that barrier ({_effect(sync, "4 independent streams"):+.1%}).

### 4.5 GPU launch overhead

{_ab(launch)}

v0.1 charged one launch per expert, which understated the conventional baseline by
{_effect(launch, "experts batched"):.0%}. All results in this report use batched launches.

### 4.6 Memory channels

{_ab(channels)}

{channel_note}

### 4.7 Communication

{_ab(comm)}

Streaming the hidden state to the modules before the routing decision overlaps one transfer latency
per layer: worth {_effect(comm, "activations streamed eagerly (latencies x1)"):+.1%} here and {_effect(comm, "activations streamed eagerly (latencies x8)"):+.1%} at eight times the latency.

### 4.8 Output projection

{_ab(head)}

The projection is placed on both the modules and the trunk device when it fits, and the scheduler
picks per token ({_effect(head, "both"):+.1%} against module shards only). v0.1 pinned it to the modules.

### 4.9 Energy

{_ab(energy)}

{energy_note}

### 4.10 Scheduler

{table(["scheduler", "tokens/s", "plan evaluations per batch", "improvements found",
        "overhead at 50 / 200 / 1000 ns per evaluation"],
       [[r["variant"], r["tokens_per_s"], r["plans_per_batch"], r["joint_improvements"],
         " / ".join(f"{v:.1%}" for v in r["overhead"].values())] for r in sched])}

{sched_note}

### 4.11 Weight compression

Order-0 entropy of INT4 codes plus scales, on synthetic weights{measured_compression}:

{table(["weight distribution (synthetic)", "bits per weight", "saving vs 4.5-bit INT4"],
       [[k, v["entropy_bits_per_weight"], f"{v['saving_vs_int4']:.1%}"] for k, v in compression.items()])}

Real weights have not been measured. `python -m benchmarks.profile_compression` reports the achieved
ratio and software decode throughput for a weight file; decode throughput of a hardware decoder
remains an estimate.

## 5. Model-size suite: where does each architecture become advantageous?

Single-user decoding on the smallest standard system that holds each model (a 12 GiB GPU, or the
next size up when the trunk blocks alone do not fit; host memory and four modules sized to the
model). Tokens per second; systems for the largest models are hypothetical.

{table(["model", "parameters", "weights (GiB)", "shape", "GPU (GiB)", "A", "D", "E", "F", "H",
        "best module arch. vs A", "per dollar vs A"], lines_models)}

{suite_note}

## 6. Workload types

{spec.name}; each workload is a separate question with its own metric.

{table(["workload", "metric", "A", "E", "F", "H", "best vs A"], lines_work)}

## 7. Design-space exploration
"""
    if dse is None:
        md += "\nNo exploration results found; run `python -m benchmarks.explore` first.\n"
    else:
        front = dse["front"]
        shown = front if len(front) <= 14 else [front[round(i * (len(front) - 1) / 13)] for i in range(14)]
        front_rows = []
        for e in shown:
            m, c = e["metrics"], e["corners"]
            low = c["pessimistic"]["tokens_per_s"] if c.get("pessimistic") else float("nan")
            high_cost = c["pessimistic"]["cost_usd"] if c.get("pessimistic") else float("nan")
            front_rows.append([_design_text(e["config"]), f"{m['tokens_per_s']:.0f} ({low:.0f})",
                               f"{m['tokens_per_joule']:.2f}", f"{m['cost_usd']:,.0f} ({high_cost:,.0f})",
                               f"{m['tokens_per_s_per_kusd']:.1f}"])  # fmt: skip
        arch_rows = []
        for arch, info in dse["per_arch"].items():
            f, d = info["fastest"], info["best_per_dollar"]
            arch_rows.append([f"{arch} {ARCHITECTURES[arch].name}", info["evaluated"],
                              f"{f['metrics']['tokens_per_s']:.0f}", _design_text(f["config"]),
                              f"{d['metrics']['tokens_per_s_per_kusd']:.1f}",
                              dse["front_archs"].get(arch, 0)])  # fmt: skip
        cost_rows = []
        for budget, info in dse["equal_cost"].items():
            b, c = info["best"], info["best_conventional"]
            cost_rows.append([f"{int(budget):,} USD", _design_text(b["config"]),
                              f"{b['metrics']['tokens_per_s']:.0f}",
                              f"{c['metrics']['tokens_per_s']:.0f}" if c else "-"])  # fmt: skip
        power_rows = []
        for watts, info in dse["equal_power"].items():
            b, c = info["best"], info["best_conventional"]
            power_rows.append([f"{watts} W", _design_text(b["config"]), f"{b['metrics']['tokens_per_s']:.0f}",
                               f"{c['metrics']['tokens_per_s']:.0f}" if c else "-"])  # fmt: skip
        on_front = {a: n for a, n in dse["front_archs"].items() if n}
        absent = [a for a in dse["front_archs"] if a not in on_front]
        composition = ", ".join(f"{n} {ARCHITECTURES[a].name.lower()} ({a})" for a, n in on_front.items())
        top = max(front, key=lambda e: e["metrics"]["tokens_per_s"])
        conventional = [e for e in front if e["config"]["arch"] == "A"]
        if conventional and top["config"]["arch"] != "A":
            ref = max(conventional, key=lambda e: e["metrics"]["tokens_per_s"])
            worst = top["corners"].get("pessimistic") or top["metrics"]
            ref_worst = ref["corners"].get("pessimistic") or ref["metrics"]
            verdict = (
                f"The fastest design found is {top['metrics']['tokens_per_s'] / ref['metrics']['tokens_per_s']:.2f}x "
                f"the fastest conventional system at {top['metrics']['cost_usd'] / ref['metrics']['cost_usd']:.1f}x "
                f"its estimated price ({worst['tokens_per_s'] / ref_worst['tokens_per_s']:.2f}x at "
                f"{worst['cost_usd'] / ref_worst['cost_usd']:.1f}x the price when both are priced with "
                f"pessimistic estimates).")  # fmt: skip
        elif conventional:
            verdict = "The fastest design found is a conventional system."
        else:
            verdict = "No conventional system is on the front."
        stale = ("\n> **STALE -- do not draw conclusions from this section.** The stored exploration no "
                 f"longer matches the repository: {'; '.join(dse_stale)}. Rerun `python -m benchmarks.explore`.\n"
                 if dse_stale else "")  # fmt: skip
        if dse_stale:
            verdict = "(No verdict: the exploration is stale.)"
        md += f"""{stale}
{dse["evaluated"]:,} designs were simulated ({dse["method"]} search) out of {dse["feasible_designs"]:,} feasible
combinations of architecture, GPU memory, module count, module memory, memory channels, MAC lanes,
link bandwidth and expert replication ({dse["space_size"]:,} before constraints), on the 30B MoE
single-user workload. Objectives: tokens per second, tokens per joule, estimated price. Each design
gets a short run ({dse["workload"].get("tokens", "?")} tokens, {dse["workload"].get("warmup_tokens", "?")} of them warm-up), so absolute figures differ by a few percent
from section 3.

Pareto front ({dse["front_size"]} designs; a spread of them, cheapest first; the pessimistic-corner
value is in brackets):

{table(["design", "tokens/s", "tokens/J", "est. price USD", "tokens/s per k$"], front_rows)}

By architecture:

{table(["architecture", "designs evaluated", "best tokens/s", "fastest design", "best tokens/s per k$",
        "designs on the front"], arch_rows)}

Equal cost -- the fastest design within a price budget, against the fastest conventional one:

{table(["budget", "fastest design", "tokens/s", "fastest conventional (A) tokens/s"], cost_rows)}

Equal power -- the fastest design within an average power budget:

{table(["budget", "fastest design", "tokens/s", "fastest conventional (A) tokens/s"], power_rows)}

The front holds {composition}; no {", ".join(absent) if absent else "other"} design is on it. {verdict}
The conventional candidates are one GPU per system, in the classes of `configs/design_estimates.json`
(12, 24 and 32 GiB here, each with its own bandwidth and compute range); systems with several GPUs,
which a buyer with the same budget could choose, are not modelled, so the equal-cost rows above are
still generous to the module-based designs.
"""
    runs = correct["runs"]
    gate_max = thresholds["numeric_rel_l2_max"]
    available = ", ".join(b["name"] for b in backends if b["available"])
    missing = "; ".join(f"{b['kind']}: {b['reason']}" for b in backends if not b["available"])
    if verified:
        vrows = []
        for v in verified:
            ppl = v["perplexity"] or {}
            routing_text = ("dense" if v["routing"] is None else
                            "identical" if v["routing"]["exact"] else f"{v['routing']['mismatches']} differ")  # fmt: skip
            vrows.append([f"{v['model']} (`{v['file']}`)", f"{v['numerical_error']:.1e}",
                          f"{v['hidden_rel_l2_max']:.1e} over {v['layers']} layers", routing_text,
                          f"{v['generated_tokens']} identical" if v["generation_identical"] else "diverge",
                          f"{ppl['relative']:.6f}" if ppl else "-",
                          v["verdict"].upper() + (" (stale: runtime code changed since)" if v["stale"] else "")])  # fmt: skip
        first = verified[0]
        quant = first["quantisation"]
        quant_text = ""
        if quant:
            quant_text = (f" For information, the quantised file against the original checkpoint (not a property of "
                          f"the runtime): perplexity {quant['quantised_runtime_perplexity']:.2f} vs "
                          f"{quant['original_perplexity']:.2f} on the same {first['perplexity']['tokens']} tokens, greedy "
                          f"choice identical for {quant['logits']['argmax_agreement']:.0%} of positions; a text this short "
                          "does not measure quality.")  # fmt: skip
        verified_text = (table(["model", "logits rel. L2 (max)", "layer outputs rel. L2 (max)", "expert routing",
                                "greedy tokens", "perplexity ratio", "verdict"], vrows)
                         + f"\n\nPrompt `{first['prompt']}` -> `{first['generated_text']}` (greedy, "
                         f"{first['decode_tokens_per_s']:.0f} tokens/s on the verification machine's CPU in NumPy)."
                         + quant_text)  # fmt: skip
    else:
        verified_text = ("No runtime verification has been recorded (`python -m benchmarks.verify_runtime`).")
    runtime_section = f"""## 9. The physical runtime and its verification

v0.4 adds a runtime that executes real models, separate from the simulator: it reads GGUF files
(dequantising F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K and Q6_K, bit-identical to the
reference `gguf` package), runs Qwen3 dense and mixture-of-experts models through an execution plan
on the available backends, and keeps a KV cache per session. Backends available: {available}.
Declared but not implemented, and refusing every call: {missing}. No GPU inference exists in this
release, and nothing below says anything about module hardware.

Verification runs the same GGUF weights in transformers (float32; weights dequantised by the reference
package) and compares logits for every prompt position, every layer's output, every expert selection,
16 greedy tokens through the KV cache, and perplexity on a fixed text (target: relative L2 <= {gate_max:g},
perplexity ratio <= {thresholds["model_quality_relative_perplexity_max"]:g}):

{verified_text}

These are measurements (E3) of the NumPy runtime on the model named in each row. They do not
qualify the gate's numerical-correctness or model-quality criteria for {spec.name}: the receipts are
bound to the model they measured, and the gate requires the model its regime is about.
CI repeats the comparison on small random-weight dense and MoE models, and checks that a deliberately
broken runtime fails it."""
    crow = [r for r in caching["rows"]]
    by_cap: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in crow:
        by_cap[r["capacity_gib"]].append(r)
    cache_rows = [[f"{r['capacity_gib']} GiB", r["policy"], f"{r['hit_rate']:.0%}", f"{r['cpu_share']:.0%}",
                   f"{r['mb_migrated_per_token']:.0f}", f"{r['ms_per_token']:.1f}"]
                  for r in crow if r["policy"] not in ("unbounded cache",)]  # fmt: skip
    bounds = next(r for r in crow if r["policy"] == "unbounded cache")
    best = {cap: min((r for r in rows if r["policy"] in ("LRU", "LFU", "cost-aware", "predictive")),
                     key=lambda r: r["ms_per_token"]) for cap, rows in by_cap.items()}  # fmt: skip
    lru = {cap: next(r for r in rows if r["policy"] == "LRU") for cap, rows in by_cap.items()}
    gains = ", ".join(f"{cap} GiB: {best[cap]['policy']}"
                      + (f" ({lru[cap]['ms_per_token'] / best[cap]['ms_per_token']:.2f}x faster than LRU)"
                         if best[cap]["policy"] != "LRU" else "") for cap in sorted(best))  # fmt: skip
    aware = {cap: next(r for r in rows if r["policy"] == "cost-aware") for cap, rows in by_cap.items()}
    traffic = ", ".join(f"{lru[cap]['mb_migrated_per_token']:.0f} vs {aware[cap]['mb_migrated_per_token']:.0f} MB "
                        f"at {cap} GiB" for cap in sorted(by_cap))  # fmt: skip
    cache_section = f"""## 10. Memory domains and expert caching

Memory is managed per physical domain (`basp.memory`): devices that share memory (the M2's CPU
and GPU) share one pool, every allocation is charged to one of weights, KV cache, activations or
workspace, and weights + KV + activations + workspace + reserved never exceed the physical capacity
(checked after every operation in the tests). Requests are admitted only if their whole KV cache fits
(unreferenced cached experts are evicted first, and under memory pressure a margin is kept free);
blocks in use or being migrated are reference-counted and cannot be evicted.

Expert caching on the GPU of `{caching["profile"]}`, for {spec.name} ({caching["all_experts_gib"]:.1f} GiB
of experts at 4.5 bits per weight, {caching["expert_mib"]:.2f} MiB each): a miss either runs on the CPU
({caching["cost"]["t_cpu_us"]:.0f} us) or migrates the expert over PCIe ({caching["cost"]["t_migration_us"]:.0f} us)
and runs on the GPU ({caching["cost"]["t_gpu_us"]:.1f} us). Cost-aware caching values an expert at
`V_e = (N_e (T_cpu - T_gpu) - T_migration) / M_e`; predictive caching estimates `N_e` from the measured
next-step repeat rate. Expert time per token over {caching["steps"]} synthetic decoding steps (E1; no overlap of
migration with computation for any policy):

{table(["GPU expert cache", "policy", "hit rate", "run on CPU", "MB migrated per token", "expert ms per token"], cache_rows)}

An unbounded cache (every expert migrated once) needs {bounds["ms_per_token"]:.1f} ms per token over this
trace. Fastest policy at each capacity: {gains}. LRU migrates far more data per token than the
cost-aware policy ({traffic}), which instead runs rarely used experts on the CPU; whether that wins
depends on how much of the working set the cache holds. The cost-aware horizon (32 steps of expected
reuse) is a parameter, not tuned on this trace. These are simulated costs on a synthetic trace, not
runtime measurements."""
    md += f"""
## 8. Numerical verification of the simulated dataflows

A small attention-free MoE language model with real INT4 weights is decoded by the streaming
kernels following the scheduled plans, and compared with an independent float64 reference.

{table(["dataflow", "max relative error", "routing identical", "tokens identical",
        "dependency violations", "accounting violations"],
       [[r["dataflow"], f"{r['max_rel_l2']:.2e}", r["routing_matches"], r["tokens_match"],
         r["dependency_violations"], r["accounting_violations"]] for r in runs])}

With the reduction order fixed, fused and unfused graphs give bit-identical logits:
{correct["fused_equals_unfused"]}. Dataflows that accumulate partial sums on devices, or partition experts across
modules, add the same FP32 terms in a different order: the three runs above differ from each other
by at most {correct["max_abs_diff_between_dataflows"]:.1e} on logits of magnitude {correct["logit_scale"]:.1f}, and select the same experts.
Such differences are at rounding level but are not zero, and a reordered sum could in principle
tip a near-tied routing decision; none was observed. This co-simulation does not run real weights;
section 9 does.

{runtime_section}

{cache_section}

## 11. What changed since earlier reports

New in v0.4: the gate decides each criterion from **metric-specific receipts** bound to the
identities that produced them (simulator source, configuration, profile, validation, model weights,
runtime, benchmark), so a throughput result cannot stand in for energy, a stale result is rejected,
and a result about another model cannot qualify the regime. Calibration merges repeated runs of a
benchmark into one data point, judges each machine on its own, excludes measurements made with an
executable or model file that has since changed, and qualifies a profile only if it is still the
profile that was validated. A physical runtime (section 9) executes real GGUF models; memory is
managed in physical domains with shared accounting, admission control and safe eviction (section 10).
Repaired: designs built on a unified-memory machine (Apple M2) split its one memory domain in two
and were rejected; they now share one pool, and the integrated GPU is not priced separately.

Corrected in v0.3 (each found in an audit of v0.2 or while fixing it, and covered by a regression test):

* Module links on the local fabric were free: every hierarchical design on the v0.2 front used the
  cheapest link, because the simulator gave it the fabric's bandwidth anyway.
* Price corners favoured modules: the pessimistic corner priced host and GPU high. It now prices them
  low (the strongest baseline), and GPU classes differ in speed, not only in memory.
* A prompt was simulated as independent sequences, each with its own KV cache; it is now one sequence.
* The host CPU's arithmetic rate (80 GMAC/s in v0.2, 30 in v0.1) made prompt processing several times
  too slow; it is 400 GMAC/s, with generation still memory-bound.
* The gate qualified one architecture with another's scaling result; every criterion now concerns one
  candidate in one regime, at a required evidence level, and borrowed evidence counts as missing.
* Unified memory (Apple M2) is one shared bus and one capacity pool, not two.

Corrected in v0.2: batched GPU launches; the output projection's location chosen per token; the v0.1
claim of bit-identical logits under every policy (true only while the reduction order is unchanged).

The reports in `reports/v0.1/`, `reports/v0.2/` and `reports/v0.3/` were produced by earlier models and
are kept as records; their numbers are superseded by this report.
"""
    return md, results


def write_report(outdir: str | Path = "reports/v0.4", *, quick: bool = False, progress=None,
                 dse_path: str | Path | None = "reports/v0.4/dse.json",
                 measurements_path: str | Path | None = "evidence", base: str | None = None,
                 trace_path: str | Path | None = None,
                 compression_path: str | Path | None = None) -> tuple[Path, Path]:  # fmt: skip
    """Write ``report.md`` and ``results.json`` (deterministic) and ``provenance.json``."""
    outdir = Path(outdir)
    markdown, results = build_report(quick=quick, progress=progress, dse_path=dse_path,
                                     measurements_path=measurements_path, base=base, trace_path=trace_path,
                                     compression_path=compression_path)  # fmt: skip
    record = results.pop("provenance")
    # what the deterministic files depend on, without the machine-specific environment
    results["inputs"] = {"source": record["source"], "configs": record["configs"], "inputs": record["inputs"]}
    outdir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = outdir / "report.md", outdir / "results.json"
    md_path.write_text(markdown)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=float) + "\n")
    (outdir / "provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return md_path, json_path


__all__ = ["SUITE", "HardwareDesign", "build_report", "summarise_dse", "write_report"]
