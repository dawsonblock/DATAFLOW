"""Architecture comparison, model-size suite and workload-type suite.

The same simulator, hardware estimates and synthetic routing are used for every
row.  Module-based architectures are evaluated at the pessimistic, nominal and
optimistic corner of the estimates, so each of their figures is a range.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from basp.core.tensor import CompressionFormat
from basp.dse.hardware import CORNERS, HardwareDesign
from basp.experiments.architectures import ARCHITECTURES
from basp.experiments.investigations import Lab
from basp.experiments.variants import (
    Outcome,
    design_for,
    routing_provenance,
    run_architecture,
    traces,
)
from basp.workloads.model import MoEModelSpec
from basp.workloads.zoo import SUITE

GiB = 1024**3
#: copies of each expert on distinct modules where the architecture can use them
REPLICAS = {"A": 1, "B": 2, "C": 1, "D": 2, "E": 1, "F": 2, "G": 2, "H": 2}


def default_design(key: str, base: HardwareDesign | None = None) -> HardwareDesign:
    """The standard four-module (or module-free) system each architecture is shown on."""
    base = base or HardwareDesign(modules=4)
    if key == "A":
        return replace(base, modules=0)
    return replace(base, gpu_gib=0 if key in ("B", "C") else base.gpu_gib,
                   fabric="local" if key == "H" else "star", persistent=key == "G")  # fmt: skip


def _metrics(out: Outcome) -> dict[str, float]:
    return {
        "tokens_per_s": out.result.tokens_per_s,
        "tokens_per_joule": out.result.tokens_per_joule,
        "average_power_w": out.result.average_power_w,
        "cost_usd": out.cost_usd,
        "tokens_per_s_per_kusd": out.per_kusd,
        "link_mb_per_token": out.result.link_bytes_per_token / 1e6,
    }


def architecture_table(lab: Lab) -> list[dict[str, Any]]:
    """All eight architectures on the 30B MoE workload, with estimate ranges."""
    rows = []
    for key, arch in ARCHITECTURES.items():
        design = default_design(key)
        corners = {}
        for corner in CORNERS:
            corners[corner] = _metrics(lab.run(key, design, replicas=REPLICAS[key], corner=corner))
        rows.append({"key": key, "name": arch.name, "concept": arch.concept,
                     "design": design.as_dict(), "replicas": REPLICAS[key], **corners})  # fmt: skip
    baseline = rows[0]
    for row in rows:
        for corner in CORNERS:
            ref, own = baseline[corner], row[corner]
            own["vs_a"] = {
                "throughput": own["tokens_per_s"] / ref["tokens_per_s"],
                "energy": own["tokens_per_joule"] / ref["tokens_per_joule"],
                "cost_efficiency": own["tokens_per_s_per_kusd"] / ref["tokens_per_s_per_kusd"],
            }
    return rows


def _try(key: str, design: HardwareDesign, spec: MoEModelSpec, prior, held, **kw) -> Outcome | None:
    try:
        return run_architecture(key, design, spec, prior, held, replicas=REPLICAS[key], **kw)
    except Exception as exc:
        if type(exc).__name__ in ("PlacementError", "ArchitectureError", "DesignError",
                                  "SchedulingError", "OutOfMemory"):  # fmt: skip
            return None
        raise


def suite_entries() -> list[tuple[str, MoEModelSpec, str, float]]:
    """(key, shape, purpose, bits per weight) for every model of the suite, INT8 variant included."""
    entries: list[tuple[str, MoEModelSpec, str, float]] = [(k, s, p, 4.5) for k, s, p in SUITE]
    entries.insert(3, ("qwen3-14b-int8", replace(SUITE[2][1], name="Qwen3-14B (INT8)",
                                                 weight_format=CompressionFormat.INT8),
                       "memory pressure and quantization", 8.5))  # fmt: skip
    return entries


def suite_spec(key: str) -> tuple[MoEModelSpec, float]:
    """The shape and bits per weight the suite uses for ``key``."""
    for k, spec, _, bits in suite_entries():
        if k == key:
            return spec, bits
    raise KeyError(key)


def model_suite(architectures=("A", "D", "E", "F", "H"), *, tokens: int = 4, warmup_tokens: int = 1,
                keys: tuple[str, ...] | None = None,
                gpu_sizes: tuple[int, ...] = (12, 24, 48)) -> list[dict[str, Any]]:  # fmt: skip
    """Single-user decoding across model sizes: where does each architecture become advantageous?

    Every model runs on the smallest standard system that can hold it: a 12 GiB
    GPU (the next size up when the trunk blocks alone do not fit), host memory
    sized to the model, and four modules sized to the model.  Systems for the
    largest models are hypothetical.
    """
    rows = []
    for key, spec, purpose, bits in suite_entries():
        if keys is not None and key not in keys:
            continue
        prior, held = traces(spec, tokens=tokens, prior_steps=8)
        row: dict[str, Any] = {
            "model": spec.name, "key": key, "purpose": purpose, "source": spec.source,
            "params_b": spec.total_params / 1e9, "active_params_b": spec.active_params / 1e9,
            "bits_per_weight": bits, "weights_gib": spec.bytes_at(bits) / GiB, "tokens_per_s": {},
            "per_kusd": {}, "cost_usd": {},
        }  # fmt: skip
        for gpu_gib in gpu_sizes:
            row["gpu_gib"] = gpu_gib
            for arch in architectures:
                design = default_design(arch, design_for(spec, bits_per_weight=bits, gpu_gib=gpu_gib,
                                                         replicas=REPLICAS[arch]))  # fmt: skip
                out = _try(arch, design, spec, prior, held, tokens=tokens, warmup_tokens=warmup_tokens,
                           context_len=2048)  # fmt: skip
                if out is not None:
                    row["tokens_per_s"][arch] = out.result.tokens_per_s
                    row["per_kusd"][arch] = out.per_kusd
                    row["cost_usd"][arch] = out.cost_usd
            if row["tokens_per_s"]:
                break  # the smallest GPU on which anything runs
        speeds = row["tokens_per_s"]
        if speeds:
            # ties go to the conventional system: a module that adds nothing is not "fastest"
            row["fastest"] = max(speeds, key=lambda a: (round(speeds[a], 6), a == "A", a))
            row["best_per_dollar"] = max(row["per_kusd"], key=lambda a: (row["per_kusd"][a], a))
            if "A" in speeds:
                others = {a: v for a, v in speeds.items() if a != "A"}
                if others:
                    best = max(others, key=lambda a: (others[a], a))
                    row["best_module_arch"] = best
                    row["speedup_vs_a"] = others[best] / speeds["A"]
                    row["per_dollar_vs_a"] = max(
                        v for a, v in row["per_kusd"].items() if a != "A"
                    ) / row["per_kusd"]["A"]
        rows.append(row)
    return rows


def workload_suite(spec: MoEModelSpec, architectures=("A", "E", "F", "H"), *,
                   quick: bool = False) -> list[dict[str, Any]]:  # fmt: skip
    """The six workload types on one model; each is a separate question with its own metric."""
    n = 2 if quick else 4
    prompt = 32 if quick else 128
    sustained = 12 if quick else 48
    if quick:
        architectures = tuple(a for a in architectures if a in ("A", "H"))
    cases = [
        ("single-user decoding", {"tokens": 6, "warmup_tokens": 2}, "tokens/s"),
        (f"batched inference ({n} sequences in lock step)",
         {"tokens": 4, "warmup_tokens": 1, "batch": n}, "aggregate tokens/s"),
        (f"concurrent requests ({n} independent streams)",
         {"tokens": 4, "warmup_tokens": 1, "concurrent": n}, "aggregate tokens/s"),
        ("long-context attention (32k context)",
         {"tokens": 6, "warmup_tokens": 2, "context_len": 32768}, "tokens/s"),
        (f"prefill ({prompt}-token prompt)",
         {"tokens": 1, "warmup_tokens": 0, "batch": prompt, "context_len": 0, "prompt": True},
         "prompt tokens/s"),
        (f"sustained decoding ({sustained} tokens)",
         {"tokens": sustained, "warmup_tokens": sustained // 4}, "tokens/s"),
    ]  # fmt: skip
    rows = []
    for name, options, metric in cases:
        options = dict(options)
        rows_in_trace = max(options.get("batch", 1), options.get("concurrent", 0))
        # windows of many rows can be longer than a recording: fall back to synthetic routing
        # explicitly, and say so in the row
        prior, held = traces(spec, tokens=options["tokens"], prior_steps=16 if rows_in_trace > 8 else 32,
                             batch=rows_in_trace, fallback="synthetic")  # fmt: skip
        used = routing_provenance()[-1]
        options.setdefault("context_len", 2048)
        row: dict[str, Any] = {"workload": name, "metric": metric, "tokens_per_s": {},
                               "latency_ms": {}, "tokens_per_joule": {},
                               "routing": used["source"] + (f" ({used['reason']})" if used["reason"] else "")}  # fmt: skip
        for arch in architectures:
            out = run_architecture(arch, default_design(arch), spec, prior, held,
                                   replicas=REPLICAS[arch], **options)  # fmt: skip
            row["tokens_per_s"][arch] = out.result.tokens_per_s
            row["latency_ms"][arch] = out.result.token_latency_s * 1e3
            row["tokens_per_joule"][arch] = out.result.tokens_per_joule
        speeds = row["tokens_per_s"]
        row["fastest"] = max(speeds, key=lambda a: (speeds[a], a))
        row["speedup_vs_a"] = max(v for a, v in speeds.items() if a != "A") / speeds["A"]
        rows.append(row)
    return rows
