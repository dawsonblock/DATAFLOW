"""Calibration coverage: which execution regimes a machine's validation actually exercised.

A validated simulator is trusted only in regimes its held-out benchmarks covered.  Each
benchmark is classified along the dimensions the simulator models differently -- where layers
run (CPU only, split, GPU only), whether experts are split off to the CPU, long context, dense
or mixture-of-experts -- and by what it measured (decode or prefill throughput).

Two different things are reported and must not be confused:

* ``coverage(plan runs)`` -- what the pre-registered plan *would* cover.  A plan confers nothing.
* ``observed_coverage(scored rows)`` -- what successful validation observations *do* cover.
  Only this qualifies anything.

``required_regimes`` says what a comparison relies on, and ``coverage_gaps`` what of it the
observations do not establish: a dense-model validation does not qualify a mixture-of-experts
comparison on the same machine and the same validated profile; a timing validation
establishes nothing about energy.  It reads results and the plan; it never changes either.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from basp.workloads.zoo import MODELS

#: every regime a decoding comparison may rely on
REGIMES = ("cpu-only", "split-layers", "gpu-only", "experts-on-cpu", "long-context", "dense", "moe")


def regimes_of(run: Mapping[str, Any]) -> set[str]:
    spec = MODELS.get(run["model"])
    n_layers = spec.n_layers if spec is not None else None
    ngl = int(run["ngl"])
    out = set()
    if ngl == 0:
        out.add("cpu-only")
    elif n_layers is not None and ngl < n_layers:
        out.add("split-layers")
    else:
        out.add("gpu-only")
    if int(run.get("n_cpu_moe", 0) or 0) > 0:
        out.add("experts-on-cpu")
    if int(run.get("depth", 0) or 0) >= 2048:
        out.add("long-context")
    if spec is not None:
        out.add("dense" if spec.dense else "moe")
    return out


def coverage(runs: list[Mapping[str, Any]], *, required: tuple[str, ...] = REGIMES) -> dict[str, Any]:
    """``runs``: planned runs (``benchmarks.run_calibration.plan_runs``) of one machine."""
    validated: set[str] = set()
    in_sample: set[str] = set()
    for run in runs:
        (validated if run["split"] == "validation" else in_sample).update(regimes_of(run))
    missing = [r for r in required if r not in validated]
    return {"validated": sorted(validated), "calibration_only": sorted(in_sample - validated),
            "missing": missing, "complete": not missing}  # fmt: skip


def regime_covered(regime: set[str], report: Mapping[str, Any]) -> list[str]:
    """Parts of ``regime`` (a set of regime names) the validation split does not cover."""
    return sorted(r for r in regime if r not in set(report["validated"]))


#: what a validation measurement's kind establishes
MEASURED = {"decode": "decode", "prefill": "prefill"}
#: a metric of the comparison needs more than the regime: simulated energy rests on power
#: parameters that no timing benchmark validates
METRIC_REQUIRES: dict[str, tuple[str, ...]] = {"energy_ratio": ("energy",)}


def observed_coverage(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Regimes established by *successful validation observations* of one machine.

    ``rows``: ``CalibrationReport.per_measurement``.  A row counts if it is in the validation
    split and was both measured and predicted.  For each regime tag: how many observations, and
    their median and worst relative error.
    """
    errors: dict[str, list[float]] = {}
    for row in rows:
        if row.get("split") != "validation" or row.get("rel_error") is None:
            continue
        tags = regimes_of({"model": row["model"], "ngl": row.get("n_gpu_layers", 0),
                           "n_cpu_moe": row.get("n_cpu_moe", 0), "depth": row.get("context_len", 0)})  # fmt: skip
        tags.add(MEASURED.get(row.get("kind"), str(row.get("kind"))))
        for tag in tags:
            errors.setdefault(tag, []).append(float(row["rel_error"]))
    out = {}
    for tag, values in sorted(errors.items()):
        ordered = sorted(values)
        out[tag] = {"n": len(ordered), "median": ordered[len(ordered) // 2] if len(ordered) % 2 else
                    (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2, "worst": ordered[-1]}  # fmt: skip
    return out


def required_regimes(spec, *, gpu_gib: float, bits_per_weight: float = 4.5, context_len: int = 0,
                     kind: str = "decode") -> tuple[str, ...]:  # fmt: skip
    """What a comparison on ``spec`` with a ``gpu_gib`` GPU relies on the simulator for.

    The conventional baseline keeps a model that fits entirely on the GPU; a dense model that
    does not fit is split by layers; a mixture-of-experts model that does not fit keeps its
    experts on the CPU.  Long contexts and the measured quantity are regimes of their own.
    """
    fits = spec.bytes_at(bits_per_weight) <= gpu_gib * 1024**3
    tags = {"dense" if spec.dense else "moe", kind}
    if fits:
        tags.add("gpu-only")
    else:
        tags.add("split-layers" if spec.dense else "experts-on-cpu")
    if context_len >= 2048:
        tags.add("long-context")
    return tuple(sorted(tags))


def coverage_gaps(required, observed: Mapping[str, Mapping[str, Any]] | None, *, min_per_regime: int = 1,
                  metric: str | None = None) -> list[str]:  # fmt: skip
    """Requirements the observations do not establish (empty: covered)."""
    observed = observed or {}
    needed = [*required, *METRIC_REQUIRES.get(metric or "", ())]
    return [tag for tag in needed if int(observed.get(tag, {}).get("n", 0)) < min_per_regime]
