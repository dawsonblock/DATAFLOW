"""Calibration coverage: which execution regimes a machine's validation split exercises.

A validated simulator is trusted only in regimes its held-out benchmarks covered.  Each planned
run is classified along the dimensions the simulator models differently -- where layers run
(CPU only, split, GPU only), whether experts are split off to the CPU, long context, dense or
mixture-of-experts -- and ``coverage`` lists what the validation split covers, what only the
calibration split covers (in-sample: not validated), and what is missing.  It reads the plan;
it never changes it.
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
