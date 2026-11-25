"""Predictive-accuracy statistics for the simulator calibration criterion of the hardware gate.

The simulator is tuned on the calibration split and judged on the validation split only.  The
in-sample (calibration) error is reported for information; it can never make a failing
validation result pass.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from basp.calibration.measurements import Measurement, MeasurementSet


@dataclass(frozen=True)
class ErrorStats:
    """Relative errors ``|predicted - measured| / measured`` of a set of predictions.

    ``mean_signed`` is the mean of ``(predicted - measured) / measured`` and exposes bias: positive
    when the simulator is optimistic (predicts more throughput than was measured).
    """

    n: int
    median: float
    p95: float
    worst: float
    mean_signed: float


def error_stats(pairs: Sequence[tuple[float, float]]) -> ErrorStats:
    """Statistics over ``(predicted, measured)`` pairs; p95 uses linear interpolation."""
    if len(pairs) == 0:
        nan = float("nan")
        return ErrorStats(0, nan, nan, nan, nan)
    predicted = np.array([p for p, _ in pairs], dtype=np.float64)
    measured = np.array([m for _, m in pairs], dtype=np.float64)
    if np.any(measured <= 0) or not np.all(np.isfinite(measured)):
        raise ValueError("measured values must be positive and finite")
    signed = (predicted - measured) / measured
    rel = np.abs(signed)
    return ErrorStats(
        n=len(pairs),
        median=float(np.median(rel)),
        p95=float(np.percentile(rel, 95)),
        worst=float(rel.max()),
        mean_signed=float(signed.mean()),
    )


@dataclass
class CalibrationReport:
    """Outcome of comparing simulator predictions with measurements.

    ``status`` is decided by the validation split alone.  ``calibration`` is the in-sample error
    of the measurements the simulator was tuned on and is informational.
    """

    calibration: ErrorStats
    validation: ErrorStats
    per_measurement: list[dict[str, Any]]
    thresholds: dict[str, float]
    status: str  # "pass" | "fail" | "not_evaluated"
    reasons: list[str] = field(default_factory=list)  # why the status is not "pass"
    notes: list[str] = field(default_factory=list)  # informational, never affects the status
    excluded: list[dict[str, Any]] = field(default_factory=list)  # measurements not counted, and why
    by_machine: dict[str, CalibrationReport] = field(default_factory=dict)  # each machine on its own

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_json(self) -> dict[str, Any]:
        def clean(stats: ErrorStats) -> dict[str, Any]:
            return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in asdict(stats).items()}  # fmt: skip

        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "notes": list(self.notes),
            "thresholds": dict(self.thresholds),
            "validation": clean(self.validation),
            "calibration_in_sample": clean(self.calibration),
            "per_measurement": self.per_measurement,
            "excluded": self.excluded,
            "by_machine": {m: r.to_json() for m, r in self.by_machine.items()},
        }


def _usable(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def evaluate_calibration(
    predict: Callable[[Measurement], float | None],
    measurements: MeasurementSet,
    *,
    median_max: float = 0.10,
    p95_max: float = 0.20,
    min_validation: int = 8,
    exclude: Callable[[Measurement], str | None] | None = None,
) -> CalibrationReport:
    """Compare ``predict(measurement)`` (tokens/s, or ``None`` if unsupported) with measurements.

    Repetitions of one benchmark (same signature) form **one** data point, their median: eight
    runs of one workload are one benchmark, not eight.  ``exclude(m)`` returns a reason to leave a
    measurement out (for example: produced by a benchmark executable or weights that have since
    changed); excluded measurements are listed, never counted.  Only the validation split decides:

    * ``not_evaluated`` -- fewer than ``min_validation`` distinct validation benchmarks, or no
      usable prediction for at least one of them;
    * ``fail`` -- the median or the 95th-percentile relative error exceeds its threshold;
    * ``pass`` -- otherwise.
    """
    groups: dict[tuple, list[Measurement]] = {}
    excluded: list[dict[str, Any]] = []
    for m in measurements:
        reason = exclude(m) if exclude is not None else None
        if reason:
            excluded.append({"id": m.id, "machine": m.machine, "split": m.split, "reason": reason})
            continue
        groups.setdefault(m.signature, []).append(m)
    rows: list[dict[str, Any]] = []
    pairs: dict[str, list[tuple[float, float]]] = {"calibration": [], "validation": []}
    missing: list[str] = []
    for members in groups.values():
        m = members[0]
        measured = float(np.median([x.tokens_per_s for x in members]))
        predicted = predict(m)
        ok = _usable(predicted)
        rel = abs(predicted - measured) / measured if ok else None
        rows.append({"id": m.id, "machine": m.machine, "model": m.model, "kind": m.kind, "split": m.split,
                     "repetitions": len(members), "measured": measured,
                     "predicted": float(predicted) if ok else None, "rel_error": rel})  # fmt: skip
        if ok:
            pairs[m.split].append((float(predicted), measured))
        elif m.split == "validation":
            missing.append(m.id)

    calibration = error_stats(pairs["calibration"])
    validation = error_stats(pairs["validation"])
    n_validation = sum(1 for members in groups.values() if members[0].split == "validation")
    thresholds = {"median_max": median_max, "p95_max": p95_max,
                  "min_validation": float(min_validation)}  # fmt: skip
    reasons: list[str] = []
    if n_validation < min_validation:
        status = "not_evaluated"
        reasons.append(f"only {n_validation} distinct validation benchmark(s); at least "
                       f"{min_validation} are required")  # fmt: skip
    if missing:
        status = "not_evaluated"
        reasons.append(f"no usable prediction for validation measurement(s): {sorted(missing)}")
    if not reasons:
        if validation.median > median_max:
            reasons.append(f"validation median error {validation.median:.1%} exceeds "
                           f"{median_max:.0%}")  # fmt: skip
        if validation.p95 > p95_max:
            reasons.append(f"validation p95 error {validation.p95:.1%} exceeds {p95_max:.0%}")
        status = "fail" if reasons else "pass"
    notes: list[str] = []
    if calibration.n:
        notes.append(f"in-sample (calibration split) median error {calibration.median:.1%} over "
                     f"{calibration.n} benchmark(s): informational, does not affect the status")  # fmt: skip
    repeated = sum(len(g) - 1 for g in groups.values())
    if repeated:
        notes.append(f"{repeated} repeated run(s) were merged into their benchmark (median)")
    if excluded:
        notes.append(f"{len(excluded)} measurement(s) excluded: {sorted({e['reason'] for e in excluded})}")
    return CalibrationReport(calibration, validation, rows, thresholds, status, reasons, notes, excluded)


def evaluate_by_machine(predict: Callable[[Measurement], float | None], measurements: MeasurementSet,
                        **kwargs: Any) -> CalibrationReport:  # fmt: skip
    """Evaluate every machine on its own; the combined status passes only if every machine with
    measurements passes, and fails if any fails.  Machines are never pooled: a good fit on one
    cannot hide a bad fit on another."""
    per_machine: dict[str, CalibrationReport] = {}
    for machine in sorted({m.machine for m in measurements}):
        subset = MeasurementSet([m for m in measurements if m.machine == machine])
        per_machine[machine] = evaluate_calibration(predict, subset, **kwargs)
    combined = evaluate_calibration(predict, measurements, **kwargs)
    statuses = [r.status for r in per_machine.values()]
    if not statuses:
        combined.status = "not_evaluated"
    elif "fail" in statuses:
        combined.status = "fail"
    elif all(st == "pass" for st in statuses):
        combined.status = "pass"
    else:
        combined.status = "not_evaluated"
    combined.reasons = [f"{machine}: {reason}" for machine, r in per_machine.items() for reason in r.reasons]
    combined.by_machine = per_machine
    return combined
