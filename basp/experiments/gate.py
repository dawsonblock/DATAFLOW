"""The evidence-gated hardware development decision.

Hardware development starts only if every criterion is met **on calibrated
evidence**, in a declared workload regime, at the pessimistic corner of every
estimate.  A criterion that cannot be evaluated counts against the decision:
absence of evidence is not a pass.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from basp.calibration import CalibrationReport


@dataclass
class Criterion:
    name: str
    threshold: str
    result: str
    status: str  # pass | fail | not_evaluated
    evidence: str  # measured | simulated, calibrated | simulated, uncalibrated | by construction | none

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass
class GateDecision:
    regime: str
    criteria: list[Criterion] = field(default_factory=list)

    @property
    def go(self) -> bool:
        return all(c.passed for c in self.criteria)

    @property
    def blockers(self) -> list[str]:
        return [f"{c.name}: {c.result}" for c in self.criteria if not c.passed]

    @property
    def verdict(self) -> str:
        if self.go:
            return "GO"
        failed = sum(1 for c in self.criteria if c.status == "fail")
        missing = sum(1 for c in self.criteria if c.status == "not_evaluated")
        noun = "criterion" if failed == 1 else "criteria"
        return f"NO-GO ({failed} {noun} failed, {missing} not evaluated)"

    def to_json(self) -> dict[str, Any]:
        return {"regime": self.regime, "verdict": self.verdict, "go": self.go,
                "criteria": [vars(c) for c in self.criteria]}  # fmt: skip


def _ratio(name: str, value: float | None, minimum: float, what: str, evidence: str) -> Criterion:
    if value is None or not math.isfinite(value):
        return Criterion(name, f">= {minimum:g}x", "no candidate could be evaluated", "not_evaluated", "none")
    status = "pass" if value >= minimum else "fail"
    return Criterion(name, f">= {minimum:g}x", f"{value:.2f}x {what}", status, evidence)


def evaluate_gate(
    regime: str,
    thresholds: Mapping[str, Any],
    *,
    calibration: CalibrationReport,
    throughput_ratio: float | None,
    energy_ratio: float | None,
    cost_efficiency_ratio: float | None,
    scaling_efficiency: float | None,
    plans_per_batch: float | None,
    batch_time_s: float | None,
    numeric_rel_error: float | None,
    real_weights_verified: bool = False,
    candidate: str = "best module architecture",
    scaling_of: str | None = None,
    comparison_on_calibrated_profile: bool = True,
) -> GateDecision:
    """Evaluate every criterion for one workload regime.

    The three ratios compare the candidate with the conventional baseline at the
    decision corner of the estimates.  Simulated evidence is labelled calibrated
    only when the calibration criterion itself passes *and* the comparison ran on
    the profile that was validated (``comparison_on_calibrated_profile``): a
    validated profile of one machine says nothing about results simulated on another.
    """
    calibrated = calibration.status == "pass" and comparison_on_calibrated_profile
    simulated = "simulated, calibrated" if calibrated else "simulated, uncalibrated"
    corner = thresholds.get("decision_corner", "pessimistic")
    decision = GateDecision(regime)
    add = decision.criteria.append

    v = calibration.validation
    if calibration.status == "not_evaluated":
        result = f"{v.n} validation measurements (at least {thresholds['min_validation_measurements']} needed)"
    else:
        result = f"median {v.median:.1%}, p95 {v.p95:.1%} over {v.n} held-out measurements"
    status = calibration.status
    if status == "pass" and not comparison_on_calibrated_profile:
        status = "not_evaluated"
        result += "; but the comparison below did not run on the validated profile"
    add(Criterion(
        "Simulator calibration error",
        f"<= {thresholds['calibration_median_max']:.0%} median, <= {thresholds['calibration_p95_max']:.0%} p95",
        result, status, "measured" if v.n else "none",
    ))  # fmt: skip
    versus = f"the conventional baseline ({candidate}, {corner} estimates)"
    add(_ratio("Throughput improvement", throughput_ratio, thresholds["throughput_min"], versus, simulated))
    add(_ratio("Energy efficiency", energy_ratio, thresholds["energy_min"], versus, simulated))
    add(_ratio("Cost efficiency", cost_efficiency_ratio, thresholds["cost_efficiency_min"], versus,
               simulated))  # fmt: skip
    if scaling_efficiency is None:
        add(Criterion("Four-module scaling efficiency", f">= {thresholds['scaling_efficiency_min']:.0%}",
                      "not measured", "not_evaluated", "none"))  # fmt: skip
    else:
        ok = scaling_efficiency >= thresholds["scaling_efficiency_min"]
        add(Criterion("Four-module scaling efficiency", f">= {thresholds['scaling_efficiency_min']:.0%}",
                      f"{scaling_efficiency:.0%} from one to four modules ({scaling_of or candidate})",
                      "pass" if ok else "fail", simulated))  # fmt: skip

    limit = thresholds["scheduler_overhead_max"]
    costs = thresholds["plan_evaluation_cost_s"]
    if plans_per_batch is None or not batch_time_s:
        add(Criterion("Scheduler overhead", f"<= {limit:.0%}", "not measured", "not_evaluated", "none"))
    else:
        low = plans_per_batch * costs["optimistic"] / batch_time_s
        high = plans_per_batch * costs["pessimistic"] / batch_time_s
        budget = limit * batch_time_s / plans_per_batch
        result = (f"{low:.1%} to {high:.1%} for {plans_per_batch:.0f} plan evaluations per batch at an assumed "
                  f"{costs['optimistic'] * 1e9:.0f}-{costs['pessimistic'] * 1e9:.0f} ns each; needs "
                  f"<= {budget * 1e9:.0f} ns per evaluation")  # fmt: skip
        status = "pass" if high <= limit else "fail" if low > limit else "not_evaluated"
        add(Criterion("Scheduler overhead", f"<= {limit:.0%}", result, status,
                      "estimated; no production scheduler exists"))  # fmt: skip

    tolerance = thresholds["numeric_rel_l2_max"]
    if numeric_rel_error is None:
        add(Criterion("Numerical correctness", f"relative error <= {tolerance:g}", "not measured",
                      "not_evaluated", "none"))  # fmt: skip
    elif numeric_rel_error > tolerance:
        add(Criterion("Numerical correctness", f"relative error <= {tolerance:g}",
                      f"{numeric_rel_error:.1e} on the small numeric model", "fail", "measured"))  # fmt: skip
    else:
        add(Criterion(
            "Numerical correctness", f"relative error <= {tolerance:g}",
            f"{numeric_rel_error:.1e} on a small random-weight model"
            + ("" if real_weights_verified else "; real model weights not verified"),
            "pass" if real_weights_verified else "not_evaluated", "measured",
        ))  # fmt: skip
    add(Criterion("Model quality", "no material degradation",
                  "the runtime never selects experts and exchanges only lossless-equivalent weight "
                  "formats; dataflows differ only in FP32 summation order (rounding level)",
                  "pass", "by construction"))  # fmt: skip
    return decision
