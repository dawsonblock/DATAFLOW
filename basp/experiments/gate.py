"""The evidence-gated hardware development decision.

Hardware development starts only if every criterion is met, for **one candidate
architecture in one declared workload regime**, at the pessimistic corner of every
estimate, and **on evidence of at least the required level** (``basp.evidence``):

* a criterion whose evidence is below its required level cannot pass -- it is "not
  evaluated" -- but it can fail when even the optimistic reading misses the threshold;
* a criterion that cannot be evaluated counts against the decision;
* evidence about one architecture or regime never stands in for another: a scaling result
  measured for architecture E does not qualify architecture H.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from basp.calibration import CalibrationReport
from basp.evidence import EvidenceLevel

E0, E1, E2, E3 = EvidenceLevel.E0, EvidenceLevel.E1, EvidenceLevel.E2, EvidenceLevel.E3


@dataclass
class Criterion:
    name: str
    threshold: str
    result: str
    status: str  # pass | fail | not_evaluated
    level: EvidenceLevel | None  # evidence actually available (None: nothing)
    required: EvidenceLevel  # minimum evidence for a pass
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def evidence(self) -> str:
        if self.level is None:
            return "none"
        return f"{self.level.name} {self.level.description}" + (f"; {self.note}" if self.note else "")

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "threshold": self.threshold, "result": self.result, "status": self.status,
                "evidence": self.evidence, "level": None if self.level is None else self.level.name,
                "required": self.required.name}  # fmt: skip


@dataclass
class Candidate:
    """Everything the gate needs about one candidate architecture in one regime.

    Ratios compare the candidate with the conventional baseline at the decision corner.
    ``scaling_arch``/``scaling_regime`` say what the scaling efficiency was measured for; the
    gate accepts it only if they are the candidate's own.
    """

    arch: str
    name: str
    regime: str
    throughput_ratio: float | None
    energy_ratio: float | None
    cost_efficiency_ratio: float | None
    scaling_efficiency: float | None = None
    scaling_arch: str | None = None
    scaling_regime: str | None = None
    corner: str = "pessimistic"

    @property
    def label(self) -> str:
        return f"{self.arch} {self.name}"


@dataclass
class GateDecision:
    regime: str
    candidate: str
    criteria: list[Criterion] = field(default_factory=list)

    @property
    def go(self) -> bool:
        return bool(self.criteria) and all(c.passed for c in self.criteria)

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
        return {"regime": self.regime, "candidate": self.candidate, "verdict": self.verdict, "go": self.go,
                "criteria": [c.to_json() for c in self.criteria]}  # fmt: skip


def _status(meets: bool | None, level: EvidenceLevel | None, required: EvidenceLevel) -> str:
    """Pass only on sufficient evidence; insufficient evidence can still show a failure."""
    if meets is None or level is None:
        return "not_evaluated"
    if not meets:
        return "fail"
    return "pass" if level >= required else "not_evaluated"


def _ratio(name: str, value: float | None, minimum: float, what: str, level: EvidenceLevel) -> Criterion:
    if value is None or not math.isfinite(value):
        return Criterion(name, f">= {minimum:g}x", "no candidate could be evaluated", "not_evaluated", None, E2)
    status = _status(value >= minimum, level, E2)
    return Criterion(name, f">= {minimum:g}x", f"{value:.2f}x {what}", status, level, E2)


def evaluate_gate(
    candidate: Candidate,
    thresholds: Mapping[str, Any],
    *,
    calibration: CalibrationReport,
    comparison_on_calibrated_profile: bool,
    plans_per_batch: float | None,
    batch_time_s: float | None,
    numeric_rel_error: float | None,
    real_weights_verified: bool = False,
    model_quality_measured: bool = False,
) -> GateDecision:
    """Evaluate every criterion for one candidate in one regime.

    Simulated evidence is E2 only when the calibration criterion passes *and* the comparison
    ran on the profile that was validated; a validated profile of one machine says nothing
    about results simulated on another.
    """
    calibrated = calibration.status == "pass" and comparison_on_calibrated_profile
    sim = E2 if calibrated else E1
    decision = GateDecision(candidate.regime, candidate.label)
    add = decision.criteria.append

    # -- 1. the simulator's own accuracy, on held-out measurements --------------------------
    v = calibration.validation
    if calibration.status == "not_evaluated":
        result = f"{v.n} validation measurements (at least {thresholds['min_validation_measurements']} needed)"
    else:
        result = f"median {v.median:.1%}, p95 {v.p95:.1%} over {v.n} held-out measurements"
    status = calibration.status
    if status == "pass" and not comparison_on_calibrated_profile:
        status = "not_evaluated"
        result += "; but the comparison below did not run on the validated profile"
    add(Criterion("Simulator calibration error",
                  f"<= {thresholds['calibration_median_max']:.0%} median, <= {thresholds['calibration_p95_max']:.0%} p95",
                  result, status, E3 if v.n else None, E3))  # fmt: skip

    # -- 2-4. the candidate against the conventional baseline --------------------------------
    versus = f"the conventional baseline ({candidate.label}, {candidate.corner} estimates)"
    add(_ratio("Throughput improvement", candidate.throughput_ratio, thresholds["throughput_min"], versus, sim))
    add(_ratio("Energy efficiency", candidate.energy_ratio, thresholds["energy_min"], versus, sim))
    add(_ratio("Cost efficiency", candidate.cost_efficiency_ratio, thresholds["cost_efficiency_min"], versus, sim))

    # -- 5. scaling, of this candidate in this regime -------------------------------------------
    minimum = thresholds["scaling_efficiency_min"]
    name, threshold = "Four-module scaling efficiency", f">= {minimum:.0%}"
    if candidate.scaling_efficiency is None or not math.isfinite(candidate.scaling_efficiency):
        add(Criterion(name, threshold, "not measured", "not_evaluated", None, E2))
    elif candidate.scaling_arch != candidate.arch or candidate.scaling_regime != candidate.regime:
        add(Criterion(name, threshold,
                      f"{candidate.scaling_efficiency:.0%} was measured for {candidate.scaling_arch} in "
                      f"{candidate.scaling_regime!r}, not for the candidate in this regime",
                      "not_evaluated", None, E2))  # fmt: skip
    else:
        ok = candidate.scaling_efficiency >= minimum
        add(Criterion(name, threshold, f"{candidate.scaling_efficiency:.0%} from one to four modules",
                      _status(ok, sim, E2), sim, E2))  # fmt: skip

    # -- 6. scheduler overhead: only a production scheduler can be measured --------------------
    limit = thresholds["scheduler_overhead_max"]
    costs = thresholds["plan_evaluation_cost_s"]
    if plans_per_batch is None or not batch_time_s:
        add(Criterion("Scheduler overhead", f"<= {limit:.0%}", "not measured", "not_evaluated", None, E3))
    else:
        low = plans_per_batch * costs["optimistic"] / batch_time_s
        high = plans_per_batch * costs["pessimistic"] / batch_time_s
        budget = limit * batch_time_s / plans_per_batch
        result = (f"{low:.1%} to {high:.1%} for {plans_per_batch:.0f} plan evaluations per batch at an assumed "
                  f"{costs['optimistic'] * 1e9:.0f}-{costs['pessimistic'] * 1e9:.0f} ns each; needs "
                  f"<= {budget * 1e9:.0f} ns per evaluation")  # fmt: skip
        meets = False if low > limit else (True if high <= limit else None)
        status = "fail" if meets is False else "not_evaluated"  # an estimate can fail, never pass
        add(Criterion("Scheduler overhead", f"<= {limit:.0%}", result, status, E0, E3,
                      "no production scheduler exists"))  # fmt: skip

    # -- 7. numerical correctness on the real model ----------------------------------------------
    tolerance = thresholds["numeric_rel_l2_max"]
    name, threshold = "Numerical correctness", f"relative error <= {tolerance:g}"
    if numeric_rel_error is None:
        add(Criterion(name, threshold, "not measured", "not_evaluated", None, E3))
    elif numeric_rel_error > tolerance:
        add(Criterion(name, threshold, f"{numeric_rel_error:.1e} on the small numeric model", "fail", E3, E3))
    else:
        scope = "the real model" if real_weights_verified else "a small random-weight model; real weights not run"
        add(Criterion(name, threshold, f"{numeric_rel_error:.1e} on {scope}",
                      "pass" if real_weights_verified else "not_evaluated", E3, E3,
                      "" if real_weights_verified else "wrong model"))  # fmt: skip

    # -- 8. model quality: an argument is not a measurement -----------------------------------
    argument = ("the runtime never selects experts and exchanges only lossless-equivalent weight formats; "
                "dataflows differ only in FP32 summation order")  # fmt: skip
    if model_quality_measured:
        add(Criterion("Model quality", "no material degradation", "measured parity", "pass", E3, E3))
    else:
        add(Criterion("Model quality", "no material degradation", f"not measured ({argument})",
                      "not_evaluated", E0, E3, "by construction"))  # fmt: skip
    return decision
