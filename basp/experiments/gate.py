"""The evidence-gated hardware development decision.

Hardware development starts only if every criterion is met, for **one candidate
architecture in one declared workload regime**, at the pessimistic corner of every
estimate, and **on evidence of at least the required level** (``basp.evidence``):

* a criterion whose evidence is below its required level cannot pass -- it is "not
  evaluated" -- but it can fail when even the optimistic reading misses the threshold;
* a criterion that cannot be evaluated counts against the decision;
* evidence about one architecture or regime never stands in for another: a scaling result
  measured for architecture E does not qualify architecture H;
* each criterion is decided by **metric-specific receipts** (``basp.evidence``): a receipt
  binds one value to the identities that produced it, and a receipt whose identities have
  changed since (simulator source, profile, estimates, model weights, validation) is stale.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from basp.calibration import CalibrationReport
from basp.evidence import EvidenceLevel, Qualification, Receipt, ReceiptRegistry, qualify

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
    """One candidate architecture in one declared workload regime.

    ``model`` names the model the regime is about: evidence measured on any other model (a
    small test model, say) cannot qualify the candidate.
    """

    arch: str
    name: str
    regime: str
    model: str = ""
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


def simulation_receipts(candidate: Candidate, *, level: EvidenceLevel, bindings: Mapping[str, str],
                        values: Mapping[str, float], raw: Mapping[str, Any] | None = None,
                        baseline: str = "A") -> list[Receipt]:  # fmt: skip
    """Receipts for simulated metrics of ``candidate`` (one per metric)."""
    return [Receipt(metric, float(value), level, candidate.arch, candidate.regime, baseline, dict(bindings),
                    raw=dict(raw or {}), note=f"{candidate.corner} corner") for metric, value in values.items()]  # fmt: skip


def scheduler_receipt(candidate: Candidate, *, plans_per_batch: float, batch_time_s: float,
                      costs: Mapping[str, float], limit: float) -> Receipt:  # fmt: skip
    """An E0 estimate of scheduler overhead: plan evaluations per batch times an *assumed* cost
    per evaluation, relative to the batch time.  No production scheduler exists to measure, so
    this can show a failure (even the optimistic cost is too slow) but never a pass."""
    if batch_time_s <= 0 or plans_per_batch < 0:
        raise ValueError("batch time must be positive and plan evaluations non-negative")
    low = plans_per_batch * costs["optimistic"] / batch_time_s
    high = plans_per_batch * costs["pessimistic"] / batch_time_s
    raw = {"high": high, "plans_per_batch": float(plans_per_batch), "cost_low_ns": costs["optimistic"] * 1e9,
           "cost_high_ns": costs["pessimistic"] * 1e9,
           # with no plan evaluations at all there is no per-evaluation budget to state
           "budget_ns": limit * batch_time_s / plans_per_batch * 1e9 if plans_per_batch > 0 else None}  # fmt: skip
    return Receipt("scheduler_overhead", low, E0, candidate.arch, candidate.regime, raw=raw,
                   note="estimate: assumed cost per plan evaluation; no production scheduler exists")  # fmt: skip


def _criterion(name: str, threshold: str, q: Qualification, describe) -> Criterion:
    if q.value is None:
        result = "no usable evidence: " + (q.reasons[0] if q.reasons else "none recorded")
    else:
        result = describe(q.value)
        if q.status == "not_evaluated" and q.reasons:
            result += f" ({q.reasons[-1]})"
    return Criterion(name, threshold, result, q.status, q.level, q.required)


def evaluate_gate(
    candidate: Candidate,
    thresholds: Mapping[str, Any],
    *,
    receipts: ReceiptRegistry,
    calibration: CalibrationReport,
    comparison_on_calibrated_profile: bool,
    current: Mapping[str, str] | None = None,
) -> GateDecision:
    """Evaluate every criterion for one candidate in one regime, from receipts.

    ``current`` holds the identities receipts must still match (simulator source, profile,
    estimates, validation).  Simulated evidence counts as E2 only through receipts bound to a
    passing validation of the profile the comparison ran on.
    """
    current = dict(current or {})
    decision = GateDecision(candidate.regime, candidate.label)
    add = decision.criteria.append

    def q(metric, meets, required, must_bind=None):
        return qualify(metric, receipts, candidate=candidate.arch, regime=candidate.regime, meets=meets,
                       required=required, current=current, must_bind=must_bind)  # fmt: skip

    # -- 1. the simulator's own accuracy, on held-out measurements --------------------------
    v = calibration.validation
    if calibration.status == "not_evaluated":
        result = f"{v.n} validation benchmarks (at least {thresholds['min_validation_measurements']} needed)"
    else:
        result = f"median {v.median:.1%}, p95 {v.p95:.1%} over {v.n} held-out benchmarks"
    status = calibration.status
    if status == "pass" and not comparison_on_calibrated_profile:
        status = "not_evaluated"
        result += "; but the comparison below did not run on the validated profile"
    add(Criterion("Simulator calibration error",
                  f"<= {thresholds['calibration_median_max']:.0%} median, <= {thresholds['calibration_p95_max']:.0%} p95",
                  result, status, E3 if v.n else None, E3))  # fmt: skip

    # -- 2-5. the candidate against the conventional baseline, and its own scaling ------------
    versus = f"the conventional baseline ({candidate.label}, {candidate.corner} estimates)"
    for metric, name, key in (("throughput_ratio", "Throughput improvement", "throughput_min"),
                              ("energy_ratio", "Energy efficiency", "energy_min"),
                              ("cost_efficiency_ratio", "Cost efficiency", "cost_efficiency_min")):  # fmt: skip
        minimum = thresholds[key]
        add(_criterion(name, f">= {minimum:g}x", q(metric, lambda x, m=minimum: x >= m, E2),
                       lambda x: f"{x:.2f}x {versus}"))  # fmt: skip
    minimum = thresholds["scaling_efficiency_min"]
    add(_criterion("Four-module scaling efficiency", f">= {minimum:.0%}",
                   q("scaling_efficiency", lambda x: x >= minimum, E2),
                   lambda x: f"{x:.0%} from one to four modules"))  # fmt: skip

    # -- 6. scheduler overhead: only a production scheduler can be measured --------------------
    limit = thresholds["scheduler_overhead_max"]
    sched = q("scheduler_overhead", lambda x: x <= limit, E3)
    best = next((r for r in receipts.find("scheduler_overhead", candidate=candidate.arch, regime=candidate.regime)
                 if r.receipt_id == sched.receipt_id), None)  # fmt: skip

    def describe_overhead(x: float) -> str:
        raw = best.raw if best is not None else {}
        if "high" not in raw:
            return f"{x:.1%}"
        text = (f"{x:.1%} to {raw['high']:.1%} for {raw['plans_per_batch']:.0f} plan evaluations per batch at an "
                f"assumed {raw['cost_low_ns']:.0f}-{raw['cost_high_ns']:.0f} ns each")  # fmt: skip
        if raw.get("budget_ns") is None:
            return text + "; no plan evaluations, so no per-evaluation budget"
        return text + f"; needs <= {raw['budget_ns']:.0f} ns per evaluation"

    add(_criterion("Scheduler overhead", f"<= {limit:.0%}", sched, describe_overhead))

    # -- 7-8. numerical correctness and model quality, on the model the regime is about --------
    tolerance = thresholds["numeric_rel_l2_max"]
    bind = {"model": candidate.model} if candidate.model else None
    add(_criterion("Numerical correctness", f"relative error <= {tolerance:g}",
                   q("numerical_error", lambda x: x <= tolerance, E3, bind), lambda x: f"{x:.1e}"))  # fmt: skip
    quality_max = thresholds["model_quality_relative_perplexity_max"]
    quality = q("model_quality", lambda x: x <= quality_max, E3, bind)
    if not receipts.find("model_quality", candidate=candidate.arch, regime=candidate.regime):
        quality.reasons.insert(0, "not measured (the runtime never selects experts and exchanges only "
                                  "lossless-equivalent weight formats; that is an argument, not a measurement)")  # fmt: skip
    add(_criterion("Model quality", f"perplexity <= {quality_max:g}x the reference",
                   quality, lambda x: f"{x:.3f}x the reference perplexity"))  # fmt: skip
    return decision
