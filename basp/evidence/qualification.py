"""Qualifying one metric for one candidate in one regime from receipts.

The rules:

* only receipts for this metric, candidate and regime are considered -- a throughput
  receipt never qualifies energy, and one architecture's receipt never qualifies another;
* a receipt with an invalid value or missing bindings is rejected;
* a receipt bound to an identity that has since changed (simulator source, profile, model
  weights, benchmark executable, estimates, ...) is stale and rejected;
* receipts that record the same observation (identical in everything but their free-text note)
  count once;
* of the remaining receipts those at the highest level count, **all of them**: the verdict
  uses the worst value among them (lowest for a ratio, highest for an error).  Nothing about a
  receipt's hash, its note, or the order receipts were filed in can change the verdict, and an
  applicable failing measurement cannot be outvoted or disappear;
* a value that misses the threshold fails at any level; one that meets it passes only at the
  required level or above, and is "not evaluated" otherwise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .receipts import Receipt
from .registry import ReceiptRegistry
from .schema import METRICS, EvidenceLevel


@dataclass
class Qualification:
    metric: str
    status: str  # pass | fail | not_evaluated
    required: EvidenceLevel
    level: EvidenceLevel | None = None
    value: float | None = None  # the worst value among the counted receipts
    receipt_id: str | None = None  # the receipt that carries that value (lowest id among equals)
    receipt_ids: list[str] = field(default_factory=list)  # every counted receipt
    values: list[float] = field(default_factory=list)  # their values, worst first
    conflict: bool = False  # counted receipts disagree about meeting the threshold
    reasons: list[str] = field(default_factory=list)  # why receipts were rejected, or why not evaluated

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def stale_bindings(receipt: Receipt, current: Mapping[str, str]) -> list[str]:
    """Bindings of ``receipt`` that differ from the current identities."""
    return [k for k, v in current.items() if k in receipt.bindings and receipt.bindings[k] != v]


def qualify(metric: str, registry: ReceiptRegistry, *, candidate: str, regime: str,
            meets: Callable[[float], bool], required: EvidenceLevel,
            current: Mapping[str, str] | None = None,
            must_bind: Mapping[str, str] | None = None) -> Qualification:  # fmt: skip
    """``current``: identities a stale receipt would differ in.  ``must_bind``: identities a
    receipt has to carry with exactly this value (e.g. the model the claim is about)."""
    result = Qualification(metric, "not_evaluated", required)
    usable = []
    seen: set[str] = set()
    # a fixed order that depends on what was observed, not on notes or on filing order
    found = sorted(registry.find(metric, candidate=candidate, regime=regime),
                   key=lambda r: (r.measurement_id, r.receipt_id))  # fmt: skip
    for receipt in found:
        if receipt.measurement_id in seen:
            continue
        seen.add(receipt.measurement_id)
        problems = receipt.problems()
        problems += [f"stale: {k} changed" for k in stale_bindings(receipt, current or {})]
        for key, value in (must_bind or {}).items():
            if receipt.bindings.get(key) != value:
                problems.append(f"about {key} {receipt.bindings.get(key)!r}, not {value!r}")
        if problems:
            result.reasons.append(f"receipt {receipt.receipt_id} rejected: " + "; ".join(problems))
        else:
            usable.append(receipt)
    if not usable:
        if not result.reasons:
            result.reasons.append(f"no {metric} receipt for {candidate} in this regime")
        return result
    top = max(r.level for r in usable)
    counted: dict[str, Receipt] = {}
    for receipt in sorted((r for r in usable if r.level == top), key=lambda r: r.receipt_id):
        counted.setdefault(receipt.measurement_id, receipt)  # one observation, however annotated
    spec = METRICS[metric]
    worst = spec.worst(r.value for r in counted.values())
    ordered = sorted(counted.values(), key=lambda r: (r.value != worst, r.receipt_id))
    result.level, result.value, result.receipt_id = top, worst, ordered[0].receipt_id
    result.receipt_ids = sorted(r.receipt_id for r in counted.values())
    result.values = sorted((float(r.value) for r in counted.values()), reverse=spec.higher_is_better)[::-1]
    verdicts = {bool(meets(float(r.value))) for r in counted.values()}
    result.conflict = len(verdicts) > 1
    if result.conflict:
        result.reasons.append(f"{len(counted)} {top.name} receipts disagree (values {min(result.values):g} to "
                              f"{max(result.values):g}); the worst counts")  # fmt: skip
    if not meets(worst):
        result.status = "fail"
    elif top >= required:
        result.status = "pass"
    else:
        result.reasons.append(f"best evidence is {top.name}; {required.name} required")
    return result
