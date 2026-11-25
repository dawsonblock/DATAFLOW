"""Qualifying one metric for one candidate in one regime from receipts.

The rules:

* only receipts for this metric, candidate and regime are considered -- a throughput
  receipt never qualifies energy, and one architecture's receipt never qualifies another;
* a receipt with an invalid value or missing bindings is rejected;
* a receipt bound to an identity that has since changed (simulator source, profile, model
  weights, benchmark executable, estimates, ...) is stale and rejected;
* of the remaining receipts the highest level counts;
* a value that misses the threshold fails at any level; one that meets it passes only at the
  required level or above, and is "not evaluated" otherwise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .receipts import Receipt
from .registry import ReceiptRegistry
from .schema import EvidenceLevel


@dataclass
class Qualification:
    metric: str
    status: str  # pass | fail | not_evaluated
    required: EvidenceLevel
    level: EvidenceLevel | None = None
    value: float | None = None
    receipt_id: str | None = None
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
    for receipt in registry.find(metric, candidate=candidate, regime=regime):
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
    best = max(usable, key=lambda r: (r.level, r.repetitions, r.receipt_id))
    result.level, result.value, result.receipt_id = best.level, float(best.value), best.receipt_id
    if not meets(result.value):
        result.status = "fail"
    elif best.level >= required:
        result.status = "pass"
    else:
        result.reasons.append(f"best evidence is {best.level.name}; {required.name} required")
    return result
