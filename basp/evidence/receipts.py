"""Evidence receipts: one metric, one value, and exactly what produced it.

A receipt binds a value to the identities it depends on -- machine and profile, model
weights, runtime executable and source, benchmark configuration, candidate and baseline,
workload regime -- and to its raw results.  Its id is the hash of that content, so a receipt
cannot be edited without becoming a different receipt.  One receipt evidences one metric only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .identity import sha256_json
from .schema import METRICS, EvidenceLevel
from .validation import value_problems

#: identities a receipt of each level must bind
REQUIRED_BINDINGS: dict[EvidenceLevel, tuple[str, ...]] = {
    EvidenceLevel.E0: (),
    EvidenceLevel.E1: ("source", "config"),
    EvidenceLevel.E2: ("source", "config", "profile", "validation"),
    EvidenceLevel.E3: ("machine", "model", "runtime", "benchmark"),
}


@dataclass(frozen=True)
class Receipt:
    metric: str
    value: float
    level: EvidenceLevel
    candidate: str
    regime: str
    baseline: str | None = None
    bindings: Mapping[str, str] = field(default_factory=dict)
    repetitions: int = 1
    raw: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def unit(self) -> str:
        return METRICS[self.metric].unit if self.metric in METRICS else "?"

    @property
    def receipt_id(self) -> str:
        return sha256_json(self.content())[:16]

    def content(self) -> dict[str, Any]:
        data = asdict(self)
        data["level"] = self.level.name
        data["bindings"] = dict(self.bindings)
        data["raw"] = dict(self.raw)
        return data

    def problems(self) -> list[str]:
        """Why this receipt cannot serve as evidence at its level (empty: it can)."""
        problems = value_problems(self.metric, self.value)
        missing = [b for b in REQUIRED_BINDINGS[self.level] if not self.bindings.get(b)]
        if missing:
            problems.append(f"{self.level.name} receipt lacks binding(s) {missing}")
        if not isinstance(self.repetitions, int) or self.repetitions < 1:
            problems.append(f"repetitions must be a positive integer, got {self.repetitions!r}")
        if not self.candidate or not self.regime:
            problems.append("a receipt names its candidate and its regime")
        return problems

    def to_json(self) -> dict[str, Any]:
        return {**self.content(), "receipt_id": self.receipt_id}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Receipt:
        data = {k: v for k, v in data.items() if k != "receipt_id"}
        data["level"] = EvidenceLevel[data["level"]]
        receipt = cls(**data)
        return receipt
