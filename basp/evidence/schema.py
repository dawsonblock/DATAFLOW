"""Evidence levels and the metrics evidence is about.

=====  ==========================================================================
E0     theoretical estimate, argument by construction, or a guessed parameter
E1     simulation that has not been validated against measurements
E2     simulation validated within a measured hardware scope
E3     direct measurement of the claimed capability
=====  ==========================================================================

A claim about hypothetical hardware can never exceed E2 before a prototype exists:
calibrating the simulator on a GPU validates the model of that GPU, not of a module.

Each metric is evidenced separately.  A throughput measurement qualifies throughput; it says
nothing about energy, cost, numerical correctness or model quality.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class EvidenceLevel(enum.IntEnum):
    E0 = 0
    E1 = 1
    E2 = 2
    E3 = 3

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]

    def __str__(self) -> str:
        return f"{self.name} ({self.description})"


_DESCRIPTIONS = {
    EvidenceLevel.E0: "estimate or argument",
    EvidenceLevel.E1: "uncalibrated simulation",
    EvidenceLevel.E2: "simulation validated on existing hardware",
    EvidenceLevel.E3: "measurement",
}


@dataclass(frozen=True)
class Metric:
    """A quantity evidence can be about, with what makes a value of it meaningful."""

    name: str
    unit: str
    minimum: float | None = 0.0  # values below are invalid (None: unbounded)
    maximum: float | None = None
    description: str = ""
    #: which way is better.  Several applicable receipts are combined conservatively: the
    #: verdict uses the worst of them, so a failing measurement can never be outvoted or dropped.
    higher_is_better: bool = True

    def worst(self, values) -> float:
        values = [float(v) for v in values]
        return min(values) if self.higher_is_better else max(values)


METRICS: dict[str, Metric] = {m.name: m for m in (
    Metric("throughput_ratio", "x", 0.0, description="candidate tokens/s over the baseline's, same workload"),
    Metric("energy_ratio", "x", 0.0, description="candidate tokens per joule over the baseline's"),
    Metric("cost_efficiency_ratio", "x", 0.0, description="candidate tokens/s per dollar over the baseline's"),
    Metric("scaling_efficiency", "fraction", 0.0, 2.0, description="speed-up from one to N modules, divided by N"),
    Metric("calibration_error", "fraction", 0.0, description="relative error of predictions on held-out measurements",
           higher_is_better=False),
    Metric("scheduler_overhead", "fraction", 0.0, description="scheduling time over execution time",
           higher_is_better=False),
    Metric("numerical_error", "relative L2", 0.0, description="relative L2 error against an independent reference",
           higher_is_better=False),
    Metric("model_quality", "relative perplexity", 0.0, description="perplexity relative to the reference runtime",
           higher_is_better=False),
    Metric("request_speedup", "x", 0.0, description="baseline request latency over the candidate's, same requests"),
    Metric("tokens_per_s", "tokens/s", 0.0, description="measured or simulated throughput"),
    Metric("weight_traffic_reduction", "fraction", None, 1.0, description="weight bytes moved, relative reduction"),
)}  # fmt: skip
