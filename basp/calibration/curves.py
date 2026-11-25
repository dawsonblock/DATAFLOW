"""Hardware performance curves: measured throughput as a function of one size parameter.

A curve records how one device (or link) performs as a size grows -- bytes per transfer,
batch rows, context length -- with the units of both axes, the measurement each point came
from, and the machine.  ``validate_curve`` rejects malformed curves (unknown units, unsorted or
duplicate sizes, non-finite or non-positive values) so a profile is never fitted to garbage.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any

X_UNITS = {"bytes", "rows", "tokens", "elements"}
Y_UNITS = {"bytes/s", "flop/s", "mac/s", "tokens/s", "s"}
KINDS = {"memory_bandwidth", "link_transfer", "compute_roofline", "kernel_launch", "decode", "prefill"}


@dataclass(frozen=True)
class PerformanceCurve:
    machine: str
    device: str
    kind: str
    x_unit: str
    y_unit: str
    points: tuple[tuple[float, float], ...]
    sources: tuple[str, ...] = ()  # measurement ids or tool output the points came from
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["points"] = [list(p) for p in self.points]
        data["sources"] = list(self.sources)
        data["notes"] = dict(self.notes)
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PerformanceCurve:
        return cls(data["machine"], data["device"], data["kind"], data["x_unit"], data["y_unit"],
                   tuple((float(x), float(y)) for x, y in data["points"]), tuple(data.get("sources", ())),
                   dict(data.get("notes", {})))  # fmt: skip

    def interpolate(self, x: float) -> float:
        """Linear interpolation in log(x); clamped at the ends (no extrapolation)."""
        problems = validate_curve(self)
        if problems:
            raise ValueError("invalid curve: " + "; ".join(problems))
        xs = [p[0] for p in self.points]
        if x <= xs[0]:
            return self.points[0][1]
        if x >= xs[-1]:
            return self.points[-1][1]
        for (x0, y0), (x1, y1) in pairwise(self.points):
            if x0 <= x <= x1:
                t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
                return y0 + t * (y1 - y0)
        raise AssertionError("unreachable")


def validate_curve(curve: PerformanceCurve) -> list[str]:
    problems = []
    if not curve.machine or not curve.device:
        problems.append("a curve names its machine and device")
    if curve.kind not in KINDS:
        problems.append(f"unknown kind {curve.kind!r}")
    if curve.x_unit not in X_UNITS or curve.y_unit not in Y_UNITS:
        problems.append(f"unknown units {curve.x_unit!r} -> {curve.y_unit!r}")
    if len(curve.points) < 2:
        problems.append("a curve needs at least two points")
    for x, y in curve.points:
        if not (math.isfinite(x) and math.isfinite(y)) or x <= 0 or y <= 0:
            problems.append(f"point ({x}, {y}) is not finite and positive")
            break
    xs = [p[0] for p in curve.points]
    if any(b <= a for a, b in pairwise(xs)):
        problems.append("sizes must be strictly increasing")
    return problems
