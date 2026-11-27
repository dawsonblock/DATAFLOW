"""Multi-objective comparison: dominance, fronts, crowding and hypervolume.

Objectives are named metrics to maximise or minimise.  Everything here works on
plain mappings ``{objective name: value}`` and is exact; the hypervolume uses
recursive slicing, which is fine for the few hundred points of a search front
in two to four objectives and too slow for much more.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")
Metrics = Mapping[str, float]

_CHUNK = 256


@dataclass(frozen=True)
class Objective:
    name: str
    maximize: bool = True


def _oriented(points: Sequence[Metrics], objectives: Sequence[Objective]) -> np.ndarray:
    """Matrix [points, objectives] in which larger is always better."""
    signs = np.array([1.0 if o.maximize else -1.0 for o in objectives])
    raw = np.array([[p[o.name] for o in objectives] for p in points], dtype=np.float64)
    return raw.reshape(len(points), len(objectives)) * signs


def dominates(a: Metrics, b: Metrics, objectives: Sequence[Objective]) -> bool:
    """``a`` is at least as good as ``b`` everywhere and strictly better somewhere."""
    better = False
    for o in objectives:
        x, y = (a[o.name], b[o.name]) if o.maximize else (b[o.name], a[o.name])
        if x < y:
            return False
        better = better or x > y
    return better


def _dominated(matrix: np.ndarray) -> np.ndarray:
    """Boolean mask of rows dominated by some other row."""
    n = len(matrix)
    out = np.zeros(n, dtype=bool)
    for start in range(0, n, _CHUNK):
        block = matrix[start:start + _CHUNK]
        no_worse = (matrix[None, :, :] >= block[:, None, :]).all(axis=2)
        better = (matrix[None, :, :] > block[:, None, :]).any(axis=2)
        out[start:start + _CHUNK] = (no_worse & better).any(axis=1)
    return out


def pareto_front(items: Sequence[T], objectives: Sequence[Objective],
                 metrics: Callable[[Any], Metrics] = lambda item: item.metrics) -> list[T]:  # fmt: skip
    """Non-dominated items in input order; items equal in every objective are all kept."""
    if not items:
        return []
    keep = ~_dominated(_oriented([metrics(item) for item in items], objectives))
    return [item for item, ok in zip(items, keep) if ok]


def nondominated_sort(points: Sequence[Metrics], objectives: Sequence[Objective]) -> list[list[int]]:
    """Indices grouped into successive Pareto fronts, best first."""
    n = len(points)
    if n == 0:
        return []
    matrix = _oriented(points, objectives)
    no_worse = np.ones((n, n), dtype=bool)
    better = np.zeros((n, n), dtype=bool)
    for k in range(matrix.shape[1]):
        column = matrix[:, k]
        no_worse &= column[:, None] >= column[None, :]
        better |= column[:, None] > column[None, :]
    dominating = no_worse & better  # [i, j]: i dominates j
    count = dominating.sum(axis=0)
    remaining = np.ones(n, dtype=bool)
    fronts: list[list[int]] = []
    while remaining.any():
        front = np.flatnonzero(remaining & (count == 0))
        fronts.append(front.tolist())
        remaining[front] = False
        count = count - dominating[front].sum(axis=0)
    return fronts


def crowding_distance(points: Sequence[Metrics], objectives: Sequence[Objective]) -> list[float]:
    """NSGA-II crowding distance of each point within one front (infinite at the extremes)."""
    n = len(points)
    if n == 0:
        return []
    matrix = _oriented(points, objectives)
    distance = np.zeros(n)
    for k in range(matrix.shape[1]):
        order = np.argsort(matrix[:, k], kind="stable")
        low, high = matrix[order[0], k], matrix[order[-1], k]
        distance[order[0]] = distance[order[-1]] = np.inf
        if n > 2 and high > low:
            distance[order[1:-1]] += (matrix[order[2:], k] - matrix[order[:-2], k]) / (high - low)
    return distance.tolist()


def _slice_volume(points: list[tuple[float, ...]]) -> float:
    """Volume dominated by ``points`` above the origin (all coordinates positive)."""
    dims = len(points[0])
    if dims == 1:
        return max(p[0] for p in points)
    ordered = sorted(points, reverse=True)
    if dims == 2:
        volume = height = 0.0
        for x, y in ordered:
            if y > height:
                volume += x * (y - height)
                height = y
        return volume
    volume = 0.0
    for i, point in enumerate(ordered):
        below = ordered[i + 1][0] if i + 1 < len(ordered) else 0.0
        if point[0] > below:
            volume += (point[0] - below) * _slice_volume([q[1:] for q in ordered[:i + 1]])
    return volume


def hypervolume(points: Sequence[Metrics], objectives: Sequence[Objective],
                reference: Metrics) -> float:  # fmt: skip
    """Exact volume of objective space dominated by ``points`` and bounded by ``reference``.

    Points that are not strictly better than the reference in every objective
    contribute nothing.
    """
    if not points or not objectives:
        return 0.0
    origin = _oriented([reference], objectives)[0]
    shifted = _oriented(points, objectives) - origin
    shifted = shifted[(shifted > 0).all(axis=1)]
    if len(shifted) == 0:
        return 0.0
    shifted = shifted[~_dominated(shifted)]
    return _slice_volume(sorted({tuple(row) for row in shifted.tolist()}))


def ideal_nadir(points: Sequence[Metrics],
                objectives: Sequence[Objective]) -> tuple[dict[str, float], dict[str, float]]:  # fmt: skip
    """Best and worst value of every objective over ``points``."""
    ideal: dict[str, float] = {}
    nadir: dict[str, float] = {}
    for o in objectives:
        values = [p[o.name] for p in points]
        ideal[o.name] = max(values) if o.maximize else min(values)
        nadir[o.name] = min(values) if o.maximize else max(values)
    return ideal, nadir


def chebyshev(metrics: Metrics, objectives: Sequence[Objective],
              weights: Sequence[float] | Mapping[str, float], ideal: Metrics, nadir: Metrics,
              rho: float = 0.05) -> float:  # fmt: skip
    """Augmented Chebyshev scalarisation (lower is better), as used by ParEGO.

    Each objective is normalised so the ideal value is 0 and the nadir value 1;
    the result is ``max_k(w_k z_k) + rho * sum_k(w_k z_k)``.
    """
    terms = []
    for k, o in enumerate(objectives):
        weight = weights[o.name] if isinstance(weights, Mapping) else weights[k]
        span = nadir[o.name] - ideal[o.name]
        terms.append(weight * ((metrics[o.name] - ideal[o.name]) / span if span else 0.0))
    return max(terms) + rho * sum(terms)
