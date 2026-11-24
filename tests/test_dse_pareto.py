"""Dominance, Pareto fronts, crowding distance, hypervolume and scalarisation."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from basp.dse import (
    Objective,
    chebyshev,
    crowding_distance,
    dominates,
    hypervolume,
    ideal_nadir,
    nondominated_sort,
    pareto_front,
)

SPEED, COST, POWER = Objective("speed"), Objective("cost", maximize=False), Objective("power", False)
TWO = (SPEED, COST)


def pt(speed: float, cost: float, power: float = 0.0) -> dict[str, float]:
    return {"speed": speed, "cost": cost, "power": power}


def test_dominance_respects_objective_direction():
    assert dominates(pt(10, 5), pt(9, 5), TWO)  # faster at equal cost
    assert dominates(pt(10, 4), pt(10, 5), TWO)  # cheaper at equal speed
    assert not dominates(pt(10, 5), pt(10, 5), TWO)  # equal points do not dominate each other
    assert not dominates(pt(10, 6), pt(9, 5), TWO) and not dominates(pt(9, 5), pt(10, 6), TWO)
    assert dominates(pt(1, 1, 1), pt(1, 1, 2), (SPEED, COST, POWER))


def test_pareto_front_keeps_input_order_and_duplicates():
    points = [pt(1, 1), pt(3, 3), pt(2, 3), pt(3, 3), pt(5, 9), pt(0, 2)]
    items = [SimpleNamespace(metrics=p, tag=i) for i, p in enumerate(points)]
    assert [item.tag for item in pareto_front(items, TWO)] == [0, 1, 3, 4]
    assert pareto_front(points, TWO, metrics=lambda p: p) == [points[i] for i in (0, 1, 3, 4)]
    assert pareto_front([], TWO) == []
    # a single objective keeps exactly the maxima
    assert [i.tag for i in pareto_front(items, (SPEED,))] == [4]


def test_nondominated_sort_peels_successive_fronts():
    points = [pt(1, 1), pt(2, 2), pt(3, 3), pt(1, 2), pt(2, 3), pt(1, 3)]
    assert nondominated_sort(points, TWO) == [[0, 1, 2], [3, 4], [5]]
    assert nondominated_sort([], TWO) == []
    assert nondominated_sort([pt(1, 1), pt(1, 1)], TWO) == [[0, 1]]
    rng = np.random.default_rng(0)
    cloud = [pt(*rng.integers(0, 10, size=3)) for _ in range(60)]
    objectives = (SPEED, COST, POWER)
    fronts = nondominated_sort(cloud, objectives)
    assert sorted(i for front in fronts for i in front) == list(range(60))
    for level, front in enumerate(fronts):
        for i in front:
            assert not any(dominates(cloud[j], cloud[i], objectives) for j in front)
            if level:  # every point is dominated by something in the previous front
                assert any(dominates(cloud[j], cloud[i], objectives) for j in fronts[level - 1])


def test_crowding_distance_matches_hand_computation():
    points = [pt(0, 0), pt(1, 2), pt(4, 3), pt(10, 10)]
    distance = crowding_distance(points, TWO)
    assert distance[0] == distance[3] == float("inf")
    # normalised neighbour gaps: speed (4-0)/10, cost (3-0)/10  and  speed (10-1)/10, cost (10-2)/10
    assert distance[1] == pytest.approx(0.4 + 0.3) and distance[2] == pytest.approx(0.9 + 0.8)
    assert crowding_distance(points[:2], TWO) == [float("inf")] * 2
    assert crowding_distance([], TWO) == []
    # no spread: only the (arbitrary) extremes are marked, nothing divides by zero
    assert crowding_distance([pt(1, 1), pt(1, 1), pt(1, 1)], TWO) == [float("inf"), 0.0, float("inf")]


def test_hypervolume_two_objectives_by_hand():
    both_max = (Objective("x"), Objective("y"))
    stairs = [{"x": 1, "y": 3}, {"x": 2, "y": 2}, {"x": 3, "y": 1}]
    assert hypervolume(stairs, both_max, {"x": 0, "y": 0}) == pytest.approx(6.0)
    assert hypervolume([*stairs, {"x": 1, "y": 1}], both_max, {"x": 0, "y": 0}) == pytest.approx(6.0)
    assert hypervolume(stairs, both_max, {"x": 1, "y": 1}) == pytest.approx(1.0)  # only (2, 2) clears it
    assert hypervolume(stairs, both_max, {"x": 5, "y": 5}) == 0.0
    assert hypervolume([], both_max, {"x": 0, "y": 0}) == 0.0
    # speed maximised, cost minimised, reference (0 speed, 10 cost)
    mixed = [pt(4, 8), pt(2, 5)]
    assert hypervolume(mixed, TWO, pt(0, 10)) == pytest.approx(4 * 2 + 2 * 3)


def test_hypervolume_three_objectives_by_hand():
    objectives = tuple(Objective(n) for n in "xyz")
    origin = dict.fromkeys("xyz", 0.0)

    def p(x, y, z):
        return {"x": x, "y": y, "z": z}

    assert hypervolume([p(1, 1, 1)], objectives, origin) == pytest.approx(1.0)
    assert hypervolume([p(2, 1, 1), p(1, 2, 1)], objectives, origin) == pytest.approx(3.0)
    assert hypervolume([p(2, 2, 1), p(1, 1, 2)], objectives, origin) == pytest.approx(5.0)
    assert hypervolume([p(3, 1, 1), p(1, 3, 1), p(1, 1, 3)], objectives, origin) == pytest.approx(7.0)


@pytest.mark.parametrize(("dims", "seed"), [(2, 0), (3, 1), (3, 2), (4, 3)])
def test_hypervolume_equals_cell_counting_on_integer_points(dims: int, seed: int):
    """Exact check: with integer coordinates the volume is the number of dominated unit cells."""
    rng = np.random.default_rng(seed)
    names = "wxyz"[:dims]
    objectives = tuple(Objective(n) for n in names)
    points = [dict(zip(names, rng.integers(1, 6, size=dims).tolist())) for _ in range(12)]
    cells = sum(
        any(all(p[n] > c for n, c in zip(names, cell)) for p in points)
        for cell in itertools.product(range(5), repeat=dims)
    )
    assert hypervolume(points, objectives, dict.fromkeys(names, 0)) == pytest.approx(cells)
    # the front alone spans the same volume
    front = pareto_front(points, objectives, metrics=lambda p: p)
    assert hypervolume(front, objectives, dict.fromkeys(names, 0)) == pytest.approx(cells)


def test_ideal_nadir_and_chebyshev_scalarisation():
    points = [pt(10, 8), pt(4, 2), pt(7, 5)]
    ideal, nadir = ideal_nadir(points, TWO)
    assert ideal == {"speed": 10, "cost": 2} and nadir == {"speed": 4, "cost": 8}
    weights = (0.5, 0.5)
    # (10, 8): speed at its ideal (0), cost at its nadir (1) -> 0.5 + 0.05 * 0.5
    assert chebyshev(points[0], TWO, weights, ideal, nadir) == pytest.approx(0.5 + 0.05 * 0.5)
    # (7, 5): half-way in both -> max 0.25, sum 0.5
    assert chebyshev(points[2], TWO, weights, ideal, nadir) == pytest.approx(0.25 + 0.05 * 0.5)
    assert chebyshev(pt(10, 2), TWO, weights, ideal, nadir) == 0.0  # the utopia point
    by_name = chebyshev(points[2], TWO, {"speed": 0.5, "cost": 0.5}, ideal, nadir)
    assert by_name == chebyshev(points[2], TWO, weights, ideal, nadir)
    assert chebyshev(points[0], TWO, weights, ideal, ideal) == 0.0  # degenerate range
    # scalarisation is consistent with dominance for positive weights
    rng = np.random.default_rng(4)
    cloud = [pt(*rng.uniform(0, 10, size=2)) for _ in range(40)]
    ideal, nadir = ideal_nadir(cloud, TWO)
    for a, b in itertools.permutations(cloud, 2):
        if dominates(a, b, TWO):
            w = rng.dirichlet(np.ones(2))
            assert chebyshev(a, TWO, w, ideal, nadir) < chebyshev(b, TWO, w, ideal, nadir)
