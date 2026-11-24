"""Multi-objective search strategies against brute-force ground truth.

Two synthetic problems with a few thousand configurations each have fronts that
uniform sampling rarely hits (the good designs need several parameters at
specific levels at once), so a search that merely samples cannot pass.
"""

from __future__ import annotations

import math
import statistics
from concurrent.futures import ProcessPoolExecutor

import pytest

from basp.dse import (
    DesignSpace,
    Objective,
    Param,
    bayesian,
    dominates,
    evolutionary,
    exhaustive,
    hypervolume,
    random_search,
    reference_point,
    search,
)

# -- problem 1: two objectives, 8^4 = 4096 configurations (a discrete ZDT1) ----------------
ZDT = DesignSpace([Param(name, tuple(range(8))) for name in "abcd"])
ZDT_OBJECTIVES = (Objective("cost", maximize=False), Objective("loss", maximize=False))


def zdt(config: dict) -> dict[str, float]:
    cost = config["a"] / 7
    g = 1 + 9 * (abs(config["b"] - 5) + abs(config["c"] - 2) + abs(config["d"] - 6)) / 21
    return {"cost": cost, "loss": g * (1 - math.sqrt(cost / g))}


# -- problem 2: three objectives, mixed directions, a categorical parameter, a space ------
# -- constraint and designs the evaluator rejects (a discrete DTLZ2): 5040 configurations --
KIND_PENALTY = {"x": 0.3, "y": 0.5, "z": 0.0, "w": 0.8}
DTLZ = DesignSpace(
    [Param(name, tuple(range(6))) for name in "abcd"] + [Param("kind", ("x", "y", "z", "w"), ordered=False)],
    constraints=[lambda c: not (c["a"] == 5 and c["b"] == 5)],
)
DTLZ_OBJECTIVES = (Objective("perf"), Objective("cost", maximize=False), Objective("power", maximize=False))


def dtlz(config: dict) -> dict[str, float] | None:
    if (config["kind"] == "w" and config["a"] < 2) or (config["c"] == 0 and config["d"] == 5):
        return None  # "does not fit"
    g = (abs(config["c"] - 3) + abs(config["d"] - 1)) / 5 + KIND_PENALTY[config["kind"]]
    ta, tb = config["a"] / 5 * math.pi / 2, config["b"] / 5 * math.pi / 2
    radius = 1 + g
    return {
        "perf": 2 - radius * math.cos(ta) * math.cos(tb),
        "cost": radius * math.cos(ta) * math.sin(tb),
        "power": radius * math.sin(ta),
    }


PROBLEMS = {"zdt": (ZDT, zdt, ZDT_OBJECTIVES), "dtlz": (DTLZ, dtlz, DTLZ_OBJECTIVES)}

# -- a small problem for brute-force and exhaustion checks: 4 * 3 * 2 = 24, 21 in the space
SMALL = DesignSpace(
    [Param("x", (0, 1, 2, 3)), Param("y", (0, 1, 2)), Param("mode", ("lo", "hi"), ordered=False)],
    constraints=[lambda c: not (c["x"] == 3 and c["y"] == 2), lambda c: c["x"] + c["y"] > 0 or c["mode"] == "lo"],
)
SMALL_OBJECTIVES = (Objective("speed"), Objective("cost", maximize=False))


def small(config: dict) -> dict[str, float] | None:
    if config["x"] == 0 and config["y"] == 2:
        return None
    boost = 1.5 if config["mode"] == "hi" else 1.0
    return {"speed": boost * (config["x"] + 2 * config["y"]), "cost": config["x"] ** 2 + config["y"] + 2 * (boost > 1)}


def signature(result) -> list[tuple]:
    return [(tuple(e.config.items()), tuple(sorted(e.metrics.items())), e.feasible) for e in result.evaluations]


@pytest.fixture(scope="module")
def truth() -> dict[str, tuple]:
    """Exhaustive ground truth: (result, hypervolume of the true front)."""
    out = {}
    for name, (space, evaluate, objectives) in PROBLEMS.items():
        result = exhaustive(space, evaluate, objectives)
        out[name] = (result, hypervolume([e.metrics for e in result.front], objectives, result.reference))
    return out


def quality(result, name: str, truth) -> float:
    """Share of the true front's hypervolume recovered (same reference: the true nadir)."""
    reference, true_volume = truth[name][0].reference, truth[name][1]
    objectives = PROBLEMS[name][2]
    return hypervolume([e.metrics for e in result.front], objectives, reference) / true_volume


def recovered(result, name: str, truth) -> float:
    """Share of the truly Pareto-optimal designs that the search found."""
    space = PROBLEMS[name][0]
    best = {space.key(e.config) for e in truth[name][0].front}
    return len(best & {space.key(e.config) for e in result.front}) / len(best)


# -- exhaustive --------------------------------------------------------------------------
def test_exhaustive_equals_brute_force():
    result = exhaustive(SMALL, small, SMALL_OBJECTIVES)
    configs = list(SMALL.enumerate())
    assert result.method == "exhaustive" and result.n_evaluated == len(configs) == 21
    assert [e.config for e in result.evaluations] == configs
    feasible = [(c, small(c)) for c in configs if small(c) is not None]
    brute = [c for c, m in feasible if not any(dominates(other, m, SMALL_OBJECTIVES) for _, other in feasible)]
    assert [e.config for e in result.front] == brute and 2 <= len(brute) < len(feasible)
    assert sum(not e.feasible for e in result.evaluations) == 2  # rejected designs stay on record
    assert all(e.feasible for e in result.front)
    with pytest.raises(ValueError, match="more than 10"):
        exhaustive(SMALL, small, SMALL_OBJECTIVES, limit=10)
    assert exhaustive(SMALL, small, SMALL_OBJECTIVES, limit=21).n_evaluated == 21


def test_exhaustive_finds_the_analytic_front(truth):
    result, _ = truth["zdt"]
    # the ZDT front is every cost level with the three other parameters at their targets
    assert sorted(tuple(e.config.values()) for e in result.front) == [(a, 5, 2, 6) for a in range(8)]
    dtlz_result, _ = truth["dtlz"]
    assert dtlz_result.n_evaluated == 6**4 * 4 - 6 * 6 * 4  # the space constraint removes a = b = 5
    assert all(e.config["kind"] == "z" and e.config["c"] == 3 and e.config["d"] == 1 for e in dtlz_result.front)
    assert sum(not e.feasible for e in dtlz_result.evaluations) > 100


# -- search quality ----------------------------------------------------------------------
SEEDS = (0, 1, 2)


@pytest.mark.parametrize("name", PROBLEMS)
def test_evolutionary_recovers_the_front_from_a_small_sample(name, truth):
    """NSGA-II with 256 evaluations (about 5-6% of the space).

    Measured over seeds 0-4: at least 99.5% of the true hypervolume on both
    problems, where random search with the same budget reaches about 96%.  The
    threshold of 98% sits between the two with margin on each side.
    """
    space, evaluate, objectives = PROBLEMS[name]
    scores, baseline, found, found_blind = [], [], [], []
    for seed in SEEDS:
        result = search(space, evaluate, objectives, method="evolutionary", budget=256, seed=seed)
        blind = random_search(space, evaluate, objectives, budget=256, seed=seed)
        assert result.n_evaluated <= 256 < 0.07 * truth[name][0].n_evaluated
        scores.append(quality(result, name, truth))
        baseline.append(quality(blind, name, truth))
        found.append(recovered(result, name, truth))
        found_blind.append(recovered(blind, name, truth))
    assert min(scores) >= 0.98, scores
    assert statistics.mean(scores) > statistics.mean(baseline) + 0.015, (scores, baseline)
    # it finds most of the truly optimal designs themselves; sampling finds a handful
    assert statistics.mean(found) >= 0.5, found
    assert statistics.mean(found) > statistics.mean(found_blind) + 0.3, (found, found_blind)


@pytest.mark.parametrize("name", PROBLEMS)
def test_bayesian_recovers_the_front_from_a_tiny_sample(name, truth):
    """ParEGO with 96 evaluations (about 2% of the space).

    Measured over seeds 0-4: at least 98.5% of the true hypervolume, against
    about 91-93% for random search with the same budget; threshold 96%.
    """
    space, evaluate, objectives = PROBLEMS[name]
    scores, baseline, found, found_blind = [], [], [], []
    for seed in SEEDS:
        result = bayesian(space, evaluate, objectives, budget=96, seed=seed)
        blind = random_search(space, evaluate, objectives, budget=96, seed=seed)
        assert result.n_evaluated == 96
        scores.append(quality(result, name, truth))
        baseline.append(quality(blind, name, truth))
        found.append(recovered(result, name, truth))
        found_blind.append(recovered(blind, name, truth))
    assert min(scores) >= 0.96, scores
    assert statistics.mean(scores) > statistics.mean(baseline) + 0.03, (scores, baseline)
    assert statistics.mean(found) > statistics.mean(found_blind) + 0.15, (found, found_blind)


def test_bayesian_beats_evolutionary_when_evaluations_are_scarce(truth):
    """The reason to have both: at 64 evaluations the surrogate is worth its cost."""
    space, evaluate, objectives = PROBLEMS["zdt"]
    model = [quality(bayesian(space, evaluate, objectives, budget=64, seed=s), "zdt", truth) for s in SEEDS]
    evo = [quality(search(space, evaluate, objectives, method="evolutionary", budget=64, seed=s), "zdt", truth)
           for s in SEEDS]  # fmt: skip
    assert statistics.mean(model) > statistics.mean(evo)


# -- contract ----------------------------------------------------------------------------
def test_results_are_deterministic_per_seed():
    space, evaluate, objectives = PROBLEMS["dtlz"]
    for run in (
        lambda seed: evolutionary(space, evaluate, objectives, population=12, generations=6, seed=seed),
        lambda seed: bayesian(space, evaluate, objectives, budget=40, seed=seed),
        lambda seed: random_search(space, evaluate, objectives, budget=40, seed=seed),
    ):
        first, again, other = run(3), run(3), run(4)
        assert signature(first) == signature(again) and first.history == again.history
        assert signature(first) != signature(other)


@pytest.mark.parametrize("method", ["exhaustive", "evolutionary", "bayesian", "random"])
def test_every_configuration_is_evaluated_at_most_once_and_is_in_the_space(method):
    calls: list[tuple] = []

    def counting(config: dict):
        calls.append(SMALL.key(config))
        return small(config)

    result = search(SMALL, counting, SMALL_OBJECTIVES, method=method, budget=64, seed=1)
    assert len(calls) == len(set(calls)) == result.n_evaluated == 21  # the whole space, once each
    assert all(SMALL.feasible(e.config) for e in result.evaluations)
    assert [SMALL.key(e.config) for e in result.evaluations] == calls
    truth_front = {SMALL.key(e.config) for e in exhaustive(SMALL, small, SMALL_OBJECTIVES).front}
    assert {SMALL.key(e.config) for e in result.front} == truth_front


def test_large_spaces_are_searched_by_sampling_without_duplicates():
    big = DesignSpace([Param(name, tuple(range(8))) for name in "abcdef"], constraints=[lambda c: c["a"] != c["f"]])
    assert big.size == 8**6  # too large to list: candidates come from sampling
    objectives = (Objective("gain"), Objective("cost", maximize=False))

    def evaluate(config: dict) -> dict[str, float]:
        return {"gain": sum(v * (i + 1) for i, v in enumerate(config.values())) - 3 * abs(config["c"] - 4),
                "cost": sum(config.values())}  # fmt: skip

    for result, budget in ((bayesian(big, evaluate, objectives, budget=40, seed=0), 40),
                           (evolutionary(big, evaluate, objectives, population=20, generations=9, seed=0), 200)):  # fmt: skip
        keys = [big.key(e.config) for e in result.evaluations]
        assert len(keys) == len(set(keys)) == budget and all(big.feasible(e.config) for e in result.evaluations)
    guided = evolutionary(big, evaluate, objectives, population=20, generations=9, seed=0)
    blind = random_search(big, evaluate, objectives, budget=200, seed=0)
    assert guided.best("gain").metrics["gain"] > blind.best("gain").metrics["gain"]


def test_infeasible_and_malformed_results_never_enter_a_front():
    def evaluate(config: dict):
        if config["x"] == 1:
            return None
        if config["x"] == 2:
            return {"speed": float("nan"), "cost": 1.0}
        if config["x"] == 3:
            return {"speed": 9.0}  # objective "cost" missing
        return {"speed": 1.0 + config["y"], "cost": 5, "note": "ok", "flag": True}

    result = exhaustive(SMALL, evaluate, SMALL_OBJECTIVES)
    by_x = {x: [e for e in result.evaluations if e.config["x"] == x] for x in range(4)}
    assert all(not e.feasible and e.metrics == {} for e in by_x[1])
    assert all(not e.feasible and "not finite" in e.info["reason"] for e in by_x[2])
    assert all(not e.feasible and "cost" in e.info["reason"] for e in by_x[3])
    good = by_x[0]
    assert all(e.feasible and e.metrics == {"speed": 1.0 + e.config["y"], "cost": 5.0} for e in good)
    assert good[0].info == {"note": "ok", "flag": True}  # non-numeric values are kept as information
    assert result.front and all(e.config["x"] == 0 for e in result.front)
    assert result.feasible == good and result.reference == reference_point([e.metrics for e in good], SMALL_OBJECTIVES)
    assert result.best("speed").metrics["speed"] == 3.0 and result.best("cost").config["x"] == 0
    assert result.best("speed", maximize=False).metrics["speed"] == 1.0
    assert result.best("unknown") is None
    nothing = exhaustive(SMALL, lambda c: None, SMALL_OBJECTIVES)
    assert nothing.front == [] and nothing.history == [0.0] and nothing.best("speed") is None
    for method in ("evolutionary", "bayesian"):
        assert search(SMALL, lambda c: None, SMALL_OBJECTIVES, method=method, budget=30).front == []
    with pytest.raises(ValueError, match="objective"):
        exhaustive(SMALL, small, ())


def test_starting_designs_are_evaluated_first_and_validated():
    seeds = [{"x": 3, "y": 1, "mode": "hi"}, {"mode": "lo", "y": 0, "x": 0}]
    evo = evolutionary(SMALL, small, SMALL_OBJECTIVES, population=6, generations=2, initial=seeds)
    assert [e.config for e in evo.evaluations[:2]] == [SMALL.canonical(c) for c in seeds]
    bo = bayesian(SMALL, small, SMALL_OBJECTIVES, budget=8, initial=4, initial_configs=seeds)
    assert [e.config for e in bo.evaluations[:2]] == [SMALL.canonical(c) for c in seeds]
    assert bo.batches[0] == 4 and bo.n_evaluated == 8
    for bad in ({"x": 3, "y": 2, "mode": "lo"}, {"x": 9, "y": 0, "mode": "lo"}):
        with pytest.raises(ValueError, match="not in the design space"):
            evolutionary(SMALL, small, SMALL_OBJECTIVES, initial=[bad])
        with pytest.raises(ValueError, match="not in the design space"):
            bayesian(SMALL, small, SMALL_OBJECTIVES, initial_configs=[bad])
    with pytest.raises(ValueError):
        evolutionary(SMALL, small, SMALL_OBJECTIVES, population=1)
    with pytest.raises(ValueError):
        bayesian(SMALL, small, SMALL_OBJECTIVES, budget=0)


def test_history_is_the_front_hypervolume_after_each_batch():
    space, evaluate, objectives = PROBLEMS["dtlz"]
    for result in (evolutionary(space, evaluate, objectives, population=16, generations=8, seed=2),
                   bayesian(space, evaluate, objectives, budget=30, initial=10, seed=2)):  # fmt: skip
        assert len(result.history) == len(result.batches) and result.batches[-1] == result.n_evaluated
        assert all(a <= b + 1e-12 for a, b in zip(result.history, result.history[1:]))  # never gets worse
        assert result.history[-1] > result.history[0] > 0
        feasible = [e.metrics for e in result.evaluations if e.feasible]
        assert result.reference == reference_point(feasible, objectives)
        final = hypervolume([e.metrics for e in result.front], objectives, result.reference)
        assert result.history[-1] == pytest.approx(final)
    assert evolutionary(space, evaluate, objectives, population=16, generations=8, seed=2).batches[0] == 16


def test_auto_picks_the_strategy_from_space_size_and_budget():
    space, evaluate, objectives = PROBLEMS["zdt"]
    assert search(SMALL, small, SMALL_OBJECTIVES, budget=21).method == "exhaustive"
    assert search(SMALL, small, SMALL_OBJECTIVES, budget=20).method == "bayesian"  # one too many to list
    scarce = search(space, evaluate, objectives, budget=40)
    assert scarce.method == "bayesian" and scarce.n_evaluated == 40
    plenty = search(space, evaluate, objectives, budget=300)
    assert plenty.method == "evolutionary" and 250 <= plenty.n_evaluated <= 300
    assert search(space, evaluate, objectives, method="random", budget=50).n_evaluated == 50
    with pytest.raises(ValueError, match="unknown search method"):
        search(space, evaluate, objectives, method="annealing")


def test_process_pool_gives_identical_results_to_builtin_map():
    space, evaluate, objectives = PROBLEMS["dtlz"]
    with ProcessPoolExecutor(max_workers=2) as pool:
        listed = exhaustive(SMALL, small, SMALL_OBJECTIVES, map_fn=pool.map)
        evolved = evolutionary(space, evaluate, objectives, population=12, generations=4, seed=5, map_fn=pool.map)
        sampled = random_search(space, evaluate, objectives, budget=48, seed=5, map_fn=pool.map)
    assert signature(listed) == signature(exhaustive(SMALL, small, SMALL_OBJECTIVES))
    serial = evolutionary(space, evaluate, objectives, population=12, generations=4, seed=5)
    assert signature(evolved) == signature(serial) and evolved.history == serial.history
    assert signature(sampled) == signature(random_search(space, evaluate, objectives, budget=48, seed=5))
