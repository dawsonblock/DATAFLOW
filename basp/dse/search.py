"""Multi-objective search over a :class:`~basp.dse.space.DesignSpace`.

Four strategies share one contract -- ``evaluate(config)`` returns a mapping of
metrics, or ``None`` when the design is infeasible -- and one result type:

* :func:`exhaustive` evaluates every feasible configuration (small spaces).
* :func:`bayesian` is ParEGO: each step scalarises the objectives with a random
  augmented-Chebyshev weight vector, fits a Gaussian process to the scalarised
  values and evaluates the candidate of highest expected improvement.  The
  surrogate is a small dense NumPy implementation (Matern 5/2 kernel, unit
  signal variance, length-scale chosen from a fixed grid by marginal
  likelihood); it is meant for hundreds of evaluations, not thousands, and it
  proposes one design at a time.
* :func:`evolutionary` is NSGA-II (non-dominated sorting, crowding distance,
  binary tournaments, uniform crossover, mutation, elitist survival) restricted
  to configurations that have not been evaluated before; each generation is one
  batch, so it parallelises.
* :func:`random_search` is the baseline the other two must beat.

Every configuration is evaluated at most once, results are deterministic for a
given seed, and they do not depend on whether ``map_fn`` is the builtin ``map``
or a process pool's ``map``.

``SearchResult.history`` is the hypervolume of the feasible front after each
batch.  All entries use the same reference point: the worst value of every
objective over the feasible evaluations of the whole run
(``SearchResult.reference``).  Histories of different runs are therefore only
comparable after recomputing them against a common reference.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np

from basp.dse.pareto import (
    Objective,
    chebyshev,
    crowding_distance,
    hypervolume,
    ideal_nadir,
    nondominated_sort,
    pareto_front,
)
from basp.dse.space import DesignSpace

Evaluator = Callable[[dict], Mapping[str, float] | None]
MapFn = Callable[[Callable[[dict], Any], Iterable[dict]], Iterable[Any]]

_ENUMERATE_LIMIT = 20_000  # spaces up to this size are listed once for candidate generation
_LENGTH_SCALES = (0.15, 0.3, 0.6, 1.2, 2.4)
_NOISE = 1e-4


@dataclass
class Evaluation:
    config: dict
    metrics: dict[str, float]
    feasible: bool = True
    info: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    method: str
    evaluations: list[Evaluation]
    front: list[Evaluation]
    history: list[float] = field(default_factory=list)
    objectives: tuple[Objective, ...] = ()
    reference: dict[str, float] = field(default_factory=dict)
    batches: list[int] = field(default_factory=list)  # evaluations completed after each batch

    @property
    def n_evaluated(self) -> int:
        return len(self.evaluations)

    @property
    def feasible(self) -> list[Evaluation]:
        return [e for e in self.evaluations if e.feasible]

    def best(self, objective_name: str, maximize: bool | None = None) -> Evaluation | None:
        """Feasible evaluation with the best value of one metric (first found on ties).

        The direction comes from the run's objectives; for a metric that was not
        an objective pass ``maximize`` (default: larger is better).
        """
        if maximize is None:
            maximize = next((o.maximize for o in self.objectives if o.name == objective_name), True)
        best = None
        for e in self.evaluations:
            if not e.feasible or objective_name not in e.metrics:
                continue
            value = e.metrics[objective_name]
            if best is None or (value > best[0] if maximize else value < best[0]):
                best = (value, e)
        return None if best is None else best[1]


def reference_point(points: Sequence[Mapping[str, float]],
                    objectives: Sequence[Objective]) -> dict[str, float]:  # fmt: skip
    """The nadir of ``points``: the worst value of each objective."""
    return ideal_nadir(points, objectives)[1] if points else {}


class _Run:
    """Evaluation cache and bookkeeping shared by all strategies."""

    def __init__(self, space: DesignSpace, evaluate: Evaluator, objectives: Sequence[Objective],
                 method: str, map_fn: MapFn = map) -> None:  # fmt: skip
        if not objectives:
            raise ValueError("at least one objective is required")
        self.space = space
        self.evaluate_fn = evaluate
        self.objectives = tuple(objectives)
        self.method = method
        self.map_fn = map_fn
        self.cache: dict[tuple, Evaluation] = {}
        self.evaluations: list[Evaluation] = []
        self.batches: list[int] = []

    @property
    def n(self) -> int:
        return len(self.evaluations)

    def seen(self, config: Mapping[str, Any]) -> bool:
        return self.space.key(config) in self.cache

    def checked(self, configs: Sequence[Mapping[str, Any]], what: str) -> list[dict]:
        for config in configs:
            if not self.space.feasible(config):
                raise ValueError(f"{what} configuration is not in the design space: {dict(config)!r}")
        return [self.space.canonical(c) for c in configs]

    def evaluate(self, configs: Sequence[Mapping[str, Any]]) -> list[Evaluation]:
        """Evaluate the not-yet-seen configurations as one batch; return all, in order."""
        fresh: list[dict] = []
        keys: set[tuple] = set()
        for config in configs:
            key = self.space.key(config)
            if key not in self.cache and key not in keys:
                keys.add(key)
                fresh.append(self.space.canonical(config))
        if fresh:
            results = list(self.map_fn(self.evaluate_fn, [dict(c) for c in fresh]))
            if len(results) != len(fresh):
                raise RuntimeError("map_fn returned a different number of results than configurations")
            for config, raw in zip(fresh, results):
                evaluation = self._record(config, raw)
                self.cache[self.space.key(config)] = evaluation
                self.evaluations.append(evaluation)
            self.batches.append(len(self.evaluations))
        return [self.cache[self.space.key(c)] for c in configs]

    def _record(self, config: dict, raw: Mapping[str, Any] | None) -> Evaluation:
        if raw is None:
            return Evaluation(config, {}, False, {"reason": "evaluator reported the design infeasible"})
        metrics: dict[str, float] = {}
        info: dict[str, Any] = {}
        for name, value in raw.items():
            if isinstance(value, (Real, np.number)) and not isinstance(value, bool):
                metrics[str(name)] = float(value)
            else:
                info[str(name)] = value
        unusable = [o.name for o in self.objectives
                    if o.name not in metrics or not math.isfinite(metrics[o.name])]  # fmt: skip
        if unusable:
            info["reason"] = f"objective(s) missing or not finite: {', '.join(unusable)}"
            return Evaluation(config, metrics, False, info)
        return Evaluation(config, metrics, True, info)

    def front(self) -> list[Evaluation]:
        return pareto_front([e for e in self.evaluations if e.feasible], self.objectives)

    def result(self) -> SearchResult:
        feasible = [e.metrics for e in self.evaluations if e.feasible]
        reference = reference_point(feasible, self.objectives)
        history: list[float] = []
        front: list[Mapping[str, float]] = []
        start = 0
        for end in self.batches:
            new = [e.metrics for e in self.evaluations[start:end] if e.feasible]
            front = pareto_front(front + new, self.objectives, metrics=lambda m: m)
            history.append(hypervolume(front, self.objectives, reference) if front else 0.0)
            start = end
        return SearchResult(self.method, list(self.evaluations), self.front(), history,
                            self.objectives, reference, list(self.batches))  # fmt: skip


def _unique_samples(run: _Run, rng: np.random.Generator, count: int, taken: set[tuple]) -> list[dict]:
    """Up to ``count`` random configurations that are neither evaluated nor in ``taken``."""
    space = run.space
    found: list[dict] = []
    for _ in range(50 * max(count, 0)):
        if len(found) >= count:
            break
        config = space.sample(rng)
        key = space.key(config)
        if key not in run.cache and key not in taken:
            taken.add(key)
            found.append(config)
    if len(found) < count and space.size <= _ENUMERATE_LIMIT:
        # Nearly exhausted small space: list what is left instead of relying on luck.
        left = [c for c in space.enumerate()
                if space.key(c) not in run.cache and space.key(c) not in taken]  # fmt: skip
        for i in rng.permutation(len(left))[:count - len(found)]:
            taken.add(space.key(left[int(i)]))
            found.append(left[int(i)])
    return found


# -- exhaustive -------------------------------------------------------------------------
def exhaustive(space: DesignSpace, evaluate: Evaluator, objectives: Sequence[Objective], *,
               limit: int | None = None, map_fn: MapFn = map) -> SearchResult:  # fmt: skip
    """Evaluate every feasible configuration (one batch)."""
    configs: list[dict] = []
    for config in space.enumerate():
        configs.append(config)
        if limit is not None and len(configs) > limit:
            raise ValueError(f"the space has more than {limit} feasible configurations")
    run = _Run(space, evaluate, objectives, "exhaustive", map_fn)
    run.evaluate(configs)
    return run.result()


# -- random baseline --------------------------------------------------------------------
def random_search(space: DesignSpace, evaluate: Evaluator, objectives: Sequence[Objective], *,
                  budget: int = 256, seed: int = 0, batch: int = 32,
                  map_fn: MapFn = map) -> SearchResult:  # fmt: skip
    """Evaluate up to ``budget`` distinct uniformly sampled configurations."""
    rng = np.random.default_rng(seed)
    run = _Run(space, evaluate, objectives, "random", map_fn)
    while run.n < budget:
        configs = _unique_samples(run, rng, min(batch, budget - run.n), set())
        if not configs:
            break
        run.evaluate(configs)
    return run.result()


# -- evolutionary (NSGA-II) -------------------------------------------------------------
def _rank(evaluations: Sequence[Evaluation],
          objectives: Sequence[Objective]) -> tuple[list[int], list[float]]:  # fmt: skip
    """Front index and crowding distance of each evaluation; infeasible ones rank last."""
    rank = [len(evaluations)] * len(evaluations)
    crowd = [0.0] * len(evaluations)
    feasible = [i for i, e in enumerate(evaluations) if e.feasible]
    points = [evaluations[i].metrics for i in feasible]
    for level, front in enumerate(nondominated_sort(points, objectives)):
        distances = crowding_distance([points[j] for j in front], objectives)
        for j, distance in zip(front, distances):
            rank[feasible[j]] = level
            crowd[feasible[j]] = distance
    return rank, crowd


def evolutionary(space: DesignSpace, evaluate: Evaluator, objectives: Sequence[Objective], *,
                 population: int = 32, generations: int = 20, seed: int = 0,
                 mutation_rate: float | None = None, initial: Sequence[dict] = (),
                 map_fn: MapFn = map) -> SearchResult:  # fmt: skip
    """NSGA-II over never-before-evaluated offspring.

    Evaluates at most ``population * (generations + 1)`` configurations and
    stops early once no unevaluated configuration can be produced.
    """
    if population < 2 or generations < 0:
        raise ValueError("population must be >= 2 and generations >= 0")
    rng = np.random.default_rng(seed)
    run = _Run(space, evaluate, objectives, "evolutionary", map_fn)
    start = run.checked(initial, "initial")
    taken = {space.key(c) for c in start}
    start += _unique_samples(run, rng, population - len(start), taken)
    parents = run.evaluate(start)
    for _ in range(generations):
        if not parents:
            break
        rank, crowd = _rank(parents, objectives)

        def pick(rank=rank, crowd=crowd, pool=parents) -> dict:
            i, j = (int(v) for v in rng.integers(len(pool), size=2))
            better = i if (rank[i], -crowd[i]) <= (rank[j], -crowd[j]) else j
            return pool[better].config

        children: list[dict] = []
        taken = set()
        for _attempt in range(20 * population):
            if len(children) >= population:
                break
            child = space.mutate(space.crossover(pick(), pick(), rng), rng, mutation_rate)
            key = space.key(child)
            if key not in run.cache and key not in taken:
                taken.add(key)
                children.append(child)
        children += _unique_samples(run, rng, population - len(children), taken)
        if not children:
            break
        combined = parents + run.evaluate(children)
        rank, crowd = _rank(combined, objectives)
        order = sorted(range(len(combined)), key=lambda i: (rank[i], -crowd[i], i))
        parents = [combined[i] for i in order[:population]]
    return run.result()


# -- Bayesian (ParEGO) ------------------------------------------------------------------
def _matern52(sq_dist: np.ndarray, length: float) -> np.ndarray:
    r = np.sqrt(5.0 * np.maximum(sq_dist, 0.0)) / length
    return (1.0 + r + r * r / 3.0) * np.exp(-r)


def _sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * a @ b.T


class _GaussianProcess:
    """Zero-mean GP on standardised targets; length-scale picked by marginal likelihood."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.mean = float(y.mean())
        self.scale = float(y.std()) or 1.0
        target = (y - self.mean) / self.scale
        sq = _sq_dist(x, x)
        best: tuple[float, float, np.ndarray, np.ndarray] | None = None
        for length in _LENGTH_SCALES:
            kernel = _matern52(sq, length) + _NOISE * np.eye(len(x))
            try:
                chol = np.linalg.cholesky(kernel)
            except np.linalg.LinAlgError:
                continue
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, target))
            evidence = -0.5 * float(target @ alpha) - float(np.log(np.diag(chol)).sum())
            if best is None or evidence > best[0]:
                best = (evidence, length, chol, alpha)
        if best is None:
            raise np.linalg.LinAlgError("kernel matrix is not positive definite")
        _, self.length, self.chol, self.alpha = best

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cross = _matern52(_sq_dist(x, self.x), self.length)
        mean = cross @ self.alpha
        solved = np.linalg.solve(self.chol, cross.T)
        variance = np.maximum(1.0 + _NOISE - (solved * solved).sum(axis=0), 1e-12)
        return mean * self.scale + self.mean, np.sqrt(variance) * self.scale


def _expected_improvement(mean: np.ndarray, std: np.ndarray, best: float) -> np.ndarray:
    """Expected improvement below ``best`` (minimisation)."""
    gain = best - mean
    z = gain / std
    cdf = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in z])
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return gain * cdf + std * pdf


class _Candidates:
    """Unevaluated configurations to score: the whole space when it is small enough to
    list, otherwise random samples; always including the neighbours of the front."""

    def __init__(self, space: DesignSpace, count: int) -> None:
        self.space = space
        self.count = count
        self.listed: list[dict] | None = None
        if space.size <= _ENUMERATE_LIMIT:
            self.listed = list(space.enumerate())
            self.index = {space.key(c): i for i, c in enumerate(self.listed)}
            self.open = np.ones(len(self.listed), dtype=bool)
            self.encoded = np.array([space.encode(c) for c in self.listed]).reshape(-1, space.dim)

    def mark(self, config: Mapping[str, Any]) -> None:
        if self.listed is not None:
            self.open[self.index[self.space.key(config)]] = False

    def draw(self, run: _Run, rng: np.random.Generator) -> tuple[list[dict], np.ndarray]:
        space = self.space
        near = [n for e in run.front() for n in space.neighbors(e.config) if not run.seen(n)]
        if self.listed is not None:
            free = np.flatnonzero(self.open)
            if len(free) > self.count:
                free = np.sort(rng.choice(free, size=self.count, replace=False))
            chosen = dict.fromkeys(free.tolist())
            chosen.update(dict.fromkeys(self.index[space.key(n)] for n in near))
            picked = list(chosen)
            return [self.listed[i] for i in picked], self.encoded[picked]
        taken: set[tuple] = set()
        configs = _unique_samples(run, rng, self.count, taken)
        for n in near:
            if space.key(n) not in taken:
                taken.add(space.key(n))
                configs.append(n)
        encoded = np.array([space.encode(c) for c in configs]).reshape(-1, space.dim)
        return configs, encoded


def bayesian(space: DesignSpace, evaluate: Evaluator, objectives: Sequence[Objective], *,
             budget: int = 64, initial: int = 12, seed: int = 0, candidates: int = 512,
             initial_configs: Sequence[dict] = ()) -> SearchResult:  # fmt: skip
    """ParEGO with a Gaussian-process surrogate; evaluates at most ``budget`` designs.

    ``initial`` is the size of the starting design: ``initial_configs`` first,
    then random configurations.  Infeasible evaluations are given a scalarised
    value worse than every feasible one so the surrogate learns to avoid their
    neighbourhood.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    rng = np.random.default_rng(seed)
    run = _Run(space, evaluate, objectives, "bayesian")
    pool = _Candidates(space, candidates)
    start = run.checked(initial_configs, "initial")
    taken = {space.key(c) for c in start}
    start += _unique_samples(run, rng, initial - len(start), taken)
    encoded: list[np.ndarray] = []

    def evaluate_batch(configs: Sequence[dict]) -> None:
        before = run.n
        run.evaluate(configs)
        for e in run.evaluations[before:]:
            pool.mark(e.config)
            encoded.append(space.encode(e.config))

    evaluate_batch(start[:budget])
    while run.n < budget:
        configs, features = pool.draw(run, rng)
        if not configs:
            break
        feasible = [e for e in run.evaluations if e.feasible]
        if len(feasible) < 2:
            choice = int(rng.integers(len(configs)))
        else:
            weights = rng.dirichlet(np.ones(len(objectives)))
            ideal, nadir = ideal_nadir([e.metrics for e in feasible], objectives)
            scores = {id(e): chebyshev(e.metrics, objectives, weights, ideal, nadir) for e in feasible}
            low, high = min(scores.values()), max(scores.values())
            penalty = high + 0.5 * ((high - low) or 1.0)
            target = np.array([scores.get(id(e), penalty) for e in run.evaluations])
            model = _GaussianProcess(np.array(encoded), target)
            mean, std = model.predict(features)
            choice = int(np.argmax(_expected_improvement(mean, std, low)))
        evaluate_batch([configs[choice]])
    return run.result()


# -- dispatcher -------------------------------------------------------------------------
def search(space: DesignSpace, evaluate: Evaluator, objectives: Sequence[Objective], *,
           method: str = "auto", budget: int = 256, seed: int = 0,
           map_fn: MapFn = map) -> SearchResult:  # fmt: skip
    """Pick a strategy for the budget and run it.

    ``auto`` enumerates the space when it has at most ``budget`` feasible
    configurations, uses the Bayesian search for budgets up to 128 evaluations
    (each one is assumed expensive) and the evolutionary search beyond that.
    """
    if method == "auto":
        feasible = sum(1 for _ in itertools.islice(space.enumerate(), budget + 1))
        method = "exhaustive" if feasible <= budget else "bayesian" if budget <= 128 else "evolutionary"
    if method == "exhaustive":
        return exhaustive(space, evaluate, objectives, map_fn=map_fn)
    if method == "bayesian":
        return bayesian(space, evaluate, objectives, budget=budget,
                        initial=max(2, min(12, budget // 4)), seed=seed)  # fmt: skip
    if method == "evolutionary":
        population = max(2, min(64, round(math.sqrt(budget))))
        generations = max(0, budget // population - 1)
        return evolutionary(space, evaluate, objectives, population=population,
                            generations=generations, seed=seed, map_fn=map_fn)  # fmt: skip
    if method == "random":
        return random_search(space, evaluate, objectives, budget=budget, seed=seed, map_fn=map_fn)
    raise ValueError(f"unknown search method {method!r}")
