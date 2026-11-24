"""Discrete design spaces.

A :class:`DesignSpace` is the Cartesian product of named parameters, each with a
finite list of values, restricted by arbitrary feasibility constraints.  Ordered
parameters (numeric levels) keep their neighbourhood structure for mutation and
for the surrogate model's encoding; unordered ones are categorical.

The space knows nothing about what a configuration means.  Values must be
hashable.  Configurations are plain dictionaries keyed by parameter name.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Config = dict[str, Any]
Constraint = Callable[[Mapping[str, Any]], bool]

_SAMPLE_TRIES = 10_000
_MUTATE_TRIES = 64
_JUMP = 0.2  # probability that an ordered parameter jumps instead of stepping to a neighbour


@dataclass(frozen=True)
class Param:
    name: str
    values: tuple
    ordered: bool = True

    def __post_init__(self) -> None:
        values = tuple(self.values)
        object.__setattr__(self, "values", values)
        if not values:
            raise ValueError(f"parameter {self.name!r} has no values")
        try:
            distinct = len(set(values))
        except TypeError as exc:
            raise ValueError(f"parameter {self.name!r} has unhashable values") from exc
        if distinct != len(values):
            raise ValueError(f"parameter {self.name!r} lists a value twice")


class DesignSpace:
    def __init__(self, params: Sequence[Param], constraints: Sequence[Constraint] = ()) -> None:
        self._params = tuple(params)
        if not self._params:
            raise ValueError("a design space needs at least one parameter")
        self._names = tuple(p.name for p in self._params)
        if len(set(self._names)) != len(self._names):
            raise ValueError("parameter names must be unique")
        self.constraints = tuple(constraints)
        self._index = {p.name: {v: i for i, v in enumerate(p.values)} for p in self._params}

    # -- shape --------------------------------------------------------------------
    @property
    def params(self) -> tuple[Param, ...]:
        return self._params

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def size(self) -> int:
        """Number of value combinations, ignoring constraints."""
        return math.prod(len(p.values) for p in self._params)

    @property
    def dim(self) -> int:
        """Length of the vector produced by :meth:`encode`."""
        return sum(1 if p.ordered else len(p.values) for p in self._params)

    # -- identity and feasibility -------------------------------------------------
    def _satisfies(self, config: Mapping[str, Any]) -> bool:
        return all(constraint(config) for constraint in self.constraints)

    def canonical(self, config: Mapping[str, Any]) -> Config:
        """Copy of ``config`` with exactly the space's parameters, in parameter order."""
        return {name: config[name] for name in self._names}

    def feasible(self, config: Mapping[str, Any]) -> bool:
        """True if ``config`` has exactly the space's parameters, legal values and
        satisfies every constraint."""
        if len(config) != len(self._names):
            return False
        try:
            if any(config[name] not in self._index[name] for name in self._names):
                return False
        except (KeyError, TypeError):
            return False
        return self._satisfies(config)

    def key(self, config: Mapping[str, Any]) -> tuple:
        return tuple(config[name] for name in self._names)

    # -- generation ---------------------------------------------------------------
    def enumerate(self) -> Iterator[Config]:
        """Every feasible configuration, in lexicographic order of the parameter values."""
        for combo in itertools.product(*(p.values for p in self._params)):
            config = dict(zip(self._names, combo))
            if self._satisfies(config):
                yield config

    def sample(self, rng: np.random.Generator) -> Config:
        """A configuration drawn uniformly per parameter, rejecting infeasible draws."""
        for _ in range(_SAMPLE_TRIES):
            config = {p.name: p.values[int(rng.integers(len(p.values)))] for p in self._params}
            if self._satisfies(config):
                return config
        raise ValueError(f"no feasible configuration found in {_SAMPLE_TRIES} draws")

    def encode(self, config: Mapping[str, Any]) -> np.ndarray:
        """Numeric vector: ordered parameters map to [0, 1], unordered ones to one-hot blocks."""
        out = np.zeros(self.dim, dtype=np.float64)
        pos = 0
        for p in self._params:
            i = self._index[p.name][config[p.name]]
            n = len(p.values)
            if p.ordered:
                out[pos] = i / (n - 1) if n > 1 else 0.5
                pos += 1
            else:
                out[pos + i] = 1.0
                pos += n
        return out

    # -- variation ----------------------------------------------------------------
    def neighbors(self, config: Mapping[str, Any]) -> list[Config]:
        """Feasible configurations that differ in one parameter: one step for ordered
        parameters, any other value for unordered ones."""
        base = self.canonical(config)
        found: list[Config] = []
        for p in self._params:
            i = self._index[p.name][base[p.name]]
            n = len(p.values)
            steps = (i - 1, i + 1) if p.ordered else tuple(j for j in range(n) if j != i)
            for j in steps:
                if 0 <= j < n:
                    candidate = {**base, p.name: p.values[j]}
                    if self._satisfies(candidate):
                        found.append(candidate)
        return found

    def _step(self, param: Param, value: Any, rng: np.random.Generator) -> Any:
        """A different value: usually an adjacent level for ordered parameters."""
        i = self._index[param.name][value]
        n = len(param.values)
        if param.ordered and rng.random() >= _JUMP:
            if i == 0:
                j = 1
            elif i == n - 1:
                j = n - 2
            else:
                j = i + (1 if rng.random() < 0.5 else -1)
        else:
            j = int(rng.integers(n - 1))
            j += j >= i
        return param.values[j]

    def mutate(self, config: Mapping[str, Any], rng: np.random.Generator,
               rate: float | None = None) -> Config:  # fmt: skip
        """A feasible configuration different from ``config``.

        Each parameter changes with probability ``rate`` (default ``1 / n_params``);
        at least one always changes.  If no feasible variant is found the input
        is returned unchanged.
        """
        base = self.canonical(config)
        movable = [p for p in self._params if len(p.values) > 1]
        if not movable:
            return base
        rate = 1.0 / len(self._params) if rate is None else rate
        for _ in range(_MUTATE_TRIES):
            child = dict(base)
            changed = False
            for p in movable:
                if rng.random() < rate:
                    child[p.name] = self._step(p, child[p.name], rng)
                    changed = True
            if not changed:
                p = movable[int(rng.integers(len(movable)))]
                child[p.name] = self._step(p, child[p.name], rng)
            if self._satisfies(child):
                return child
        options = self.neighbors(base)
        return options[int(rng.integers(len(options)))] if options else base

    def crossover(self, a: Mapping[str, Any], b: Mapping[str, Any],
                  rng: np.random.Generator) -> Config:  # fmt: skip
        """Uniform crossover; an infeasible child is replaced by one of its parents."""
        pa, pb = self.canonical(a), self.canonical(b)
        child = {name: (pa if rng.random() < 0.5 else pb)[name] for name in self._names}
        if self._satisfies(child):
            return child
        first, second = (pa, pb) if rng.random() < 0.5 else (pb, pa)
        return first if self.feasible(first) else second
