"""Design-space exploration: discrete spaces, Pareto tools and multi-objective search.

The engine optimises any ``evaluate(config) -> metrics`` callable and knows
nothing about the simulator; see :mod:`basp.dse.search` for the strategies.
"""

from basp.dse.pareto import (
    Objective,
    chebyshev,
    crowding_distance,
    dominates,
    hypervolume,
    ideal_nadir,
    nondominated_sort,
    pareto_front,
)
from basp.dse.search import (
    Evaluation,
    Evaluator,
    SearchResult,
    bayesian,
    evolutionary,
    exhaustive,
    random_search,
    reference_point,
    search,
)
from basp.dse.space import DesignSpace, Param

__all__ = [
    "DesignSpace",
    "Evaluation",
    "Evaluator",
    "Objective",
    "Param",
    "SearchResult",
    "bayesian",
    "chebyshev",
    "crowding_distance",
    "dominates",
    "evolutionary",
    "exhaustive",
    "hypervolume",
    "ideal_nadir",
    "nondominated_sort",
    "pareto_front",
    "random_search",
    "reference_point",
    "search",
]
