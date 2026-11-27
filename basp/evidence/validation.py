"""Values evidence may carry: finite, inside the metric's range, of the right type.

Adversarial inputs -- NaN, infinities, negative errors, booleans posing as numbers -- are
rejected here, before any comparison with a threshold can be fooled by them (``NaN > x`` is
false, so an unchecked NaN error passes every "error <= tolerance" test).
"""

from __future__ import annotations

import math
import numbers
from typing import Any

from .schema import METRICS


def value_problems(metric: str, value: Any) -> list[str]:
    """Why ``value`` is not a valid ``metric`` value (empty: it is)."""
    spec = METRICS.get(metric)
    if spec is None:
        return [f"unknown metric {metric!r}"]
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return [f"{metric}: {value!r} is not a number"]
    value = float(value)
    if not math.isfinite(value):
        return [f"{metric}: {value} is not finite"]
    if spec.minimum is not None and value < spec.minimum:
        return [f"{metric}: {value:g} is below the minimum {spec.minimum:g}"]
    if spec.maximum is not None and value > spec.maximum:
        return [f"{metric}: {value:g} is above the maximum {spec.maximum:g}"]
    return []


def valid_value(metric: str, value: Any) -> bool:
    return not value_problems(metric, value)
