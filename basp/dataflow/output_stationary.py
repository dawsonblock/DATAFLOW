"""Output-stationary dataflow: partial results stay where they accumulate.

Operations whose outputs feed a weighted reduction fold ``coeff * output`` into
a device-local partial sum instead of shipping each output separately.  Each
device then sends one partial per reduction, and a device co-located with the
reducer sends nothing.  With k experts on m devices the output traffic drops
from k tensors to at most m; completion time is unchanged because the reducer
needs every contribution anyway.
"""

from __future__ import annotations

from dataclasses import replace

from basp.core.graph import OpNode
from basp.dataflow.base import Choice
from basp.dataflow.context import PlanContext


def applies(op: OpNode) -> bool:
    return (
        op.attrs.get("reduce_group") is not None
        and not op.attrs.get("reducer")
        and bool(op.outputs)
    )


def transform(choices: list[Choice], op: OpNode, ctx: PlanContext) -> list[Choice]:
    if not applies(op):
        return choices
    return [
        replace(c, accumulate=True) if len(c.shards) == 1 and not c.follow_inputs else c
        for c in choices
    ]
