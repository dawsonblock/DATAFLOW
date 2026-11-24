"""Weight-stationary dataflow: move computation to the data (path B).

The operation runs on a device that already holds every weight it needs.  Only
activations cross the interconnect, so traffic shrinks from the weight matrix
to a pair of activation vectors.  Best when weights are large, used once and
the interconnect is slow relative to local memory -- the MoE expert case.
"""

from __future__ import annotations

from basp.core.graph import OpNode
from basp.dataflow.base import Choice, DataflowMode, ShardChoice
from basp.dataflow.context import PlanContext


def candidates(op: OpNode, ctx: PlanContext) -> list[Choice]:
    return [
        Choice(DataflowMode.WEIGHT_STATIONARY, (ShardChoice(device, op.weights, float(op.macs)),))
        for device in ctx.devices_for(op)
        if ctx.weights_local(op, device)
    ]
