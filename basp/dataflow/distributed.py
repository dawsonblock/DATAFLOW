"""Distributed dataflow: partition one operation across several devices.

A partitionable operation (a row-sharded matrix product such as the output
projection) lists one weight tensor per shard.  Each shard runs on a device
holding it; the input activation is broadcast, the partial outputs are gathered
at ``output_device``.  Independent operations (the experts of one layer) are
spread across devices by the scheduler itself and need no special plan.
"""

from __future__ import annotations

from basp.core.graph import OpNode
from basp.dataflow.base import Choice, DataflowMode, ShardChoice
from basp.dataflow.context import PlanContext


def candidates(op: OpNode, ctx: PlanContext) -> list[Choice]:
    if not op.partitionable or len(op.weights) < 2 or op.output_device is None:
        return []
    total = sum(ctx.registry.spec(w).n_weights for w in op.weights) or 1
    # A device holding every shard runs the whole operation as a co-located plan; the
    # distributed plan is the alternative that spreads the shards over the other holders.
    complete = {d for d in ctx.devices_for(op) if ctx.weights_local(op, d)}
    load: dict[int, float] = {}
    assigned: dict[int, list[int]] = {}
    for w in op.weights:
        holders = [
            device for device, _ in ctx.st.sources(w)
            if device not in complete and ctx.allowed(device, op)
            and ctx.consumable(w, device, device)
        ]  # fmt: skip
        if not holders:
            return []
        share = ctx.registry.spec(w).n_weights / total
        device = min(
            holders,
            key=lambda d: (
                (load.get(d, 0.0) + share) * op.macs / ctx.topology.devices[d].compute_rate,
                d,
            ),
        )
        load[device] = load.get(device, 0.0) + share
        assigned.setdefault(device, []).append(w)
    if len(assigned) < 2:
        return []
    shards = tuple(
        ShardChoice(device, tuple(ws), op.macs * load[device], load[device])
        for device, ws in sorted(assigned.items())
    )
    return [Choice(DataflowMode.DISTRIBUTED, shards)]
