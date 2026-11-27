"""Activation-stationary dataflow: move data to the computation (path A).

The operation runs where its activations already are (and where its results are
consumed); missing weights are copied there first, store-and-forward.  With
``retain`` the copy stays as a replica so later uses pay nothing, which is what
makes this path win for weights that are reused many times on a fast processor.
"""

from __future__ import annotations

from basp.core.graph import OpNode
from basp.dataflow.base import Choice, DataflowMode, ShardChoice
from basp.dataflow.context import PlanContext


def candidates(op: OpNode, ctx: PlanContext) -> list[Choice]:
    out: list[Choice] = []
    cache = ctx.policy.cache
    for device in ctx.activation_devices(op):
        missing = [w for w in op.weights if ctx.st.avail(w, device) is None]
        if not missing:
            continue  # nothing to move: covered by the co-located plan
        if not all(ctx.consumable(w, device, device) for w in op.weights if w not in missing):
            continue
        if not ctx.topology.devices[device].caps.can_compute:
            continue
        remote = []
        for w in missing:
            src = ctx.best_source(w, device)
            if src is None:
                break
            remote.append((w, src))
        else:
            shard = ShardChoice(device, op.weights, float(op.macs), 1.0, tuple(remote))
            replicable = all(ctx.registry.spec(w).policy.replicable for w in missing)
            if cache.kind != "none" and device in cache.devices and replicable:
                out.append(Choice(DataflowMode.ACTIVATION_STATIONARY, (shard,), retain=True))
            if cache.kind != "lru" or device not in cache.devices or not replicable:
                out.append(Choice(DataflowMode.ACTIVATION_STATIONARY, (shard,)))
    return out
