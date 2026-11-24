"""Streaming dataflow: one overlapped pipeline from storage to the MAC array.

Weights that are not resident on the executing device flow from their holder
through the link and the decoder straight into the arithmetic units, with
ready/valid backpressure between stages, and are never stored at the
destination:

    T_stream = max(T_read, T_link, T_decode, T_compute)

instead of the store-and-forward sum.  Requires a device whose fused pipeline
accepts the stored format (``stream_ingest`` capability).
"""

from __future__ import annotations

from basp.core.graph import OpNode
from basp.dataflow.base import Choice, DataflowMode, ShardChoice
from basp.dataflow.context import PlanContext


def candidates(op: OpNode, ctx: PlanContext) -> list[Choice]:
    out: list[Choice] = []
    for device in ctx.activation_devices(op):
        if not ctx.topology.devices[device].caps.stream_ingest:
            continue
        missing = [w for w in op.weights if ctx.st.avail(w, device) is None]
        if not missing:
            continue
        if not all(ctx.fused(w, device, device) for w in op.weights if w not in missing):
            continue
        remote = []
        for w in missing:
            src = ctx.best_source(w, device, fused=True)
            if src is None:
                break
            remote.append((w, src))
        else:
            shard = ShardChoice(device, op.weights, float(op.macs), 1.0, tuple(remote))
            out.append(Choice(DataflowMode.STREAMING, (shard,), stream=True))
    return out
