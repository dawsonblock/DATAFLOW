"""Row-stationary dataflow: maximise local operand reuse through SRAM tiles.

A weight-stationary plan on a device with enough SRAM keeps the decoded weight
tiles in the device-local tile cache.  A later invocation that finds its tiles
there skips the external memory channels entirely and becomes compute-bound.
Selecting tile shapes inside the SRAM (the loop-nest mapping) is the job of a
mapper such as Timeloop; this module models the reuse it buys across
invocations.
"""

from __future__ import annotations

from dataclasses import replace

from basp.core.graph import OpNode
from basp.dataflow.base import Choice, DataflowMode
from basp.dataflow.context import PlanContext


def fits(op: OpNode, device: int, ctx: PlanContext) -> bool:
    sram = ctx.topology.devices[device].sram_bytes
    if sram <= 0 or not op.weights:
        return False
    need = 0.0
    for w in op.weights:
        fmt = ctx.st.format_at(w, device)
        need += ctx.registry.nbytes(w, fmt) * op.weight_fraction
    return need <= sram


def transform(choices: list[Choice], op: OpNode, ctx: PlanContext, *, keep_others: bool) -> list[Choice]:
    """Upgrade co-located plans on SRAM-capable devices to row-stationary plans."""
    out: list[Choice] = []
    for choice in choices:
        local = len(choice.shards) == 1 and not choice.shards[0].remote and not choice.follow_inputs
        if local and fits(op, choice.shards[0].device, ctx):
            mode = choice.mode
            if mode == DataflowMode.WEIGHT_STATIONARY:
                mode = DataflowMode.ROW_STATIONARY
            out.append(replace(choice, use_sram=True, mode=mode))
        elif keep_others:
            out.append(choice)
    return out
