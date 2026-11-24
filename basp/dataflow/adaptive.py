"""Adaptive dataflow: enumerate every valid execution path for an operation.

The enabled modes each contribute candidate choices; the scheduler estimates
the completion time of each and picks jointly across the dependent operations
of a batch.  This is where data movement becomes an explicit optimisation
decision instead of a fixed property of the hardware.
"""

from __future__ import annotations

from basp.core.graph import OpNode
from basp.dataflow import (
    activation_stationary,
    distributed,
    output_stationary,
    row_stationary,
    streaming,
    weight_stationary,
)
from basp.dataflow.base import Choice, DataflowMode, ShardChoice
from basp.dataflow.context import PlanContext

M = DataflowMode


def _dedupe(choices: list[Choice]) -> list[Choice]:
    seen: set = set()
    out = []
    for choice in choices:
        sig = choice.signature()
        if sig not in seen:
            seen.add(sig)
            out.append(choice)
    return out


def candidates(op: OpNode, ctx: PlanContext) -> tuple[list[Choice], bool]:
    """Candidate choices for ``op`` and whether the policy had to be relaxed.

    Running an operation where both its weights and activations already are
    involves no movement decision and is always permitted.  If the enabled
    modes yield no feasible path at all, every generator is consulted so the
    model can still execute; such fallbacks are reported.
    """
    policy = ctx.policy
    base_mode = M.WEIGHT_STATIONARY if policy.enabled(M.WEIGHT_STATIONARY) else M.ACTIVATION_STATIONARY
    if not op.weights:
        if op.pinned_device is not None:
            return [Choice(base_mode, (ShardChoice(op.pinned_device, (), float(op.macs)),))], False
        return [Choice(base_mode, follow_inputs=True)], False

    stationary = weight_stationary.candidates(op, ctx)
    found: list[Choice] = []
    if policy.enabled(M.WEIGHT_STATIONARY):
        found.extend(stationary)
    else:
        found.extend(
            Choice(base_mode, c.shards) for c in stationary
            if ctx.inputs_local(op, c.shards[0].device)
        )  # fmt: skip
    if policy.enabled(M.ROW_STATIONARY):
        pool = found if policy.enabled(M.WEIGHT_STATIONARY) else found + stationary
        upgraded = row_stationary.transform(pool, op, ctx, keep_others=False)
        upgraded_devices = {c.shards[0].device for c in upgraded}
        found = upgraded + [c for c in found if c.shards[0].device not in upgraded_devices]
    if policy.enabled(M.ACTIVATION_STATIONARY):
        found.extend(activation_stationary.candidates(op, ctx))
    if policy.enabled(M.STREAMING):
        found.extend(streaming.candidates(op, ctx))
    if policy.enabled(M.DISTRIBUTED):
        found.extend(distributed.candidates(op, ctx))

    relaxed = False
    if not found:
        relaxed = True
        found = (
            stationary
            + activation_stationary.candidates(op, ctx)
            + streaming.candidates(op, ctx)
            + distributed.candidates(op, ctx)
        )
    if policy.enabled(M.OUTPUT_STATIONARY):
        found = output_stationary.transform(found, op, ctx)
    return _dedupe(found), relaxed
