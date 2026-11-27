"""Operator fusion.

Fusing a chain of primitive operations into one kernel removes the kernel
start-up and intermediate-tensor traffic between them and lets the decoder,
MAC array and accumulators run as one pipeline.  A group is fusible when it is
internally well ordered, agrees on placement constraints and reads the same
weight tensors (each member may read a fraction of them).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

from basp.core.graph import OpCode, OpNode


class FusionError(ValueError):
    pass


def fuse(
    ops: Sequence[OpNode],
    *,
    op_id: int,
    opcode: OpCode,
    name: str,
    external_outputs: Collection[int],
    attrs: Mapping[str, Any] | None = None,
) -> OpNode:
    """Fuse ``ops`` (given in dependency order) into a single operation.

    ``external_outputs`` are the tensors still needed outside the group; every
    other tensor produced inside becomes internal to the fused kernel.
    """
    if not ops:
        raise FusionError("cannot fuse an empty group")
    produced: list[int] = []
    inputs: list[int] = []
    for op in ops:
        for tid in op.inputs:
            if tid not in produced and tid not in inputs:
                inputs.append(tid)
        produced.extend(op.outputs)
    position = {tid: i for i, op in enumerate(ops) for tid in op.outputs}
    for i, op in enumerate(ops):
        for tid in op.inputs:
            if tid in position and position[tid] >= i:
                raise FusionError(f"{op.name!r} consumes tensor {tid} before it is produced")
    missing = set(external_outputs) - set(produced)
    if missing:
        raise FusionError(f"external outputs {sorted(missing)} are not produced by the group")

    def agree(field: str):
        values = {getattr(op, field) for op in ops if getattr(op, field) is not None}
        if len(values) > 1:
            raise FusionError(f"group disagrees on {field}: {sorted(values)}")
        return values.pop() if values else None

    if len({op.batch for op in ops}) != 1:
        raise FusionError("group members have different batch sizes")
    fractions: dict[int, float] = {}
    for op in ops:
        for tid in op.weights:
            fractions[tid] = fractions.get(tid, 0.0) + op.weight_fraction
    shares = {round(f, 9) for f in fractions.values()}
    if len(shares) > 1:
        raise FusionError("group reads its weight tensors in unequal fractions")
    fraction = shares.pop() if shares else 1.0
    if fraction > 1.0 + 1e-9:
        raise FusionError("group reads more than the whole of a weight tensor")
    merged: dict[str, Any] = {}
    for op in ops:
        merged.update(op.attrs)
    merged.update(attrs or {})
    merged["fused_from"] = tuple(op.opcode for op in ops)
    return OpNode(
        op_id=op_id,
        opcode=opcode,
        name=name,
        inputs=tuple(inputs),
        weights=tuple(fractions),
        outputs=tuple(t for t in produced if t in external_outputs),
        macs=sum(op.macs for op in ops),
        batch=ops[0].batch,
        weight_fraction=min(fraction, 1.0),
        extra_mem_bytes=sum(op.extra_mem_bytes for op in ops),
        pinned_device=agree("pinned_device"),
        output_device=agree("output_device"),
        partitionable=False,
        attrs=merged,
    )
