"""Hardware-independent dataflow intermediate representation.

A :class:`Graph` is a tensor dependency graph: operations read activation and
weight tensors and produce new tensors.  Dependencies are implied by tensor
production/consumption plus optional explicit control dependencies.  The graph
may be extended incrementally, which is how data-dependent MoE routing is
represented: expert operations are added once the router output is known.
"""

from __future__ import annotations

import enum
import heapq
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from basp.core.tensor import TensorSpec


class OpCode(enum.IntEnum):
    MATVEC = 1
    MATMUL = 2
    DECODE = 3
    EXPERT = 4
    REDUCE = 5
    RMS_NORM = 6
    ATTENTION = 7
    ROPE = 8
    KV_APPEND = 9
    ROUTE = 10
    ACT_MUL = 11
    ADD = 12
    TRUNK = 13  # fused normalisation + attention + routing block


#: Implementation priority of each operation (section 13).
OP_PRIORITY: Mapping[OpCode, str] = {
    OpCode.MATVEC: "P0",
    OpCode.MATMUL: "P0",
    OpCode.DECODE: "P0",
    OpCode.EXPERT: "P0",
    OpCode.REDUCE: "P0",
    OpCode.ACT_MUL: "P0",
    OpCode.ADD: "P0",
    OpCode.RMS_NORM: "P1",
    OpCode.ATTENTION: "P1",
    OpCode.ROPE: "P1",
    OpCode.KV_APPEND: "P1",
    OpCode.ROUTE: "P1",
    OpCode.TRUNK: "P1",
}


class GraphError(ValueError):
    """Raised for malformed graphs (cycles, dangling tensors, double writers)."""


@dataclass(frozen=True)
class OpNode:
    """One operation of the dataflow IR.

    ``macs`` counts multiply-accumulates for the whole operation (all batch
    rows).  ``weight_fraction`` is the share of each listed weight tensor that
    the operation reads (an unfused expert sub-operation reads one third of the
    expert blob).  ``output_device`` names the device whose consumer needs the
    outputs, when known at compile time; plans then push results there.
    """

    op_id: int
    opcode: OpCode
    name: str
    inputs: tuple[int, ...] = ()
    weights: tuple[int, ...] = ()
    outputs: tuple[int, ...] = ()
    macs: int = 0
    batch: int = 1
    weight_fraction: float = 1.0
    extra_mem_bytes: int = 0
    pinned_device: int | None = None
    output_device: int | None = None
    partitionable: bool = False
    control_deps: tuple[int, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)


class Graph:
    def __init__(self) -> None:
        self.tensors: dict[int, TensorSpec] = {}
        self.ops: dict[int, OpNode] = {}
        self._producer: dict[int, int] = {}
        self._consumers: dict[int, list[int]] = defaultdict(list)

    # -- construction -------------------------------------------------------------
    def add_tensor(self, spec: TensorSpec) -> TensorSpec:
        known = self.tensors.get(spec.tensor_id)
        if known is not None and known != spec:
            raise GraphError(f"tensor {spec.tensor_id} redefined")
        self.tensors[spec.tensor_id] = spec
        return spec

    def add_op(self, op: OpNode) -> OpNode:
        if op.op_id in self.ops:
            raise GraphError(f"duplicate op id {op.op_id}")
        for tid in (*op.inputs, *op.weights, *op.outputs):
            if tid not in self.tensors:
                raise GraphError(f"op {op.name!r} references unknown tensor {tid}")
        if set(op.outputs) & (set(op.inputs) | set(op.weights)):
            raise GraphError(f"op {op.name!r} reads one of its own outputs")
        if len(set(op.outputs)) != len(op.outputs):
            raise GraphError(f"op {op.name!r} lists an output twice")
        for tid in op.weights:
            if not self.tensors[tid].is_weight:
                raise GraphError(f"op {op.name!r}: tensor {tid} is not a weight")
        for tid in op.outputs:
            if tid in self._producer:
                raise GraphError(f"tensor {tid} has two producers")
            if self.tensors[tid].policy.immutable:
                raise GraphError(f"op {op.name!r} writes immutable tensor {tid}")
        for dep in op.control_deps:
            if dep == op.op_id:
                raise GraphError(f"op {op.name!r} depends on itself")
        self.ops[op.op_id] = op
        for tid in op.outputs:
            self._producer[tid] = op.op_id
        for tid in dict.fromkeys(op.inputs):
            self._consumers[tid].append(op.op_id)
        return op

    # -- queries ------------------------------------------------------------------
    def producer(self, tensor_id: int) -> int | None:
        return self._producer.get(tensor_id)

    def consumers(self, tensor_id: int) -> tuple[int, ...]:
        return tuple(self._consumers.get(tensor_id, ()))

    def deps(self, op_id: int) -> tuple[int, ...]:
        """Operations that must complete before ``op_id`` may start."""
        op = self.ops[op_id]
        found = {self._producer[t] for t in op.inputs if t in self._producer}
        found.update(op.control_deps)
        return tuple(sorted(found))

    def dependents(self, op_id: int) -> tuple[int, ...]:
        out: set[int] = set()
        for tid in self.ops[op_id].outputs:
            out.update(self._consumers.get(tid, ()))
        out.update(o.op_id for o in self.ops.values() if op_id in o.control_deps)
        return tuple(sorted(out))

    def external_inputs(self) -> tuple[int, ...]:
        """Non-weight tensors that are read but produced by no operation."""
        read = {t for op in self.ops.values() for t in op.inputs}
        return tuple(sorted(t for t in read if t not in self._producer))

    def sinks(self, among: Iterable[int] | None = None) -> tuple[int, ...]:
        ids = set(self.ops if among is None else among)
        return tuple(sorted(i for i in ids if not (set(self.dependents(i)) & ids)))

    def topo_order(self, among: Iterable[int] | None = None) -> list[int]:
        """Deterministic topological order (ties broken by ascending op id).

        With ``among`` the order covers only those operations; dependencies on
        operations outside the set are treated as already satisfied.
        """
        ids = set(self.ops if among is None else among)
        unknown = ids - set(self.ops)
        if unknown:
            raise GraphError(f"unknown ops {sorted(unknown)}")
        indeg: dict[int, int] = {}
        users: dict[int, list[int]] = defaultdict(list)
        for i in ids:
            deps = [d for d in self.deps(i) if d in ids]
            indeg[i] = len(deps)
            for d in deps:
                users[d].append(i)
        heap = [i for i, n in indeg.items() if n == 0]
        heapq.heapify(heap)
        order: list[int] = []
        while heap:
            i = heapq.heappop(heap)
            order.append(i)
            for u in users[i]:
                indeg[u] -= 1
                if indeg[u] == 0:
                    heapq.heappush(heap, u)
        if len(order) != len(ids):
            stuck = sorted(i for i in ids if indeg[i] > 0)
            raise GraphError(f"dependency cycle among ops {stuck}")
        return order

    def validate(self) -> None:
        """Check global consistency: known control deps and no cycles."""
        for op in self.ops.values():
            for dep in op.control_deps:
                if dep not in self.ops:
                    raise GraphError(f"op {op.name!r} has unknown control dependency {dep}")
        self.topo_order()

    def total_macs(self) -> int:
        return sum(op.macs for op in self.ops.values())
