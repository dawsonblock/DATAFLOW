"""Numeric execution of plans: values computed alongside the timing simulation.

The payload is driven by the virtual runtime.  Every operation is computed when
the simulator reports it complete, in the order the simulated system finished
it, from the values of the tensors it depends on -- so a dependency error in
scheduling or command emission surfaces as a missing input here.  Execution
follows the *plan*: sharded operations are computed shard by shard from row
slices of the weights, and output-stationary plans accumulate device-local
partial sums that the reducer then combines.
"""

from __future__ import annotations

import numpy as np

from basp.compiler.lower import ModelProgram
from basp.core.graph import OpNode
from basp.dataflow.base import ResolvedPlan
from basp.kernels import matmul_int4, reference, run_expert, silu_fp32, split_expert
from basp.workloads.tiny import TinyModel
from basp.workloads.trace import RoutingStep


class NumericError(RuntimeError):
    pass


class NumericPayload:
    def __init__(self, model: TinyModel, program: ModelProgram, x0: np.ndarray, *,
                 tile_rows: int = 64) -> None:  # fmt: skip
        if x0.shape != (program.batch, model.spec.hidden):
            raise ValueError(f"x0 must have shape {(program.batch, model.spec.hidden)}")
        self.model = model
        self.program = program
        self.tile_rows = tile_rows
        self._next_input = np.asarray(x0, dtype=np.float32)
        self.inputs: list[np.ndarray] = []
        self.values: dict[int, np.ndarray] = {}
        self.partials: dict[tuple, np.ndarray] = {}
        self._accumulated: dict[int, list] = {}
        self._producer_attrs: dict[int, dict] = {}
        self.logits: list[np.ndarray] = []
        self.tokens: list[np.ndarray] = []
        self.routing: list[list[RoutingStep]] = []
        self.executed: list[int] = []
        self._head_rows: dict[int, tuple[int, int]] = {}
        start = 0
        for tensor in program.head_w:
            n_rows = program.registry.spec(tensor).shape[0]
            self._head_rows[tensor] = (start, start + n_rows)
            start += n_rows

    # -- runtime hooks ------------------------------------------------------------
    def begin_step(self, step: int, x: int) -> None:
        self.values = {x: self._next_input}
        self.inputs.append(self._next_input)
        self.partials = {}
        self._accumulated = {}
        self._producer_attrs = {}
        self.routing.append([])

    def route(self, step: int, layer: int, logits: int) -> RoutingStep:
        idx, coeffs = reference.route_topk(self._get(logits, "router"), self.model.spec.top_k)
        decision = RoutingStep(
            tuple(tuple(int(e) for e in row) for row in idx),
            tuple(tuple(float(c) for c in row) for row in coeffs),
        )
        self.routing[-1].append(decision)
        return decision

    def end_step(self, step: int) -> None:
        if len(self.logits) != step + 1:
            raise NumericError(f"step {step} finished without producing logits")
        tokens = np.argmax(self.logits[-1], axis=1)
        self.tokens.append(tokens)
        self._next_input = self.model.embed[tokens].astype(np.float32)

    def execute(self, op: OpNode, plan: ResolvedPlan) -> None:
        self.executed.append(op.op_id)
        handler = getattr(self, f"_op_{op.opcode.name.lower()}", None)
        if handler is None:
            raise NumericError(f"no numeric implementation for {op.opcode.name}")
        handler(op, plan)
        for tensor in op.outputs:
            self._producer_attrs[tensor] = dict(op.attrs)

    # -- operations ---------------------------------------------------------------
    def _get(self, tensor: int, who: str) -> np.ndarray:
        try:
            return self.values[tensor]
        except KeyError:
            raise NumericError(f"{who}: input tensor {tensor} has not been produced") from None

    def _norm(self, x: np.ndarray, layer: int) -> np.ndarray:
        return reference.rms_norm(x, self.model.norm[layer]).astype(np.float32)

    def _logits(self, h: np.ndarray, layer: int) -> np.ndarray:
        return (h @ self.model.router[layer].T).astype(np.float32)

    def _op_trunk(self, op: OpNode, plan: ResolvedPlan) -> None:
        layer = op.attrs["layer"]
        h = self._norm(self._get(op.inputs[0], op.name), layer)
        h_id, logits_id = op.outputs[0], op.outputs[-1]
        self.values[h_id] = h
        self.values[logits_id] = self._logits(h, layer)

    def _op_rms_norm(self, op: OpNode, plan: ResolvedPlan) -> None:
        self.values[op.outputs[0]] = self._norm(self._get(op.inputs[0], op.name), op.attrs["layer"])

    def _op_route(self, op: OpNode, plan: ResolvedPlan) -> None:
        self.values[op.outputs[0]] = self._logits(self._get(op.inputs[0], op.name), op.attrs["layer"])

    def _expert(self, op: OpNode):
        return self.model.experts[op.attrs["layer"]][op.attrs["expert"]]

    def _fold(self, op: OpNode, key, y: np.ndarray) -> None:
        """Fold ``coeff * y`` into the device-local partial sum ``key``."""
        rows = list(op.attrs["rows"])
        partial = self.partials.setdefault(
            key, np.zeros((self.program.batch, self.model.spec.hidden), dtype=np.float32)
        )
        coeffs = np.asarray(op.attrs["coeffs"], dtype=np.float32)[:, None]
        partial[rows] += coeffs * y
        held = self._accumulated.setdefault(op.outputs[0], [])
        if key not in held:
            held.append(key)

    def _emit_expert_output(self, op: OpNode, plan: ResolvedPlan, y: np.ndarray) -> None:
        """Store an expert output, or fold it into the device-local partial sum."""
        shard = plan.shards[0]
        if shard.partial is None:
            self.values[op.outputs[0]] = y
        else:
            self._fold(op, shard.partial, y)

    def _op_expert(self, op: OpNode, plan: ResolvedPlan) -> None:
        h = self._get(op.inputs[0], op.name)[list(op.attrs["rows"])]
        if len(op.weights) == 1:
            y = run_expert(self._expert(op), h, tile_rows=self.tile_rows)
            self._emit_expert_output(op, plan, y)
            return
        # Tensor-parallel expert: every plan shard computes the partial outputs of the weight
        # shards it holds; the expert output is their sum.
        parts = dict(zip(op.weights, split_expert(self._expert(op), self.program.shard_bounds)))
        total = None
        covered = 0
        for executed, chosen in zip(plan.shards, plan.choice.shards):
            y = None
            for tensor in chosen.weights:
                piece = run_expert(parts[tensor], h, tile_rows=self.tile_rows)
                y = piece if y is None else y + piece
                covered += 1
            if executed.partial is not None:
                self._fold(op, executed.partial, y)
            else:
                total = y if total is None else total + y
        if covered != len(op.weights):
            raise NumericError(f"{op.name}: plan covers {covered} of {len(op.weights)} shards")
        if total is not None:
            if op.outputs[0] in self._accumulated:
                raise NumericError(f"{op.name}: output is both stored and accumulated")
            self.values[op.outputs[0]] = total

    def _op_matvec(self, op: OpNode, plan: ResolvedPlan) -> None:
        if op.attrs.get("head"):
            self._head(op, plan)
            return
        part = op.attrs["part"]
        expert = self._expert(op)
        x = self._get(op.inputs[0], op.name)
        if part in ("gate", "up"):
            x = x[list(op.attrs["rows"])]
        y = matmul_int4(getattr(expert, part), x, tile_rows=self.tile_rows)
        if part == "down":
            self._emit_expert_output(op, plan, y)
        else:
            self.values[op.outputs[0]] = y

    _op_matmul = _op_matvec

    def _op_act_mul(self, op: OpNode, plan: ResolvedPlan) -> None:
        gate, up = (self._get(t, op.name) for t in op.inputs)
        self.values[op.outputs[0]] = silu_fp32(gate) * up

    def _op_reduce(self, op: OpNode, plan: ResolvedPlan) -> None:
        resid = op.attrs["resid"]
        out = self._get(resid, op.name).astype(np.float32).copy()
        folded: set[tuple] = set()
        for tensor in op.inputs:
            if tensor == resid:
                continue
            keys = self._accumulated.get(tensor)
            if keys:
                for key in keys:
                    if key not in folded:
                        folded.add(key)
                        out += self.partials[key]
                continue
            attrs = self._producer_attrs.get(tensor)
            if attrs is None:
                raise NumericError(f"{op.name}: expert output {tensor} has not been produced")
            y = self._get(tensor, op.name)
            coeffs = np.asarray(attrs["coeffs"], dtype=np.float32)[:, None]
            out[list(attrs["rows"])] += coeffs * y
        self.values[op.outputs[0]] = out

    def _head(self, op: OpNode, plan: ResolvedPlan) -> None:
        x = self._get(op.inputs[0], op.name)
        logits = np.empty((self.program.batch, self.model.spec.vocab), dtype=np.float32)
        covered = 0
        for shard in plan.choice.shards:  # local and streamed weights alike
            for tensor in shard.weights:
                lo, hi = self._head_rows[tensor]
                block = self.model.head.row_slice(lo, hi)
                logits[:, lo:hi] = matmul_int4(block, x, tile_rows=self.tile_rows)
                covered += hi - lo
        if covered != self.model.spec.vocab:
            raise NumericError(f"{op.name}: plan covers {covered} of {self.model.spec.vocab} rows")
        self.values[op.outputs[0]] = logits
        self.logits.append(logits)
