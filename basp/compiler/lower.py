"""Lowering: MoE model -> tensor dependency graph of primitive or fused operations.

A :class:`ModelProgram` registers the model's weight tensors in the logical
namespace and builds the operations of one transformer layer at a time, because
which experts run is only known once the router has produced its output.
Primitive operations are always generated first; with ``fused=True`` the fusion
pass merges each expert's chain (and each trunk block) into one operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from basp.compiler.fusion import fuse
from basp.core.graph import Graph, OpCode, OpNode
from basp.core.tensor import DType, TensorKind, TensorSpec, activation_bytes
from basp.kernels.expert import shard_bounds
from basp.memory.tensor_registry import TensorRegistry
from basp.workloads.model import MoEModelSpec
from basp.workloads.trace import RoutingStep


@dataclass
class LayerOps:
    """Operations of one batch plus the tensors later stages need."""

    ops: list[OpNode]
    outputs: dict[str, int] = field(default_factory=dict)


class ModelProgram:
    def __init__(
        self,
        spec: MoEModelSpec,
        registry: TensorRegistry,
        *,
        batch: int = 1,
        context_len: int = 0,
        max_new_tokens: int = 0,
        head_shards: int = 1,
        expert_shards: int = 1,
        fused: bool = True,
        prompt: bool = False,
    ) -> None:
        """``prompt=True``: the ``batch`` rows are the tokens of *one* prompt being processed
        together (prefill), not independent sequences: they share one KV cache, which holds
        ``context_len`` earlier tokens plus the prompt, and each row attends to the earlier
        tokens and to the prompt tokens before it."""
        if head_shards < 1 or expert_shards < 1:
            raise ValueError("head_shards and expert_shards must be >= 1")
        if expert_shards > 1 and not fused:
            raise ValueError("tensor-parallel experts need the fused expert operation")
        self.spec = spec
        self.registry = registry
        self.batch = batch
        self.context_len = context_len
        self.fused = fused
        self.prompt = prompt
        self.sequences = 1 if prompt else batch
        self.act_bytes = activation_bytes(batch * spec.hidden)
        formats = registry.formats
        fmt = spec.weight_format
        self.trunk_w: list[int] = []
        self.kv: list[int] = []
        self.expert_w: list[list[int]] = []  # first (or only) tensor of each expert
        self.expert_parts: list[list[tuple[int, ...]]] = []  # every tensor of each expert
        self.expert_of: dict[int, tuple[int, int]] = {}
        #: ranges of the intermediate dimension held by each tensor-parallel shard
        self.shard_bounds = shard_bounds(spec.expert_inter, expert_shards)
        max_ctx = context_len + (batch if prompt else 0) + max_new_tokens
        sequences = self.sequences
        for layer in range(spec.n_layers):
            trunk = registry.add(
                f"trunk.{layer}", TensorKind.WEIGHT, (spec.trunk_params,), DType.BYTES,
                formats.weight_bytes(spec.trunk_params, fmt), fmt=fmt, n_weights=spec.trunk_params,
            )  # fmt: skip
            self.trunk_w.append(trunk.tensor_id)
            if spec.n_heads:
                kv = registry.add(
                    f"kv.{layer}", TensorKind.KV_CACHE, (sequences, max(max_ctx, 1)), DType.FP16,
                    max(sequences * max_ctx * spec.kv_bytes_per_token_layer, 1),
                )  # fmt: skip
                self.kv.append(kv.tensor_id)
            row, parts_row = [], []
            for expert in range(spec.n_experts):
                parts = []
                for j, (lo, hi) in enumerate(self.shard_bounds):
                    n = 3 * spec.hidden * (hi - lo)
                    suffix = f".p{j}" if expert_shards > 1 else ""
                    t = registry.add(
                        f"expert.{layer}.{expert}{suffix}", TensorKind.EXPERT_WEIGHT, (n,),
                        DType.BYTES, formats.weight_bytes(n, fmt), fmt=fmt, n_weights=n,
                    )  # fmt: skip
                    parts.append(t.tensor_id)
                    self.expert_of[t.tensor_id] = (layer, expert)
                row.append(parts[0])
                parts_row.append(tuple(parts))
            self.expert_w.append(row)
            self.expert_parts.append(parts_row)
        self.head_w: list[int] = []
        if spec.vocab:
            rows = [spec.vocab // head_shards + (i < spec.vocab % head_shards) for i in range(head_shards)]
            for i, n_rows in enumerate(rows):
                n = n_rows * spec.hidden
                t = registry.add(
                    f"lm_head.{i}", TensorKind.WEIGHT, (n_rows, spec.hidden), DType.BYTES,
                    formats.weight_bytes(n, fmt), fmt=fmt, n_weights=n,
                )  # fmt: skip
                self.head_w.append(t.tensor_id)
        self.trunk_device: list[int] = [0] * spec.n_layers  # set by placement
        self.sample_device: int = 0
        self._next_op = 1
        self._probes: dict[tuple[int, int], OpNode] = {}
        self._probe_io: tuple[int, int] | None = None

    # -- small helpers ------------------------------------------------------------
    def _op_id(self) -> int:
        op_id = self._next_op
        self._next_op += 1
        return op_id

    def _activation(self, graph: Graph, name: str, rows: int | None = None, width: int | None = None,
                    kind: TensorKind = TensorKind.ACTIVATION) -> int:  # fmt: skip
        rows = self.batch if rows is None else rows
        width = self.spec.hidden if width is None else width
        spec = self.registry.add(name, kind, (rows, width), DType.FP16, activation_bytes(rows * width))
        graph.add_tensor(spec)
        return spec.tensor_id

    def _weight(self, graph: Graph, tensor_id: int) -> int:
        graph.add_tensor(self.registry.spec(tensor_id))
        return tensor_id

    def _emit(self, graph: Graph, ops: list[OpNode]) -> list[OpNode]:
        for op in ops:
            graph.add_op(op)
        return ops

    def layer_expert_tensors(self, layer: int) -> list[int]:
        """Every expert weight tensor of ``layer`` (all shards of all experts)."""
        return [t for parts in self.expert_parts[layer] for t in parts]

    def temporaries(self, graph: Graph) -> list[int]:
        return [t for t, spec in graph.tensors.items() if spec.policy.temporary]

    # -- graph construction -------------------------------------------------------
    def input_tensor(self, graph: Graph, step: int) -> int:
        """Embedded input of a decoding step (one row per sequence in the batch)."""
        return self._activation(graph, f"s{step}.embed")

    def trunk(self, graph: Graph, step: int, layer: int, x: int) -> LayerOps:
        """Normalisation, attention and routing of one layer.

        Outputs: ``h`` (normalised hidden state fed to the experts) and ``resid``
        (the residual stream the expert mixture is added to).
        """
        s = self.spec
        device = self.trunk_device[layer]
        pre = f"s{step}.l{layer}"
        w = self._weight(graph, self.trunk_w[layer])
        ops: list[OpNode] = []
        resid = x
        tensors: list[TensorSpec] = []

        def temp(name: str) -> int:
            spec = self.registry.add(f"{pre}.{name}", TensorKind.ACTIVATION, (self.batch, s.hidden),
                                     DType.FP16, self.act_bytes)  # fmt: skip
            tensors.append(spec)
            return spec.tensor_id

        if s.n_heads:
            context = self.context_len + step
            if self.prompt:
                # one sequence: rows attend on average to half the prompt on top of the context,
                # and the cache (context plus prompt) is read once, not once per row
                mixed, kv_read = context + (self.batch + 1) // 2, context + self.batch
            else:
                mixed, kv_read = context, self.batch * context
            normed = temp("attn_norm")
            ops.append(OpNode(self._op_id(), OpCode.RMS_NORM, f"{pre}.norm1", (x,), (w,), (normed,),
                              macs=self.batch * s.hidden, batch=self.batch,
                              weight_fraction=s.hidden / s.trunk_params, pinned_device=device))  # fmt: skip
            resid = temp("resid")
            ops.append(OpNode(self._op_id(), OpCode.ATTENTION, f"{pre}.attn", (normed, x), (w,), (resid,),
                              macs=self.batch * s.attention_macs(mixed), batch=self.batch,
                              weight_fraction=s.attn_params / s.trunk_params,
                              extra_mem_bytes=kv_read * s.kv_bytes_per_token_layer,
                              pinned_device=device, attrs={"context": context}))  # fmt: skip
        h = temp("h")
        norm_share = (s.hidden if s.n_heads else s.norm_params) / s.trunk_params
        ops.append(OpNode(self._op_id(), OpCode.RMS_NORM, f"{pre}.norm2", (resid,), (w,), (h,),
                          macs=self.batch * s.hidden, batch=self.batch, weight_fraction=norm_share,
                          pinned_device=device, attrs={"layer": layer}))  # fmt: skip
        logits = self._temp_spec(tensors, f"{pre}.route", s.n_experts)
        route_share = s.router_params / s.trunk_params
        if not s.n_heads:
            route_share = 1.0 - norm_share  # attention-free trunk: router takes the rest
        ops.append(OpNode(self._op_id(), OpCode.ROUTE, f"{pre}.route", (h,), (w,), (logits,),
                          macs=self.batch * s.router_params, batch=self.batch,
                          weight_fraction=route_share, pinned_device=device,
                          attrs={"layer": layer, "step": step}))  # fmt: skip
        keep = {h, logits} | ({resid} if resid != x else set())
        if self.fused:
            fused = fuse(ops, op_id=self._op_id(), opcode=OpCode.TRUNK, name=f"{pre}.trunk",
                         external_outputs=keep, attrs={"layer": layer, "step": step})  # fmt: skip
            for spec in tensors:
                if spec.tensor_id in keep:
                    graph.add_tensor(spec)
                else:
                    self.registry.unregister(spec.tensor_id)
            ops = [fused]
        else:
            for spec in tensors:
                graph.add_tensor(spec)
        self._emit(graph, ops)
        return LayerOps(ops, {"h": h, "resid": resid, "logits": logits})

    def _temp_spec(self, tensors: list[TensorSpec], name: str, width: int) -> int:
        spec = self.registry.add(name, TensorKind.ACTIVATION, (self.batch, width), DType.FP16,
                                 activation_bytes(self.batch * width))  # fmt: skip
        tensors.append(spec)
        return spec.tensor_id

    def experts(self, graph: Graph, step: int, layer: int, h: int, routing: RoutingStep) -> LayerOps:
        """One operation (or primitive chain) per distinct selected expert."""
        s = self.spec
        home = self.trunk_device[layer]
        group = (step, layer)
        ops: list[OpNode] = []
        outputs: dict[str, int] = {}
        for expert in routing.unique_experts():
            rows = routing.rows_for(expert)
            coeffs = tuple(
                routing.coeffs[r][routing.experts[r].index(expert)] for r in rows
            )
            n = len(rows)
            pre = f"s{step}.l{layer}.e{expert}"
            parts = tuple(self._weight(graph, t) for t in self.expert_parts[layer][expert])
            w = parts[0]
            attrs = {"layer": layer, "expert": expert, "rows": rows, "coeffs": coeffs}
            # Only the operation whose output feeds the reduction belongs to the reduce group.
            final = {**attrs, "reduce_group": group}
            specs: list[TensorSpec] = []

            def temp(name: str, width: int, pre=pre, n=n, specs=specs) -> int:
                spec = self.registry.add(f"{pre}.{name}", TensorKind.INTERMEDIATE, (n, width),
                                         DType.FP16, activation_bytes(n * width))  # fmt: skip
                specs.append(spec)
                return spec.tensor_id

            gate, up = temp("gate", s.expert_inter), temp("up", s.expert_inter)
            mixed = temp("act", s.expert_inter)
            y = self._activation(graph, f"{pre}.out", rows=n)
            mv = OpCode.MATVEC if n == 1 else OpCode.MATMUL
            third = n * s.hidden * s.expert_inter
            chain = [
                OpNode(self._op_id(), mv, f"{pre}.gate", (h,), (w,), (gate,), macs=third, batch=n,
                       weight_fraction=1 / 3, attrs={**attrs, "part": "gate"}),
                OpNode(self._op_id(), mv, f"{pre}.up", (h,), (w,), (up,), macs=third, batch=n,
                       weight_fraction=1 / 3, attrs={**attrs, "part": "up"}),
                OpNode(self._op_id(), OpCode.ACT_MUL, f"{pre}.act", (gate, up), (), (mixed,),
                       macs=n * s.expert_inter, batch=n, attrs=attrs),
                OpNode(self._op_id(), mv, f"{pre}.down", (mixed,), (w,), (y,), macs=third, batch=n,
                       weight_fraction=1 / 3, output_device=home, attrs={**final, "part": "down"}),
            ]  # fmt: skip
            if self.fused:
                for spec in specs:
                    self.registry.unregister(spec.tensor_id)
                fused = fuse(chain, op_id=self._op_id(), opcode=OpCode.EXPERT, name=pre,
                             external_outputs={y}, attrs=final)  # fmt: skip
                if len(parts) > 1:
                    # Tensor-parallel: every shard computes a complete partial output from its
                    # slice of the intermediate dimension; the outputs add up.
                    fused = replace(fused, weights=parts, partitionable=True,
                                    attrs={**fused.attrs, "partial_outputs": True})  # fmt: skip
                chain = [fused]
            else:
                for spec in specs:
                    graph.add_tensor(spec)
            ops.extend(self._emit(graph, chain))
            outputs[f"e{expert}"] = y
        return LayerOps(ops, outputs)

    def reduce(self, graph: Graph, step: int, layer: int, expert_outputs: dict[str, int],
               resid: int) -> LayerOps:  # fmt: skip
        """Weighted reduction of the expert outputs plus the residual connection."""
        s = self.spec
        device = self.trunk_device[layer]
        y = self._activation(graph, f"s{step}.l{layer}.out")
        op = OpNode(
            self._op_id(), OpCode.REDUCE, f"s{step}.l{layer}.reduce",
            inputs=(*expert_outputs.values(), resid), outputs=(y,),
            macs=self.batch * s.hidden * (min(s.top_k, len(expert_outputs)) + 1), batch=self.batch,
            pinned_device=device,
            attrs={"layer": layer, "step": step, "reduce_group": (step, layer), "reducer": True,
                   "resid": resid},
        )  # fmt: skip
        return LayerOps(self._emit(graph, [op]), {"x": y})

    def lm_head(self, graph: Graph, step: int, x: int) -> LayerOps:
        """Output projection; row shards may run on different devices.

        For a prompt only the last token's logits are computed (as llama.cpp does when
        processing a prompt): one row, not one per prompt token.
        """
        s = self.spec
        if not self.head_w:
            return LayerOps([], {"logits": x})
        rows = 1 if self.prompt else self.batch
        logits = self._activation(graph, f"s{step}.logits", rows=rows, width=s.vocab)
        weights = tuple(self._weight(graph, w) for w in self.head_w)
        op = OpNode(
            self._op_id(), OpCode.MATVEC if rows == 1 else OpCode.MATMUL, f"s{step}.lm_head",
            inputs=(x,), weights=weights, outputs=(logits,), macs=rows * s.head_params,
            batch=rows, output_device=self.sample_device, partitionable=len(weights) > 1,
            attrs={"step": step, "head": True},
        )  # fmt: skip
        return LayerOps(self._emit(graph, [op]), {"logits": logits})

    def probe_expert(self, layer: int, expert: int) -> OpNode:
        """Template of a single-row expert operation, used for what-if cost estimates."""
        key = (layer, expert)
        if key not in self._probes:
            s = self.spec
            if self._probe_io is None:
                # one shared pair of single-row activations, so estimates include their movement
                nbytes = activation_bytes(s.hidden)
                self._probe_io = tuple(
                    self.registry.add(f"probe.{name}", TensorKind.ACTIVATION, (1, s.hidden),
                                      DType.FP16, nbytes).tensor_id
                    for name in ("x", "y")
                )  # fmt: skip
            x, y = self._probe_io
            parts = self.expert_parts[layer][expert]
            attrs = {"layer": layer, "expert": expert, "probe": True}
            if len(parts) > 1:
                attrs["partial_outputs"] = True
            self._probes[key] = OpNode(
                0, OpCode.EXPERT, f"probe.l{layer}.e{expert}", inputs=(x,), weights=parts,
                outputs=(y,), macs=s.expert_params, output_device=self.trunk_device[layer],
                partitionable=len(parts) > 1, attrs=attrs,
            )  # fmt: skip
        return self._probes[key]
