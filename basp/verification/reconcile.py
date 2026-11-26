"""Reconcile the simulator's lowered graph with the model's dimensions.

Two independent descriptions of the same work must agree:

* ``MoEModelSpec.layer_accounting`` / ``step_accounting`` -- arithmetic and parameter reads
  computed from the model's dimensions alone;
* the operations the compiler actually lowers (``ModelProgram``), whose multiply-accumulates
  and weight fractions are what the simulator charges.

And the tensors the program registers must store every parameter of the model exactly once
(tied embeddings: one table).  A disagreement means the simulator simulates a different model
than the one it names -- the defect class of audit findings S1 (shared experts) and S2
(embedding and final norm).
"""

from __future__ import annotations

from typing import Any

from basp.compiler.lower import ModelProgram
from basp.core.graph import Graph, OpCode, OpNode
from basp.workloads.trace import RoutingStep


def _role(op: OpNode) -> str:
    if op.opcode is OpCode.EMBED:
        return "embed"
    if op.attrs.get("final_norm"):
        return "final_norm"
    if op.attrs.get("head"):
        return "head"
    if op.opcode is OpCode.REDUCE:
        return "reduce"
    if op.opcode is OpCode.EXPERT or "expert" in op.attrs:
        return "expert"
    return "trunk"


def graph_accounting(program: ModelProgram, ops: list[OpNode]) -> dict[str, float]:
    """Multiply-accumulates and parameters read, by role, summed over ``ops``."""
    out: dict[str, float] = {}
    for op in ops:
        role = _role(op)
        out[f"{role}_macs"] = out.get(f"{role}_macs", 0) + op.macs
        read = sum(op.weight_fraction * program.registry.spec(w).n_weights for w in op.weights)
        out[f"{role}_params_read"] = out.get(f"{role}_params_read", 0.0) + read
    return out


def lower_step(program: ModelProgram, routing: list[RoutingStep], step: int = 0) -> list[OpNode]:
    """Every operation of one step (all layers), lowered as the virtual runtime does."""
    graph = Graph()
    x = program.input_tensor(graph, step)
    ops = list(program.embed(graph, step, x).ops)
    for layer in range(program.spec.n_layers):
        trunk = program.trunk(graph, step, layer, x)
        experts = program.experts(graph, step, layer, trunk.outputs["h"], routing[layer])
        reduce = program.reduce(graph, step, layer, experts.outputs, trunk.outputs["resid"],
                                trunk.outputs.get("shared"))  # fmt: skip
        ops += trunk.ops + experts.ops + reduce.ops
        x = reduce.outputs["x"]
    return ops + program.lm_head(graph, step, x).ops


def expected_step(program: ModelProgram, routing: list[RoutingStep], step: int = 0) -> dict[str, float]:
    """The same totals from the model's dimensions alone."""
    spec, batch = program.spec, program.batch
    context = program.context_len + step
    mixed = context + (batch + 1) // 2 if program.prompt else context
    out: dict[str, float] = {}
    for layer in range(spec.n_layers):
        selected = sum(len(row) for row in routing[layer].experts)
        unique = len(routing[layer].unique_experts())
        for key, value in spec.layer_accounting(batch=batch, mixed_context=mixed, selected=selected,
                                                unique=unique).items():  # fmt: skip
            out[key] = out.get(key, 0) + value
    out.update(spec.step_accounting(batch=batch, head_rows=1 if program.prompt else batch))
    return out


def reconcile(program: ModelProgram, routing: list[RoutingStep], *, rel: float = 1e-9) -> dict[str, Any]:
    """Compare the lowered step with the independent calculation, and stored parameters with the
    model's parameter count.  ``ok`` only if every quantity agrees."""
    got = graph_accounting(program, lower_step(program, routing))
    want = expected_step(program, routing)
    rows = []
    for key in sorted(set(got) | set(want)):
        a, b = float(got.get(key, 0.0)), float(want.get(key, 0.0))
        rows.append({"quantity": key, "graph": a, "expected": b,
                     "ok": abs(a - b) <= rel * max(abs(a), abs(b), 1.0)})  # fmt: skip
    tensors = program.weight_tensors()
    stored = sum(program.registry.spec(t).n_weights for t in tensors)
    storage = {"stored_params": stored, "model_params": program.spec.total_params,
               "tensors": len(tensors), "unique": len(set(tensors)) == len(tensors),
               "ok": stored == program.spec.total_params and len(set(tensors)) == len(tensors)}  # fmt: skip
    return {"model": program.spec.name, "quantities": rows, "storage": storage,
            "ok": storage["ok"] and all(r["ok"] for r in rows)}  # fmt: skip
