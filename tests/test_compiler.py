"""Model shapes, lowering, operator fusion, placement and command emission."""

from __future__ import annotations

import pytest

from basp.compiler.fusion import FusionError, fuse
from basp.compiler.lower import ModelProgram
from basp.core.graph import Graph, OpCode, OpNode
from basp.core.tensor import CompressionFormat, TensorKind
from basp.dataflow.base import ALL_MODES, CachePolicy, DataflowMode, ExecutionPolicy
from basp.experiments.harness import build
from basp.memory.residency import ResidencyMap
from basp.memory.tensor_registry import TensorRegistry
from basp.scheduler import placement
from basp.streams.commands import MAX_DEPS, MAX_TENSORS, validate
from basp.streams.protocol import Opcode
from basp.workloads.model import QWEN3_30B_A3B, tiny_spec
from basp.workloads.trace import RoutingStep, synth_trace
from tests.conftest import GPU, HOST, MOD0, MOD1, make_topology


def test_qwen3_shape_reproduces_published_parameter_counts():
    spec = QWEN3_30B_A3B
    assert spec.expert_params == 3 * 2048 * 768 == 4_718_592
    assert spec.attn_params == 18_874_368
    assert spec.total_params == pytest.approx(30.5e9, rel=0.005)  # "30.5B"
    assert spec.active_params == pytest.approx(3.3e9, rel=0.02)  # "3.3B activated"
    assert spec.kv_bytes_per_token_layer * spec.n_layers == 98_304
    with pytest.raises(ValueError):
        tiny_spec(top_k=99)


def program(fused: bool, batch: int = 1, **spec_kw):
    registry = TensorRegistry()
    spec = tiny_spec(**spec_kw)
    return ModelProgram(spec, registry, batch=batch, fused=fused, head_shards=2), spec


def one_layer(prog: ModelProgram, routing: RoutingStep) -> tuple[Graph, list[OpNode]]:
    graph = Graph()
    x = prog.input_tensor(graph, 0)
    trunk = prog.trunk(graph, 0, 0, x)
    experts = prog.experts(graph, 0, 0, trunk.outputs["h"], routing)
    reduce = prog.reduce(graph, 0, 0, experts.outputs, trunk.outputs["resid"])
    head = prog.lm_head(graph, 0, reduce.outputs["x"])
    graph.validate()
    return graph, trunk.ops + experts.ops + reduce.ops + head.ops


ROUTING = RoutingStep(((1, 4), (4, 6)), ((0.7, 0.3), (0.6, 0.4)))


def test_fusion_preserves_work_and_shrinks_the_graph():
    fused_prog, spec = program(True, batch=2)
    prim_prog, _ = program(False, batch=2)
    _, fused = one_layer(fused_prog, ROUTING)
    graph, prims = one_layer(prim_prog, ROUTING)
    assert sum(op.macs for op in fused) == sum(op.macs for op in prims)
    # one trunk, three distinct experts (1, 4, 6), one reduce, one head
    assert [op.opcode for op in fused] == [OpCode.TRUNK] + [OpCode.EXPERT] * 3 + [OpCode.REDUCE,
                                                                                  OpCode.MATMUL]  # fmt: skip
    assert len(prims) == 2 + 3 * 4 + 2
    expert4 = next(op for op in fused if op.attrs.get("expert") == 4)
    assert expert4.batch == 2 and expert4.attrs["rows"] == (0, 1)  # both rows selected it
    assert expert4.attrs["coeffs"] == (0.3, 0.6)
    assert expert4.macs == 2 * spec.expert_params + 2 * spec.expert_inter
    assert expert4.weight_fraction == pytest.approx(1.0)
    assert expert4.attrs["fused_from"] == (OpCode.MATMUL, OpCode.MATMUL, OpCode.ACT_MUL, OpCode.MATMUL)
    # unfused: gate and up are independent, the activation joins them, down follows
    chain = [op for op in prims if op.attrs.get("expert") == 4]
    assert graph.deps(chain[2].op_id) == (chain[0].op_id, chain[1].op_id)
    assert graph.deps(chain[3].op_id) == (chain[2].op_id,)
    assert [op.attrs.get("reduce_group") for op in chain] == [None, None, None, (0, 0)]
    head = fused[-1]
    assert head.partitionable and len(head.weights) == 2


def test_fusion_rejects_illegal_groups():
    registry = TensorRegistry()
    prog = ModelProgram(tiny_spec(), registry, fused=False)
    graph = Graph()
    x = prog.input_tensor(graph, 0)
    trunk = prog.trunk(graph, 0, 0, x)
    chain = prog.experts(graph, 0, 0, trunk.outputs["h"], RoutingStep(((0, 1),), ((0.5, 0.5),))).ops
    gate, up, act, down = chain[:4]
    ok = fuse([gate, up, act, down], op_id=900, opcode=OpCode.EXPERT, name="e",
              external_outputs=set(down.outputs))  # fmt: skip
    assert ok.inputs == gate.inputs and ok.outputs == down.outputs
    assert ok.output_device == down.output_device and ok.weights == gate.weights
    with pytest.raises(FusionError, match="before it is produced"):
        fuse([act, gate, up, down], op_id=901, opcode=OpCode.EXPERT, name="e",
             external_outputs=set(down.outputs))  # fmt: skip
    with pytest.raises(FusionError, match="not produced by the group"):
        fuse([gate, up], op_id=902, opcode=OpCode.EXPERT, name="e", external_outputs={12345})
    with pytest.raises(FusionError, match="empty"):
        fuse([], op_id=903, opcode=OpCode.EXPERT, name="e", external_outputs=set())
    other = chain[4]  # gate of a different expert: different weight tensor, different fraction mix
    with pytest.raises(FusionError, match="unequal fractions"):
        fuse([gate, up, other], op_id=904, opcode=OpCode.EXPERT, name="e",
             external_outputs=set(up.outputs))  # fmt: skip
    pinned = [OpNode(1, OpCode.ADD, "a", pinned_device=0), OpNode(2, OpCode.ADD, "b", pinned_device=1)]
    with pytest.raises(FusionError, match="pinned_device"):
        fuse(pinned, op_id=905, opcode=OpCode.ADD, name="p", external_outputs=set())


# -- placement -----------------------------------------------------------------------------
def placed(policy_fn, spec=None, **topo):
    topology = make_topology(**topo)
    registry = TensorRegistry()
    prog = ModelProgram(spec or tiny_spec(), registry, head_shards=2)
    residency = ResidencyMap(registry, topology)
    report = policy_fn(prog, residency)
    residency.check_invariants()
    return prog, residency, report


def test_static_split_offloads_the_layers_that_fit():
    spec = tiny_spec(hidden=256, expert_inter=1024, n_layers=6, n_experts=16)
    layer_bytes = 16 * spec.expert_params * 4.5 / 8
    prog, residency, report = placed(
        lambda p, r: placement.place_static_split(p, r, GPU, HOST), spec,
        gpu_mem=int(3.6 * layer_bytes * 16 / 15),
    )  # fmt: skip
    assert prog.trunk_device == [GPU, GPU, GPU, HOST, HOST, HOST]
    assert "3 of 6 layers" in report.notes[0]
    assert all(residency.valid(t, GPU) for t in prog.expert_w[0])
    assert all(residency.valid(t, HOST) and residency.copy(t, GPU) is None for t in prog.expert_w[5])
    forced, _, _ = placed(lambda p, r: placement.place_static_split(p, r, GPU, HOST, 1), spec)
    assert forced.trunk_device == [GPU] + [HOST] * 5


def test_gpu_host_placement_prefills_in_priority_order_until_full():
    spec = tiny_spec(hidden=256, expert_inter=1024, n_layers=2, n_experts=16)
    blob = int(spec.expert_params * 4.5 / 8)
    order = [(1, 5), (0, 3), (1, 0), (0, 0), (0, 1)]
    prog, residency, report = placed(
        lambda p, r: placement.place_gpu_host(p, r, GPU, HOST, prefill=order,
                                              cold_format=CompressionFormat.ENTROPY),
        spec, gpu_mem=int(3.7 * blob * 16 / 15),
    )  # fmt: skip
    on_gpu = [(layer, e) for layer, e in order if residency.valid(prog.expert_w[layer][e], GPU)]
    assert on_gpu == order[: len(on_gpu)] and 1 <= len(on_gpu) < len(order)
    assert f"prefilled {len(on_gpu)} experts" in report.notes[0]
    tensor = prog.expert_w[1][5]
    assert residency.copy(tensor, HOST).fmt is CompressionFormat.ENTROPY  # cold representation
    assert residency.copy(tensor, GPU).fmt is CompressionFormat.INT4 and residency.copy(tensor, HOST).owner
    assert residency.copy(tensor, HOST).nbytes < residency.copy(tensor, GPU).nbytes
    assert all(residency.copy(w, GPU).pinned for w in prog.trunk_w)


def test_balanced_distribution_separates_frequently_selected_experts():
    spec = tiny_spec(n_layers=1, n_experts=8)
    trace = synth_trace(200, 1, 8, 2, zipf_s=1.5, seed=3)
    freq = trace.expert_frequencies()
    prog, residency, _ = placed(
        lambda p, r: placement.place_modules(p, r, GPU, [MOD0, MOD1], frequencies=freq), spec
    )
    device = {e: residency.registry.owner(prog.expert_w[0][e]) for e in range(8)}
    hottest = sorted(range(8), key=lambda e: -freq[0][e])
    assert device[hottest[0]] != device[hottest[1]]  # the two hottest never share a module
    load = {m: sum(freq[0][e] for e in range(8) if device[e] == m) for m in (MOD0, MOD1)}
    assert abs(load[MOD0] - load[MOD1]) < 0.25 * sum(load.values())
    rr, residency, _ = placed(
        lambda p, r: placement.place_modules(p, r, GPU, [MOD0, MOD1], policy="round_robin"), spec
    )
    assert [residency.registry.owner(t) for t in rr.expert_w[0]] == [MOD0, MOD1] * 4
    assert [residency.registry.owner(t) for t in rr.head_w] == [MOD0, MOD1]
    assert placement.hottest_experts(freq)[0] == (0, hottest[0])
    with pytest.raises(placement.PlacementError):
        placed(lambda p, r: placement.place_single(p, r, GPU), tiny_spec(vocab=1 << 26))


# -- emission --------------------------------------------------------------------------------
def test_emitted_command_streams_are_valid_bsp_and_respect_descriptor_limits():
    """Wide fan-in (many experts feeding one reduction) is narrowed with BARRIER commands."""
    topology = make_topology()
    spec = tiny_spec(n_layers=1, n_experts=32, top_k=2, hidden=32, expert_inter=32)
    # no output-stationary mode: every expert output travels separately to the reducer
    modes = ALL_MODES - {DataflowMode.OUTPUT_STATIONARY}
    policy = ExecutionPolicy("wide", modes, cache=CachePolicy("benefit", (GPU,)))
    setup = build(topology, spec, policy,
                  lambda p, r: placement.place_modules(p, r, GPU, [MOD0, MOD1], policy="round_robin"),
                  batch=16, head_shards=2)  # fmt: skip
    trace = synth_trace(3, 1, 32, 2, batch=16, zipf_s=0.0, stickiness=0.0, seed=1)
    result = setup.run(trace, tokens=3, keep_log=True)
    assert result.ok, (result.failures, result.violations)
    commands = [state.cmd for state in setup.system.cmds.values()]
    assert len(commands) > 100
    for cmd in commands:
        validate(cmd)
        assert len(cmd.deps) <= MAX_DEPS and len(cmd.inputs) <= MAX_TENSORS
        assert cmd.decode(cmd.encode()) == cmd  # every descriptor survives the wire format
    opcodes = {cmd.opcode for cmd in commands}
    assert {Opcode.TRANSFER, Opcode.EXECUTE, Opcode.REDUCE, Opcode.BARRIER} <= opcodes
    widest = max(len(r.deps) for r in result.op_log)
    assert widest > MAX_TENSORS  # the reduction really did have more inputs than a descriptor holds
    kinds = {setup.registry.spec(t).kind for t in setup.program.expert_of}
    assert kinds == {TensorKind.EXPERT_WEIGHT}
