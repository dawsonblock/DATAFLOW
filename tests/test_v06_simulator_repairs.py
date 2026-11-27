"""v0.6 simulator repairs (audit findings S1-S3), as behaviour checks.

S1: shared experts' compute, parameter reads and contribution to the reduction are simulated.
S2: the embedding lookup and table and the final norm are simulated; tied storage counted once.
    Both: the lowered graph reconciles with an independent calculation from the dimensions.
S3: an explicit recording that cannot supply an experiment's routing is rejected, or the
    synthetic fallback is asked for and labelled; nothing synthetic is ever called "recorded".
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from basp.compiler.lower import ModelProgram
from basp.core.graph import Graph, OpCode
from basp.dse.hardware import HardwareDesign
from basp.experiments import variants
from basp.experiments.explore import _traces
from basp.experiments.suite import default_design
from basp.experiments.variants import (
    Context,
    TraceUnusable,
    routing_provenance,
    routing_source,
    run_architecture,
    set_context,
    traces,
)
from basp.memory.tensor_registry import TensorRegistry
from basp.verification.reconcile import expected_step, graph_accounting, lower_step, reconcile
from basp.workloads.model import tiny_spec
from basp.workloads.trace import synth_trace
from basp.workloads.zoo import MODELS

SHAPES = {
    "qwen3-30b-a3b": MODELS["qwen3-30b-a3b"],  # untied embeddings, query/key norms
    "deepseek-v3": MODELS["deepseek-v3"],  # one shared expert per layer
    "llama-3.3-70b": MODELS["llama-3.3-70b"],  # dense
    "tied": replace(MODELS["qwen3-8b"], name="tied-8b", tied_embeddings=True),
    "two-shared": tiny_spec(name="two-shared", n_shared_experts=2, n_heads=4, n_kv_heads=2, head_dim=16,
                            final_norm=True),
    "numeric": tiny_spec(),  # the small numeric model: attention-free, no final norm
}  # fmt: skip


def program(spec, **kw):
    return ModelProgram(spec, TensorRegistry(), **kw)


def routing(spec, batch):
    return synth_trace(1, spec.n_layers, spec.n_experts, spec.top_k, batch=batch, seed=3).steps[0]


# -- S1 + S2: graph and storage reconcile with the dimensions ------------------------------------
@pytest.mark.parametrize("name", sorted(SHAPES))
@pytest.mark.parametrize("options", [
    {"batch": 1}, {"batch": 3, "fused": False}, {"batch": 4, "prompt": True, "context_len": 64},
    {"batch": 2, "head_shards": 3, "context_len": 512},
], ids=["decode", "unfused-batch", "prompt", "sharded-head"])  # fmt: skip
def test_lowered_graph_reconciles_with_the_model_dimensions(name, options):
    spec = SHAPES[name]
    prog = program(spec, **options)
    result = reconcile(prog, routing(spec, prog.batch))
    assert result["ok"], [q for q in result["quantities"] if not q["ok"]] + [result["storage"]]
    assert result["storage"]["stored_params"] == spec.total_params and result["storage"]["unique"]


def test_shared_experts_are_computed_read_and_reduced():
    spec = SHAPES["deepseek-v3"]
    prog = program(spec)
    got = graph_accounting(prog, lower_step(prog, routing(spec, 1)))
    without = replace(spec, n_shared_experts=0)
    base = program(without)
    plain = graph_accounting(base, lower_step(base, routing(without, 1)))
    layers = spec.n_layers
    assert got["trunk_macs"] - plain["trunk_macs"] == layers * spec.shared_params  # their arithmetic
    assert got["trunk_params_read"] - plain["trunk_params_read"] == pytest.approx(layers * spec.shared_params)
    assert got["reduce_macs"] - plain["reduce_macs"] == layers * spec.hidden  # one more term in every reduction
    # the shared output really is an input of the reduction, in fused and unfused graphs
    for fused in (True, False):
        prog = program(spec, fused=fused)
        graph = Graph()
        x = prog.input_tensor(graph, 0)
        trunk = prog.trunk(graph, 0, 0, x)
        experts = prog.experts(graph, 0, 0, trunk.outputs["h"], routing(spec, 1)[0])
        reduce = prog.reduce(graph, 0, 0, experts.outputs, trunk.outputs["resid"], trunk.outputs["shared"]).ops[0]
        assert trunk.outputs["shared"] in reduce.inputs
        with pytest.raises(ValueError, match="shared experts"):
            prog.reduce(graph, 0, 0, experts.outputs, trunk.outputs["resid"])


def test_embedding_and_final_norm_are_stored_once_and_read_per_token():
    untied, tied = SHAPES["qwen3-30b-a3b"], SHAPES["tied"]
    a, b = program(untied, batch=2), program(tied, batch=2)
    assert a.embed_w is not None and b.embed_w is None  # tied: the output projection is the table
    assert a.final_norm_w is not None and b.final_norm_w is not None
    stored = lambda p: sum(p.registry.spec(t).n_weights for t in p.weight_tensors())
    assert stored(a) == untied.total_params and stored(b) == tied.total_params
    assert tied.total_params == replace(tied, tied_embeddings=False).total_params - tied.embed_params
    for prog, spec in ((a, untied), (b, tied)):
        ops = lower_step(prog, routing(spec, 2))
        embed = next(o for o in ops if o.opcode is OpCode.EMBED)
        norm = next(o for o in ops if o.attrs.get("final_norm"))
        table = prog.registry.spec(embed.weights[0])
        assert embed.weight_fraction * table.n_weights == pytest.approx(2 * spec.hidden)  # two rows read
        assert norm.macs == 2 * spec.hidden and ops[-1].inputs == norm.outputs  # the head reads the normed state
    numeric = program(SHAPES["numeric"])
    assert numeric.final_norm_w is None and expected_step(numeric, routing(SHAPES["numeric"], 1))["final_norm_macs"] == 0


def test_the_simulation_charges_the_new_work():
    spec = MODELS["deepseek-v3"]
    small = replace(spec, name="shared-small", n_layers=4, hidden=512, expert_inter=256, n_experts=16, top_k=4,
                    vocab=4096, n_heads=8, n_kv_heads=2, head_dim=64, attn_params_override=None)  # fmt: skip
    plain = replace(small, name="plain-small", n_shared_experts=0)
    speeds = {}
    for s in (small, plain):
        prior, held = synth_trace(12, s.n_layers, s.n_experts, s.top_k, seed=1).split(8)
        out = run_architecture("A", default_design("A", HardwareDesign(modules=0)), s, prior, held, tokens=4,
                               warmup_tokens=1, keep_log=True)  # fmt: skip
        speeds[s.name] = out.result.tokens_per_s
        names = {r.name.split(".")[-1] for r in out.result.op_log}
        assert {"embed", "final_norm", "lm_head"} <= names
    assert speeds["shared-small"] < speeds["plain-small"]  # the shared expert's work takes time


# -- S3: recorded means recorded ------------------------------------------------------------------
@pytest.fixture
def recorded_context():
    spec = MODELS["qwen3-30b-a3b"]
    short = synth_trace(20, spec.n_layers, spec.n_experts, spec.top_k, seed=9)
    previous = set_context(Context(trace=short, trace_path="short.decode.json"))
    yield spec, short
    set_context(previous)


def test_a_recording_that_is_too_short_is_rejected_not_replaced(recorded_context):
    spec, _ = recorded_context
    traces(spec, tokens=4, prior_steps=8)  # 12 steps: fits
    assert routing_provenance()[-1]["source"] == "recorded" and routing_source(spec) == "recorded (short.decode.json)"
    with pytest.raises(TraceUnusable, match="has 20 steps; 38 are needed"):
        traces(spec, tokens=6, prior_steps=32)
    assert len(routing_provenance()) == 1  # the refused window was not logged as anything
    # the fallback has to be asked for, and is labelled
    traces(spec, tokens=6, prior_steps=32, fallback="synthetic")
    last = routing_provenance()[-1]
    assert last["source"] == "synthetic" and "fallback: the recording has 20 steps" in last["reason"]
    assert routing_source(spec) == "recorded (short.decode.json) for 1 of 2 experiment setups, synthetic for the rest"
    with pytest.raises(ValueError, match="fallback must be"):
        traces(spec, tokens=1, fallback="quietly")


def test_other_models_and_batched_recordings_are_never_called_recorded(recorded_context):
    spec, _ = recorded_context
    other = MODELS["llama-3.3-70b"]
    traces(other, tokens=2, prior_steps=4)
    assert routing_provenance()[-1] == {"model": other.name, "tokens": 2, "prior_steps": 4, "batch": 1,
                                        "source": "synthetic", "reason": "the recording is of another model shape"}  # fmt: skip
    assert routing_source(other) == "synthetic"
    batched = synth_trace(40, spec.n_layers, spec.n_experts, spec.top_k, batch=4, seed=2)
    set_context(Context(trace=batched, trace_path="prefill.json"))
    with pytest.raises(TraceUnusable, match="single-stream"):
        traces(spec, tokens=2, prior_steps=4)
    set_context(Context())
    traces(spec, tokens=2, prior_steps=4)
    assert routing_provenance()[-1]["reason"] == "no recording was given" and routing_source(spec) == "synthetic"


def test_the_explorer_refuses_an_unusable_recording(tmp_path):
    spec = MODELS["qwen3-30b-a3b"]
    path = tmp_path / "short.json"
    synth_trace(5, spec.n_layers, spec.n_experts, spec.top_k, seed=1).save(path)
    with pytest.raises(TraceUnusable, match="has 5 steps"):
        _traces("qwen3-30b-a3b", 4, 0.8, 0.6, 7, 24, 1, str(path))
    with pytest.raises(ValueError, match="does not match"):
        _traces("llama-3.3-70b", 4, 0.8, 0.6, 7, 24, 1, str(path))
    assert variants.unusable_reason(synth_trace(40, 2, 8, 2), 8, 4, 2) is None
