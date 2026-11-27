"""End-to-end verification: numerics, dependencies, accounting and model-behaviour invariance."""

from __future__ import annotations

import numpy as np
import pytest

from basp.backends.cpu import NumericPayload
from basp.backends.cpu.executor import NumericError
from basp.dataflow.base import ALL_MODES, CachePolicy, DataflowMode, ExecutionPolicy, PrefetchPolicy
from basp.experiments.harness import build
from basp.scheduler import placement
from basp.verification import check_accounting, check_dependencies, compare_to_reference
from basp.workloads.model import tiny_spec
from basp.workloads.tiny import TinyModel
from basp.workloads.trace import synth_trace
from tests.conftest import GPU, HOST, MOD0, MOD1, make_topology

M = DataflowMode
SPEC = tiny_spec()
MODEL = TinyModel.random(SPEC, seed=3)

POLICIES = {
    "weight-stationary": ExecutionPolicy("ws", frozenset({M.WEIGHT_STATIONARY, M.DISTRIBUTED})),
    "gpu-paging": ExecutionPolicy("paging", frozenset({M.ACTIVATION_STATIONARY}), frozenset({GPU}),
                                  cache=CachePolicy("lru", (GPU,), background=False)),
    "streaming": ExecutionPolicy("stream", frozenset({M.STREAMING}), frozenset({GPU})),
    "complete": ExecutionPolicy("complete", ALL_MODES, cache=CachePolicy("benefit", (GPU,)),
                                prefetch=PrefetchPolicy(min_probability=0.2)),
}  # fmt: skip


def modules(program, residency):
    return placement.place_modules(program, residency, GPU, [MOD0, MOD1], policy="round_robin")


def co_simulate(policy: ExecutionPolicy, *, batch: int = 2, fused: bool = True, tokens: int = 5,
                head_shards: int = 2, place=modules):  # fmt: skip
    setup = build(make_topology(), SPEC, policy, place, batch=batch, fused=fused, head_shards=head_shards)
    payload = NumericPayload(MODEL, setup.program, MODEL.initial_input(batch))
    result = setup.run(None, tokens=tokens, payload=payload, keep_log=True)
    return setup, payload, result


@pytest.mark.parametrize("name", POLICIES)
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "unfused"])
def test_co_simulation_matches_the_reference_under_every_policy(name: str, fused: bool):
    setup, payload, result = co_simulate(POLICIES[name], fused=fused)
    assert result.ok, (result.failures[:3], result.violations[:3])
    report = compare_to_reference(MODEL, payload)
    # FP16 activation rounding in the streaming kernels: about 5e-4 per expert stage
    assert report.ok and report.max_rel_l2 < 2e-3, report
    assert check_dependencies(result.op_log, setup.system) == []
    assert check_accounting(setup.system) == []
    setup.residency.check_invariants()
    assert len(payload.logits) == result.tokens == 5
    assert setup.system.stats["completion_overflow"] == 0


#: FP32 partial sums added in a different order can flip an FP16 rounding downstream; the effect
#: on logits of magnitude ~4 was at most 8e-4 over 60 model seeds
REORDER_ATOL = 5e-3


def run_logits(model: TinyModel, policy: ExecutionPolicy, *, fused: bool = True, place=modules,
               head_shards: int = 2):  # fmt: skip
    setup = build(make_topology(), SPEC, policy, place, batch=2, fused=fused, head_shards=head_shards)
    payload = NumericPayload(model, setup.program, model.initial_input(2))
    result = setup.run(None, tokens=5, payload=payload)
    assert result.ok
    routing = [[step.experts for step in layer] for layer in payload.routing]
    return routing, np.stack(payload.logits), payload


@pytest.mark.parametrize("seed", [3, 11, 20, 47])
def test_dataflow_decisions_never_change_model_behaviour(seed: int):
    """The runtime never changes which experts run, and changes values only by summation order.

    Policies that keep the reduction order give bit-identical logits whatever the placement,
    transfer strategy or graph form.  A policy that accumulates expert outputs as device-local
    partial sums adds the same FP32 terms in a different order: identical routing, logits equal
    within rounding -- not bit for bit (v0.1 asserted bit-identity for one seed, which held by
    luck and failed on another platform).
    """
    model = TinyModel.random(SPEC, seed=seed)
    routing, logits, _ = run_logits(model, POLICIES["weight-stationary"])
    for name in ("weight-stationary", "gpu-paging", "streaming"):
        for fused in (True, False):
            other_routing, other, payload = run_logits(model, POLICIES[name], fused=fused)
            assert not payload.partials, name
            assert other_routing == routing, f"{name}: routing differs"
            np.testing.assert_array_equal(other, logits, err_msg=name)
    no_partials = ExecutionPolicy("np", ALL_MODES - {M.OUTPUT_STATIONARY}, cache=CachePolicy("benefit", (GPU,)),
                                  prefetch=PrefetchPolicy(min_probability=0.2))  # fmt: skip
    for policy, kwargs in ((no_partials, {}), (no_partials, {"fused": False}),
                           (no_partials, {"place": lambda p, r: placement.place_single(p, r, HOST),
                                          "head_shards": 1})):  # fmt: skip
        other_routing, other, payload = run_logits(model, policy, **kwargs)
        assert not payload.partials and other_routing == routing
        np.testing.assert_array_equal(other, logits)
    for fused in (True, False):
        other_routing, other, payload = run_logits(model, POLICIES["complete"], fused=fused)
        assert payload.partials  # device-local partial sums: a different summation order
        assert other_routing == routing, "partial sums must not change routing"
        np.testing.assert_allclose(other, logits, rtol=0, atol=REORDER_ATOL)


def test_output_stationary_partial_sums_are_exercised_and_equivalent():
    both = ExecutionPolicy("os", frozenset({M.WEIGHT_STATIONARY, M.OUTPUT_STATIONARY, M.DISTRIBUTED}))
    _, payload, result = co_simulate(both, batch=3)
    _, plain, plain_result = co_simulate(POLICIES["weight-stationary"], batch=3)
    assert result.mode_counts["output_stationary"] > 0
    assert any(len(key) == 3 and key[0] == "partial" for key in payload.partials)
    np.testing.assert_allclose(np.stack(payload.logits), np.stack(plain.logits), rtol=0, atol=REORDER_ATOL)
    assert result.link_bytes["output"] < plain_result.link_bytes["output"]  # fewer tensors return
    assert result.token_latency_s == pytest.approx(plain_result.token_latency_s, rel=0.02)


def test_runs_are_deterministic():
    def run():
        setup, payload, result = co_simulate(POLICIES["complete"])
        log = [(r.cmd_id, r.opcode, r.device, r.status, r.finished) for r in setup.system.log]
        return result.token_times, result.link_bytes, log, np.stack(payload.logits).tobytes()

    assert run() == run()


def test_missing_values_are_detected_by_the_numeric_payload():
    setup, payload, _ = co_simulate(POLICIES["weight-stationary"], tokens=1)
    op = setup.program.probe_expert(0, 0)
    assert op.inputs and op.outputs and setup.program.probe_expert(0, 0) is op  # cached template
    with pytest.raises(NumericError):
        payload._get(987654, "probe")
    with pytest.raises(NumericError, match="no numeric implementation"):
        payload.execute(type(op)(1, type(op.opcode).ATTENTION, "attn"), None)
    with pytest.raises(ValueError):
        NumericPayload(MODEL, setup.program, np.zeros((7, 3), dtype=np.float32))


def test_trace_driven_run_reports_consistent_metrics():
    spec = tiny_spec(hidden=128, expert_inter=256, n_layers=4, n_experts=16, top_k=4, n_heads=4,
                     n_kv_heads=2, head_dim=32, vocab=1024)  # fmt: skip
    trace = synth_trace(40, spec.n_layers, spec.n_experts, spec.top_k, seed=2)
    prior, held = trace.split(24)
    policy = POLICIES["complete"]
    setup = build(make_topology(), spec, policy,
                  lambda p, r: placement.place_modules(p, r, GPU, [MOD0, MOD1],
                                                       frequencies=prior.expert_frequencies()),
                  context_len=64, max_new_tokens=16, head_shards=2, prior=prior)  # fmt: skip
    result = setup.run(held, tokens=12, warmup_tokens=4, keep_log=True)
    assert result.ok and result.tokens == 8 and len(result.token_times) == 8
    assert result.tokens_per_s == pytest.approx(8 / result.elapsed_s)
    assert result.elapsed_s == pytest.approx(sum(result.token_times), rel=1e-9)
    assert result.token_latency_s == pytest.approx(result.elapsed_s / 8)
    assert result.cache["expert_ops"] == 8 * spec.n_layers * spec.top_k
    assert 0.0 <= result.cache["hit_rate"] <= 1.0
    assert result.cache["fills_failed"] == 0
    assert result.energy.total_j == 0.0  # the test topology declares no power figures
    assert result.estimate_error["mean_abs_rel"] < 0.05  # estimates track simulation closely
    assert result.link_bytes_per_token == pytest.approx(result.link_bytes_total / 8)
    assert check_dependencies(result.op_log, setup.system) == []
    assert check_accounting(setup.system) == []
    # every KV cache tensor was appended to once per decoded token
    assert {setup.registry.version(kv) for kv in setup.program.kv} == {12}
    for row in result.devices.values():
        assert 0.0 <= row["mem_utilisation"] <= 1.0 + 1e-9
    assert sum(result.mode_counts.values()) >= result.scheduler["ops"]
    with pytest.raises(ValueError):
        setup.run(held, tokens=2, warmup_tokens=2)
