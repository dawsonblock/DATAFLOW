"""v0.2 simulator mechanisms: batched launches, gating, energy-aware scheduling, host-free
dispatch, eager activation streaming, concurrent streams, placement and tensor-parallel experts."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from basp.analysis import analyse_critical_path, scaling_limit
from basp.backends.cpu import NumericPayload
from basp.core.capabilities import DeviceCapabilities, DeviceKind
from basp.core.graph import OpCode
from basp.core.tensor import CompressionFormat as F
from basp.core.topology import DeviceSpec, LinkSpec, PowerSpec, Topology
from basp.dataflow.base import ALL_MODES, CachePolicy, DataflowMode, ExecutionPolicy, PrefetchPolicy
from basp.experiments.harness import build
from basp.experiments.microbench import MicroSystem
from basp.kernels import QuantizedExpert, run_expert, shard_bounds, split_expert
from basp.scheduler import placement
from basp.simulator.energy import energy_report
from basp.verification import check_accounting, check_dependencies, compare_to_reference
from basp.workloads.model import tiny_spec
from basp.workloads.tiny import TinyModel
from basp.workloads.trace import synth_trace
from tests.conftest import GPU, HOST, MOD0, MOD1, make_topology

M = DataflowMode
PARAMS = 3 * 2048 * 768
ACT = 4096


def with_devices(topology: Topology, **changes) -> Topology:
    """Copy of the test topology with fields changed on every compute device."""
    devices = [replace(d, **changes) if d.caps.can_compute else d for d in topology.devices.values()]
    return Topology(devices, topology.links, host=topology.host,
                    sync_latency_s=topology.sync_latency_s, name=topology.name)  # fmt: skip


def gpu_experts(topology: Topology, n: int = 3):
    """``n`` experts resident on the GPU, issued as one batch."""
    ms = MicroSystem(topology, ExecutionPolicy("t", frozenset({M.WEIGHT_STATIONARY}), joint_rounds=0))
    ops = []
    for i in range(n):
        w = ms.weight(f"w{i}", PARAMS, GPU)
        x, y = ms.activation(f"x{i}", ACT, GPU), ms.activation(f"y{i}", ACT)
        ops.append(ms.op(OpCode.EXPERT, f"e{i}", inputs=(x,), weights=(w,), outputs=(y,), macs=PARAMS,
                         output_device=GPU))  # fmt: skip
    return ms, ms.run(ops)


# -- batched launches and persistent kernels ---------------------------------------------------
def test_back_to_back_kernels_share_one_launch():
    kernel = max(PARAMS * 4.5 / 8 / 500e9, PARAMS / 20e12)
    cold = 3e-6
    _, every = gpu_experts(make_topology())  # no warm start-up declared: one launch each
    assert every.actual_s == pytest.approx(3 * (cold + kernel), rel=1e-9)
    ms, batched = gpu_experts(with_devices(make_topology(), warm_startup_s=0.2e-6))
    assert batched.actual_s == pytest.approx(cold + 2 * 0.2e-6 + 3 * kernel, rel=1e-9)
    assert batched.actual_s == pytest.approx(batched.estimate_s, rel=1e-9)  # estimator agrees
    assert ms.system.executors[GPU].warm_starts == 2
    _, resident = gpu_experts(with_devices(make_topology(), warm_startup_s=0.0, persistent=True))
    assert resident.actual_s == pytest.approx(3 * kernel, rel=1e-9)  # no launch at all


def test_a_kernel_after_an_idle_gap_pays_the_full_launch():
    topology = with_devices(make_topology(), warm_startup_s=0.0)
    ms = MicroSystem(topology, ExecutionPolicy("t", frozenset({M.WEIGHT_STATIONARY})))
    w = ms.weight("w", PARAMS, GPU)
    times = []
    for i in range(2):
        x, y = ms.activation(f"x{i}", ACT, GPU), ms.activation(f"y{i}", ACT)
        op = ms.op(OpCode.EXPERT, f"e{i}", inputs=(x,), weights=(w,), outputs=(y,), macs=PARAMS,
                   output_device=GPU)  # fmt: skip
        times.append(ms.run([op]).actual_s)
        ms.system.sim.run(until=ms.system.sim.now + 1e-3)  # let the device go idle
    assert times[0] == pytest.approx(times[1]) and ms.system.executors[GPU].warm_starts == 0


# -- energy: gating and energy-aware scheduling ------------------------------------------------
def test_idle_devices_are_gated_after_their_threshold():
    power = PowerSpec(idle_w=10.0, sleep_w=1.0, gate_after_s=1e-3, wake_j=0.5)
    topology = with_devices(make_topology(), power=power)
    ms, _ = gpu_experts(topology, n=1)
    system = ms.system
    system.sim.run(until=0.1)
    system.net.settle()
    gating = {d: (ex.gated_seconds(0.1), ex.wakes) for d, ex in system.executors.items()}
    # modules never ran: gated for the whole run minus the threshold
    assert gating[MOD0][0] == pytest.approx(0.1 - 1e-3)
    assert 0.09 < gating[GPU][0] < 0.1 - 1e-3  # busy for a moment, gated after the threshold
    report = energy_report(topology, system.res, 0.1, gating)
    always_on = energy_report(topology, system.res, 0.1)
    assert report.gated_s["mod0"] == pytest.approx(0.099)
    assert report.static_j < 0.2 * always_on.static_j
    assert always_on.static_j == pytest.approx(4 * 10.0 * 0.1)
    persistent = with_devices(topology, persistent=True)
    ms, _ = gpu_experts(persistent, n=1)
    ms.system.sim.run(until=0.1)
    assert ms.system.executors[MOD0].gated_seconds(0.1) == 0.0  # resident kernels keep it awake


def test_energy_aware_scheduling_trades_time_for_energy():
    hungry = PowerSpec(joules_per_mac=5e-9)  # the module is fast but expensive per operation
    frugal = PowerSpec(joules_per_mac=1e-12)
    base = make_topology()
    devices = [replace(d, power=hungry if d.device_id == MOD0 else frugal) for d in base.devices.values()]
    topology = Topology(devices, base.links, host=HOST, sync_latency_s=base.sync_latency_s)

    def run(weight: float):
        policy = ExecutionPolicy("t", frozenset({M.WEIGHT_STATIONARY}), energy_weight_s_per_j=weight)
        ms = MicroSystem(topology, policy)
        w = ms.weight("w", PARAMS, MOD0)
        ms.replicate(w, HOST)
        x, y = ms.activation("x", ACT, GPU), ms.activation("y", ACT)
        op = ms.op(OpCode.EXPERT, "e", inputs=(x,), weights=(w,), outputs=(y,), macs=PARAMS,
                   output_device=GPU)  # fmt: skip
        return ms.run([op]).plan

    fast, thrifty = run(0.0), run(1.0)
    assert fast.devices == (MOD0,) and thrifty.devices == (HOST,)
    assert thrifty.energy_j < 0.01 * fast.energy_j and thrifty.finish > fast.finish
    assert fast.energy_j == pytest.approx(PARAMS * 5e-9, rel=1e-6)
    with pytest.raises(ValueError):
        ExecutionPolicy("bad", energy_weight_s_per_j=-1.0)


# -- host-free dispatch and eager activations ---------------------------------------------------
SPEC = tiny_spec(hidden=128, expert_inter=256, n_layers=4, n_experts=16, top_k=4, n_heads=4,
                 n_kv_heads=2, head_dim=32, vocab=1024)  # fmt: skip
STATIC = frozenset({M.WEIGHT_STATIONARY, M.DISTRIBUTED})


def module_run(policy: ExecutionPolicy, *, tokens: int = 8, topology: Topology | None = None,
               concurrent: int = 0, batch: int = 1, replicas: int = 1, keep_log: bool = True):  # fmt: skip
    rows = max(batch, concurrent)
    trace = synth_trace(24 + tokens, SPEC.n_layers, SPEC.n_experts, SPEC.top_k, batch=rows, seed=4)
    prior, held = trace.split(24)
    setup = build(topology or make_topology(), SPEC, policy,
                  lambda p, r: placement.place_modules(p, r, GPU, [MOD0, MOD1], replicas=replicas,
                                                       frequencies=prior.expert_frequencies()),
                  batch=batch, context_len=64, max_new_tokens=tokens, head_shards=2, prior=prior)  # fmt: skip
    if concurrent:
        result = setup.run_concurrent(held, streams=concurrent, tokens=tokens, keep_log=keep_log)
    else:
        result = setup.run(held, tokens=tokens, keep_log=keep_log)
    assert result.ok, (result.failures[:2], result.violations[:2])
    assert check_accounting(setup.system) == []
    assert check_dependencies(result.op_log, setup.system) == []
    return setup, result


def test_dependency_events_remove_exactly_the_host_round_trips():
    _, host = module_run(ExecutionPolicy("host", STATIC))
    _, free = module_run(ExecutionPolicy("free", STATIC, host_in_loop=False))
    # one batch per layer plus the first trunk block, each followed by a 2 us host round trip
    saved = host.token_latency_s - free.token_latency_s
    assert saved == pytest.approx((SPEC.n_layers + 1) * 2e-6, rel=0.02)
    for modes, extra in ((ALL_MODES, {}), (STATIC, {"cache": CachePolicy("lru", (GPU,))}),
                         (STATIC, {"prefetch": PrefetchPolicy()})):  # fmt: skip
        with pytest.raises(ValueError, match="static dataflow"):
            ExecutionPolicy("bad", modes, host_in_loop=False, **extra)


def test_eager_activation_streaming_overlaps_the_routing_decision():
    slow = make_topology()
    slow = Topology(list(slow.devices.values()), [replace(k, latency_s=k.latency_s * 10) for k in slow.links],
                    host=HOST, sync_latency_s=40e-6)  # fmt: skip
    _, lazy = module_run(ExecutionPolicy("lazy", STATIC), topology=slow)
    setup, eager = module_run(ExecutionPolicy("eager", STATIC, eager_activations=True), topology=slow)
    # the hidden state travels (70 us) while the host round trip (40 us) is in progress
    saved = lazy.token_latency_s - eager.token_latency_s
    assert saved == pytest.approx(SPEC.n_layers * 40e-6, rel=0.1)
    assert eager.link_bytes["activation"] >= lazy.link_bytes["activation"]  # sent to every module
    tags = {s.tag for s in setup.system.cmds.values() if s.meta and s.meta[0] == "eager"}
    assert tags == {"activation"}


# -- concurrent request streams -------------------------------------------------------------------
def test_concurrent_streams_share_the_system_and_complete_independently():
    _, single = module_run(ExecutionPolicy("d", STATIC), tokens=6)
    setup, group = module_run(ExecutionPolicy("d", STATIC), tokens=6, concurrent=3)
    assert group.streams == 3 and group.tokens == 18 and len(group.token_times) == 18
    assert group.tokens_per_s > single.tokens_per_s  # idle module time is used by other streams
    assert group.token_latency_s > single.token_latency_s  # each stream waits for the others
    steps = sorted((r.stream, r.step) for r in group.op_log if r.kind == "head")
    assert steps == [(s, t) for s in range(3) for t in range(6)]
    setup.residency.check_invariants()
    _, batch = module_run(ExecutionPolicy("d", STATIC), tokens=6, batch=3)
    assert batch.tokens == 6 and batch.batch == 3
    with pytest.raises(ValueError, match="rows"):
        build(make_topology(), SPEC, ExecutionPolicy("d", STATIC),
              lambda p, r: placement.place_modules(p, r, GPU, [MOD0, MOD1])).run_concurrent(
            synth_trace(4, SPEC.n_layers, SPEC.n_experts, SPEC.top_k), streams=2, tokens=2)  # fmt: skip


def test_replicated_experts_let_the_scheduler_balance_modules():
    _, single = module_run(ExecutionPolicy("d", STATIC), tokens=10)
    setup, double = module_run(ExecutionPolicy("d", STATIC), tokens=10, replicas=2)
    assert double.token_latency_s < single.token_latency_s
    holders = [len(setup.residency.holders(t)) for t in setup.program.layer_expert_tensors(0)]
    assert set(holders) == {2}
    assert "expert replicas" in " ".join(setup.placement.notes)


# -- critical path ---------------------------------------------------------------------------------
def test_critical_path_components_account_for_the_whole_token():
    _, result = module_run(ExecutionPolicy("d", STATIC), tokens=8)
    report = analyse_critical_path(result.op_log, skip_steps=2)
    assert report.tokens == 6
    assert report.residual_s < 1e-12
    assert sum(report.by_component.values()) == pytest.approx(report.token_time_s, rel=1e-9)
    assert sum(report.by_kind.values()) + report.by_component["host"] == pytest.approx(
        report.token_time_s, rel=1e-9)  # fmt: skip
    assert report.token_time_s == pytest.approx(np.mean(result.token_times[2:]), rel=1e-9)
    assert report.expert_ops == SPEC.n_layers * SPEC.top_k
    assert 0 < report.expert_phase_s < report.token_time_s and report.serial_s > 0
    # the host issues the first trunk block and then one batch per layer: five round trips
    assert report.by_component["host"] == pytest.approx((SPEC.n_layers + 1) * 2e-6, rel=1e-6)
    limits = [scaling_limit(report, 2, n, layers=SPEC.n_layers) for n in (1, 2, 4, 8)]
    assert limits[0]["ideal_efficiency"] == pytest.approx(1.0)
    efficiencies = [x["ideal_efficiency"] for x in limits]
    assert efficiencies == sorted(efficiencies, reverse=True)  # the serial part bites harder
    assert limits[1]["ideal_token_time_s"] <= report.token_time_s * (1 + 1e-9)
    assert analyse_critical_path([]).tokens == 0


# -- placement -------------------------------------------------------------------------------------
def test_coactivation_estimate_is_regularised_towards_independence():
    short = synth_trace(16, 1, 32, 4, seed=1)
    co = placement.coactivation(short)
    freq = short.expert_frequencies()[0]
    independent = np.outer(freq, freq)
    off = ~np.eye(32, dtype=bool)
    # 16 visits against a prior of 16 * (32 / 4)^2 = 1024 pseudo visits: close to independence
    assert np.abs(co[0] - independent)[off].max() < 0.02
    raw = placement.coactivation(short, prior_visits=0.0)
    assert np.abs(raw[0] - independent)[off].max() > 0.05  # the unregularised estimate is noisy
    assert np.allclose(np.diag(raw[0]), freq)
    paired = [[__import__("basp.workloads.trace", fromlist=["RoutingStep"]).RoutingStep(
        ((0, 1, 2, 3) if s % 2 else (4, 5, 6, 7),), ((0.25,) * 4,))] for s in range(4000)]  # fmt: skip
    trace = type(short)(1, 8, 4, 1, paired)
    structured = placement.coactivation(trace)
    assert structured[0, 0, 1] > 0.4 and structured[0, 0, 4] < 0.05  # real structure survives


def test_output_projection_is_placed_where_the_scheduler_can_choose():
    topology = make_topology()
    setup = build(topology, SPEC, ExecutionPolicy("d", STATIC),
                  lambda p, r: placement.place_modules(p, r, GPU, [MOD0, MOD1]), head_shards=2)  # fmt: skip
    for shard, module in zip(setup.program.head_w, (MOD0, MOD1)):
        assert setup.residency.valid(shard, module) and setup.residency.valid(shard, GPU)
    trace = synth_trace(4, SPEC.n_layers, SPEC.n_experts, SPEC.top_k)
    result = setup.run(trace, tokens=3, keep_log=True)
    heads = [r for r in result.op_log if r.kind == "head"]
    assert all(r.devices == (GPU,) for r in heads)  # the GPU copy is faster here
    only = build(topology, SPEC, ExecutionPolicy("d", STATIC),
                 lambda p, r: placement.place_modules(p, r, GPU, [MOD0, MOD1], head="modules"),
                 head_shards=2)  # fmt: skip
    result = only.run(trace, tokens=3, keep_log=True)
    assert all(r.devices == (MOD0, MOD1) for r in result.op_log if r.kind == "head")
    for bad in ({"modules": [MOD0, MOD1], "head": "nowhere"}, {"modules": []},
                {"modules": [MOD0], "replicas": 0}, {"modules": [MOD0], "policy": "alphabetical"}):  # fmt: skip
        with pytest.raises(placement.PlacementError):
            build(topology, SPEC, ExecutionPolicy("d", STATIC),
                  lambda p, r, bad=bad: placement.place_modules(p, r, GPU, **bad), head_shards=2)  # fmt: skip


def test_dense_models_spread_their_layers_over_modules():
    dense = tiny_spec(n_experts=1, top_k=1, n_layers=4)
    setup = build(make_topology(), dense, ExecutionPolicy("d", STATIC),
                  lambda p, r: placement.place_modules(p, r, GPU, [MOD0, MOD1]), head_shards=2)  # fmt: skip
    owners = [setup.registry.owner(setup.program.expert_w[layer][0]) for layer in range(4)]
    assert owners == [MOD0, MOD1, MOD0, MOD1]
    assert dense.dense and dense.router_params == 0


# -- tensor-parallel experts -------------------------------------------------------------------------
def test_expert_shards_sum_to_the_whole_expert():
    rng = np.random.default_rng(0)
    h, m = 64, 160
    expert = QuantizedExpert.from_float(rng.standard_normal((m, h)) / 8, rng.standard_normal((m, h)) / 8,
                                        rng.standard_normal((h, m)) / 12)  # fmt: skip
    x = rng.standard_normal((3, h)).astype(np.float32)
    whole = run_expert(expert, x)
    for parts in (2, 3, 5):
        bounds = shard_bounds(m, parts)
        assert bounds[0][0] == 0 and bounds[-1][1] == m
        assert all(a % 32 == 0 for a, _ in bounds)  # shards keep whole quantisation groups
        shards = split_expert(expert, bounds)
        assert sum(s.n_weights for s in shards) == expert.n_weights
        assert sum(s.nbytes for s in shards) == expert.nbytes
        np.testing.assert_allclose(sum(run_expert(s, x) for s in shards), whole, rtol=1e-5, atol=1e-5)
    assert shard_bounds(40, 3) == ((0, 13), (13, 27), (27, 40))  # too small for group alignment
    with pytest.raises(ValueError):
        shard_bounds(4, 8)


@pytest.mark.parametrize("accumulate", [False, True], ids=["pushed", "partial-sums"])
def test_tensor_parallel_co_simulation_matches_the_reference(accumulate: bool):
    spec = tiny_spec()
    model = TinyModel.random(spec, seed=3)
    modes = {M.DISTRIBUTED, M.WEIGHT_STATIONARY} | ({M.OUTPUT_STATIONARY} if accumulate else set())
    setup = build(make_topology(), spec, ExecutionPolicy("e", frozenset(modes)),
                  lambda p, r: placement.place_tensor_parallel(p, r, GPU, [MOD0, MOD1]),
                  batch=3, head_shards=2, expert_shards=2)  # fmt: skip
    payload = NumericPayload(model, setup.program, model.initial_input(3))
    result = setup.run(None, tokens=4, payload=payload, keep_log=True)
    assert result.ok and compare_to_reference(model, payload).ok
    assert check_dependencies(result.op_log, setup.system) == []
    assert check_accounting(setup.system) == []
    setup.residency.check_invariants()
    experts = [r for r in result.op_log if r.kind == "expert"]
    assert experts and all(r.devices == (MOD0, MOD1) for r in experts)  # every expert uses both
    if accumulate:
        assert payload.partials and result.mode_counts["output_stationary"] > 0
    # each module holds one shard of every expert, and nothing else of it
    for parts in setup.program.expert_parts[0]:
        assert [setup.registry.owner(t) for t in parts] == [MOD0, MOD1]
    with pytest.raises(placement.PlacementError, match="expert shards"):
        placement.place_tensor_parallel(setup.program, setup.residency, GPU, [MOD0])
    with pytest.raises(ValueError, match="fused"):
        build(make_topology(), spec, ExecutionPolicy("e", frozenset(modes)),
              lambda p, r: placement.place_tensor_parallel(p, r, GPU, [MOD0, MOD1]),
              expert_shards=2, fused=False)  # fmt: skip


def test_tensor_parallel_output_traffic_halves_with_partial_sums():
    spec = tiny_spec()
    out = {}
    for name, modes in (("pushed", {M.DISTRIBUTED, M.WEIGHT_STATIONARY}),
                        ("summed", {M.DISTRIBUTED, M.WEIGHT_STATIONARY, M.OUTPUT_STATIONARY})):  # fmt: skip
        setup = build(make_topology(), spec, ExecutionPolicy(name, frozenset(modes)),
                      lambda p, r: placement.place_tensor_parallel(p, r, GPU, [MOD0, MOD1], head="trunk"),
                      expert_shards=2)  # fmt: skip
        trace = synth_trace(6, spec.n_layers, spec.n_experts, spec.top_k, seed=2)
        out[name] = setup.run(trace, tokens=6).link_bytes["output"]
    # top_k = 2 experts on 2 modules: 4 partial outputs per layer pushed, 2 partial sums otherwise
    assert out["summed"] == pytest.approx(out["pushed"] / 2)


def test_routing_only_devices_carry_traffic_but_hold_nothing():
    caps = DeviceCapabilities(decode_rate={F.INT4: math.inf}, stream_formats=frozenset({F.INT4}))
    devices = [
        DeviceSpec(0, "gpu", DeviceKind.GPU, 1 << 30, 500e9, 20e12, caps),
        DeviceSpec(1, "fabric", DeviceKind.SWITCH, 0, 1e15),
        DeviceSpec(2, "mod", DeviceKind.NEAR_MEMORY, 1 << 30, 54e9, 256e9, caps),
    ]
    topology = Topology(devices, [LinkSpec(0, 1, 64e9, latency_s=0.5e-6), LinkSpec(1, 2, 64e9, latency_s=0.5e-6)])
    assert topology.path_latency(0, 2) == pytest.approx(1e-6)
    assert topology.transfer_rate(0, 2) == 54e9  # endpoints and links, not the switch's memory
    ms = MicroSystem(topology, ExecutionPolicy("t", frozenset({M.WEIGHT_STATIONARY})))
    w = ms.weight("w", PARAMS, 2)
    x, y = ms.activation("x", ACT, 0), ms.activation("y", ACT)
    op = ms.op(OpCode.EXPERT, "e", inputs=(x,), weights=(w,), outputs=(y,), macs=PARAMS, output_device=0)
    result = ms.run([op])
    assert result.plan.devices == (2,) and not result.failures
    assert result.actual_s == pytest.approx(result.estimate_s, rel=1e-9)
