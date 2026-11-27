"""Cost model, dataflow modes and the joint scheduler.

The central property: on an idle system the analytical estimate
``T_queue + T_input + T_kernel + T_output`` equals the simulated completion time
for every dataflow mode, because estimate and simulation share one kernel
description.
"""

from __future__ import annotations

import pytest

from basp.core.graph import Graph, OpCode, OpNode
from basp.core.tensor import CompressionFormat as F
from basp.core.tensor import DType, TensorKind, TensorSpec
from basp.dataflow.base import CachePolicy, DataflowMode, ExecutionPolicy
from basp.experiments.microbench import MicroSystem
from basp.experiments.modes import expected_winners, mode_study
from basp.experiments.profiles import load_topology
from basp.scheduler.critical_path import critical_path, priority_order, upward_ranks
from basp.scheduler.scheduler import SchedulingError
from tests.conftest import GPU, HOST, MOD0, MOD1, NVME, make_topology

M = DataflowMode
PARAMS = 3 * 2048 * 768  # one Qwen3-30B-A3B expert: 2,654,208 bytes at 4.5 bits/weight
W_BYTES = 2_654_208
ACT = 4096


def policy(*modes, devices=None, cache=None, joint=0) -> ExecutionPolicy:
    return ExecutionPolicy("test", frozenset(modes), frozenset(devices) if devices else None,
                           joint_rounds=joint, cache=cache or CachePolicy())  # fmt: skip


def expert(ms: MicroSystem, name: str, weight: int, home: int = GPU, params: int = PARAMS, **kw):
    x = ms.activation(f"{name}.x", ACT, home)
    y = ms.activation(f"{name}.y", ACT)
    return ms.op(OpCode.EXPERT, name, inputs=(x,), weights=(weight,), outputs=(y,), macs=params,
                 output_device=home, **kw)  # fmt: skip


# -- estimate == simulation, mode by mode ----------------------------------------------
def test_weight_stationary_matches_closed_form(topology):
    ms = MicroSystem(topology, policy(M.WEIGHT_STATIONARY))
    w = ms.weight("e", PARAMS, MOD0)
    result = ms.run([expert(ms, "e", w)])
    plan = result.plan
    hop = 7e-6 + ACT / 16e9  # GPU -> host -> module, 16 GB/s bottleneck
    kernel = 1e-6 + max(W_BYTES / 54e9, PARAMS / 256e9)  # start-up + roofline (memory bound)
    assert plan.choice.mode is M.WEIGHT_STATIONARY and plan.devices == (MOD0,)
    assert plan.estimate.input == pytest.approx(hop)
    assert plan.estimate.kernel == pytest.approx(kernel)
    assert plan.estimate.output == pytest.approx(hop)
    assert plan.estimate.queue == 0.0 and plan.estimate.sync == topology.sync_latency_s
    assert result.estimate_s == pytest.approx(2 * hop + kernel)
    assert result.actual_s == pytest.approx(result.estimate_s, rel=1e-9)
    assert result.link_bytes == pytest.approx(4 * ACT)  # two activations, two hops each
    assert not result.failures


def test_activation_stationary_pays_the_transfer_then_runs_locally(topology):
    lru = CachePolicy("lru", (GPU,), background=False)
    ms = MicroSystem(topology, policy(M.ACTIVATION_STATIONARY, cache=lru))
    w = ms.weight("e", PARAMS, MOD0)
    first = ms.run([expert(ms, "a", w)])
    transfer = 7e-6 + W_BYTES / 16e9
    kernel = 3e-6 + max(W_BYTES / 500e9, PARAMS / 20e12)
    assert first.plan.choice.retain and first.plan.devices == (GPU,)
    assert first.estimate_s == pytest.approx(transfer + kernel)
    assert first.actual_s == pytest.approx(first.estimate_s, rel=1e-9)
    assert first.plan.weight_wait == pytest.approx(transfer)
    assert ms.residency.valid(w, GPU)  # retained as a replica
    second = ms.run([expert(ms, "b", w)])
    assert second.actual_s == pytest.approx(kernel, rel=1e-9)  # resident copy: no movement
    assert second.link_bytes == 0.0 and second.plan.weight_wait == 0.0


def test_transient_load_is_freed_after_its_kernel(topology):
    ms = MicroSystem(topology, policy(M.ACTIVATION_STATIONARY))
    w = ms.weight("e", PARAMS, MOD0)
    result = ms.run([expert(ms, "a", w)])
    assert not result.plan.choice.retain and not result.failures
    assert ms.residency.copy(w, GPU) is None and ms.residency.used_bytes(GPU) == 0
    assert result.actual_s == pytest.approx(result.estimate_s, rel=1e-9)


def test_streaming_overlaps_read_link_and_compute():
    topology = make_topology(with_nvme=True)
    big = 3 * 4096 * 14336  # the section 6 expert: 99,090,432 bytes
    nbytes = 99_090_432
    times = {}
    for mode in (M.STREAMING, M.ACTIVATION_STATIONARY):
        ms = MicroSystem(topology, policy(mode, devices={MOD0}))
        w = ms.weight("big", big, NVME)
        result = ms.run([expert(ms, "e", w, home=MOD0, params=big)])
        assert result.actual_s == pytest.approx(result.estimate_s, rel=1e-9) and not result.failures
        times[mode] = result.actual_s
        assert ms.residency.copy(w, MOD0) is None  # neither path keeps the weights here
    latency = 12e-6
    assert times[M.STREAMING] == pytest.approx(1e-6 + latency + max(nbytes / 7e9, big / 256e9))
    store_forward = latency + nbytes / 7e9 + 1e-6 + max(nbytes / 54e9, big / 256e9)
    assert times[M.ACTIVATION_STATIONARY] == pytest.approx(store_forward)
    assert times[M.STREAMING] < times[M.ACTIVATION_STATIONARY]


def test_row_stationary_hits_sram_on_reuse(topology):
    ms = MicroSystem(topology, policy(M.ROW_STATIONARY))
    w = ms.weight("e", PARAMS, MOD0)
    miss = ms.run([expert(ms, "a", w)])
    hit = ms.run([expert(ms, "b", w)])
    assert M.ROW_STATIONARY in miss.plan.choice.modes
    assert not miss.plan.shards[0].sram_hit and hit.plan.shards[0].sram_hit
    saved = W_BYTES / 54e9 - PARAMS / 256e9  # memory-bound -> compute-bound
    assert miss.actual_s - hit.actual_s == pytest.approx(saved, rel=1e-6)
    for result in (miss, hit):
        assert result.actual_s == pytest.approx(result.estimate_s, rel=1e-9)
    # an expert larger than the SRAM cannot use the mode and falls back to another path
    small = MicroSystem(make_topology(sram=1 << 20), policy(M.ROW_STATIONARY))
    w2 = small.weight("e", PARAMS, MOD0)
    assert M.ROW_STATIONARY not in small.run([expert(small, "a", w2)]).plan.choice.modes


def test_distributed_plan_splits_a_sharded_projection(topology):
    ms = MicroSystem(topology, policy(M.DISTRIBUTED))
    rows, hidden = 4096, 2048
    shards = [ms.weight(f"h{i}", rows * hidden, dev) for i, dev in enumerate((MOD0, MOD1))]
    x = ms.activation("x", ACT, GPU)
    logits = ms.activation("logits", 2 * rows * 2)
    op = ms.op(OpCode.MATVEC, "head", inputs=(x,), weights=shards, outputs=(logits,),
               macs=2 * rows * hidden, output_device=GPU, partitionable=True)  # fmt: skip
    result = ms.run([op])
    plan = result.plan
    assert plan.choice.mode is M.DISTRIBUTED and plan.devices == (MOD0, MOD1)
    assert [s.kernel.macs for s in plan.shards] == [rows * hidden, rows * hidden]
    # the two shards share the GPU<->host hop, which the estimator serialises: near-exact
    assert result.actual_s == pytest.approx(result.estimate_s, rel=5e-3) and not result.failures
    assert ms.residency.valid(logits, GPU)


def test_staged_decode_and_transcode_on_arrival(topology):
    """Entropy-coded copies: the GPU decodes once on arrival; the CPU needs a staged pass."""
    n = 4_000_000
    lru = CachePolicy("lru", (GPU,), background=False)
    ms = MicroSystem(topology, policy(M.ACTIVATION_STATIONARY, cache=lru))
    w = ms.weight("e", n, HOST, stored_as=F.ENTROPY)
    coded, raw = ms.registry.nbytes(w, F.ENTROPY), ms.registry.nbytes(w, F.INT4)
    assert coded < raw
    result = ms.run([expert(ms, "a", w, params=n)])
    moving = max(coded / 24e9, coded / 40e9, raw / 500e9)  # link, GPU decoder, GPU memory write
    assert result.plan.shards[0].inputs[0].nbytes == coded  # compressed bytes cross the link
    assert result.actual_s == pytest.approx(5e-6 + moving + 3e-6 + raw / 500e9, rel=1e-9)
    assert result.actual_s == pytest.approx(result.estimate_s, rel=1e-9)
    assert ms.residency.copy(w, GPU).fmt is F.INT4  # stored decoded: later uses pay no decode
    assert result.link_bytes == pytest.approx(coded)

    cpu = MicroSystem(topology, policy(M.WEIGHT_STATIONARY, devices={HOST}))
    w2 = cpu.weight("e", n, HOST, stored_as=F.ENTROPY)
    staged = cpu.run([expert(cpu, "a", w2, home=HOST, params=n)])
    assert len(staged.plan.shards[0].kernel.phases) == 2  # decode pass, then compute pass
    decode = max((coded + raw) / 54e9, coded / 4e9)
    compute = max(raw / 54e9, n / 30e9)
    assert staged.actual_s == pytest.approx(0.5e-6 + decode + compute, rel=1e-9)

    module = MicroSystem(topology, policy(M.WEIGHT_STATIONARY, devices={MOD0}))
    w3 = module.weight("e", n, MOD0, stored_as=F.ENTROPY)
    fused = module.run([expert(module, "a", w3, home=MOD0, params=n)])
    assert len(fused.plan.shards[0].kernel.phases) == 1  # fused decode -> MAC pipeline
    assert fused.actual_s == pytest.approx(1e-6 + max(coded / 54e9, coded / 8e9, n / 256e9), rel=1e-9)


def test_section_6_numbers_are_reproduced_by_the_simulator():
    """The specification's example: 94.5 MiB expert, 24 GB/s link, 54 GB/s, 128 GMAC/s."""
    import math

    from basp.core.capabilities import DeviceCapabilities, DeviceKind
    from basp.core.topology import DeviceSpec, LinkSpec, Topology
    from basp.experiments.analytic import compare_paths

    caps = DeviceCapabilities(decode_rate={F.INT4: math.inf}, stream_formats=frozenset({F.INT4}))
    topology = Topology(
        [DeviceSpec(0, "gpu", DeviceKind.GPU, 1 << 30, 1e15, 1e15, caps),
         DeviceSpec(1, "near", DeviceKind.NEAR_MEMORY, 1 << 30, 54e9, 128e9, caps)],
        [LinkSpec(0, 1, 24e9)],
    )  # fmt: skip
    expected = compare_paths()
    params = 3 * 4096 * 14336
    near = MicroSystem(topology, policy(M.WEIGHT_STATIONARY))
    w = near.weight("e", params, 1)
    x = near.activation("x", 8192, 0)
    y = near.activation("y", 8192)
    op = near.op(OpCode.EXPERT, "e", inputs=(x,), weights=(w,), outputs=(y,), macs=params,
                 output_device=0)  # fmt: skip
    result = near.run([op])
    assert result.actual_s == pytest.approx(expected.near_memory_s, rel=1e-9)
    assert result.actual_s * 1e3 == pytest.approx(1.836, abs=5e-4)
    assert result.link_bytes == expected.activation_bytes == 16 * 1024

    move = MicroSystem(topology, policy(M.ACTIVATION_STATIONARY))
    w = move.weight("e", params, 1)
    x = move.activation("x", 8192, 0)
    y = move.activation("y", 8192)
    op = move.op(OpCode.EXPERT, "e", inputs=(x,), weights=(w,), outputs=(y,), macs=params,
                 output_device=0)  # fmt: skip
    plan = move.run([op]).plan
    assert plan.estimate.input == pytest.approx(expected.cold_transfer_s, rel=1e-9)
    assert plan.estimate.input * 1e3 == pytest.approx(4.129, abs=5e-4)
    assert plan.link_bytes / result.link_bytes == pytest.approx(6048)


# -- mode selection ---------------------------------------------------------------------
def test_each_mode_wins_in_its_own_regime():
    topology = load_topology("basp_4module")
    for scenario in mode_study(topology):
        assert scenario.winner == expected_winners()[scenario.key], scenario


def test_adaptive_policy_picks_the_cheaper_path_per_situation(topology):
    adaptive = policy(*M, cache=CachePolicy("lru", (GPU,), background=False))
    ms = MicroSystem(topology, adaptive)
    cold, hot = ms.weight("cold", PARAMS, MOD0), ms.weight("hot", PARAMS, MOD0)
    ms.replicate(hot, GPU)
    assert ms.run([expert(ms, "a", cold)]).plan.devices == (MOD0,)  # move the computation
    assert ms.run([expert(ms, "b", hot)]).plan.devices == (GPU,)  # use the resident copy


def test_scheduler_never_violates_pinning_or_device_restrictions(topology):
    ms = MicroSystem(topology, policy(M.WEIGHT_STATIONARY, M.ACTIVATION_STATIONARY, devices={GPU}))
    w = ms.weight("e", PARAMS, MOD0)
    assert ms.run([expert(ms, "a", w)]).plan.devices == (GPU,)  # the module may not compute
    pinned = MicroSystem(topology, policy(*M))
    w = pinned.weight("e", PARAMS, MOD0)
    pinned.replicate(w, GPU)
    assert pinned.run([expert(pinned, "a", w, pinned_device=MOD0)]).plan.devices == (MOD0,)
    stuck = MicroSystem(topology, policy(M.WEIGHT_STATIONARY, devices={HOST}))
    w = stuck.weight("e", PARAMS, NVME if NVME in topology.devices else MOD0)
    stuck_op = expert(stuck, "a", w, home=HOST)
    stuck.scheduler.cost.compute_devices = frozenset()  # nothing may compute at all
    with pytest.raises(SchedulingError):
        stuck.run([stuck_op])


def test_policy_is_relaxed_rather_than_failing_when_no_enabled_mode_applies(topology):
    ms = MicroSystem(topology, policy(M.WEIGHT_STATIONARY, devices={GPU}))
    w = ms.weight("e", PARAMS, MOD0)  # weights are not on any allowed device
    result = ms.run([expert(ms, "a", w)])
    assert result.plan.devices == (GPU,) and not result.failures
    assert ms.scheduler.stats["relaxed"] == 1


# -- joint scheduling --------------------------------------------------------------------
def contended(joint: int):
    """Two experts on the host; the only fast alternative shares one link."""
    topology = make_topology()
    ms = MicroSystem(topology, policy(M.WEIGHT_STATIONARY, M.ACTIVATION_STATIONARY,
                                      devices={HOST, GPU}, joint=joint))  # fmt: skip
    big = 40_000_000
    ops = [expert(ms, f"e{i}", ms.weight(f"w{i}", big, HOST), params=big) for i in range(3)]
    return ms, ms.run(ops)


def test_joint_refinement_never_loses_to_greedy_and_estimates_hold_under_contention():
    _, greedy = contended(joint=0)
    _, joint = contended(joint=3)
    assert joint.estimate_s <= greedy.estimate_s * (1 + 1e-12)
    assert joint.actual_s <= greedy.actual_s * 1.02
    for result in (greedy, joint):
        assert not result.failures
        # link sharing is modelled as serialisation by the estimator: within a few percent
        assert result.actual_s == pytest.approx(result.estimate_s, rel=0.06)


def test_joint_refinement_fixes_a_greedy_collision():
    """Greedy queues both experts on one module; evaluating the whole batch splits them."""
    topology = make_topology()
    results = {}
    for joint in (0, 2):
        ms = MicroSystem(topology, policy(M.WEIGHT_STATIONARY, joint=joint))
        heavy = 200_000_000
        a, b = ms.weight("a", heavy, MOD0), ms.weight("b", heavy, MOD0)
        ms.replicate(a, MOD1)
        # The flexible op is planned first (equal rank, lower id) and greedily takes the
        # lower-numbered idle module -- the only one the constrained op can use.
        flexible = expert(ms, "flexible", a)  # may run on either module
        constrained = expert(ms, "constrained", b)  # can only run on mod0
        result = ms.run([flexible, constrained])
        results[joint] = (result, ms.scheduler.stats["joint_improvements"])
    greedy, joint = results[0][0], results[2][0]
    assert joint.actual_s <= greedy.actual_s * (1 + 1e-9)
    assert joint.actual_s == pytest.approx(joint.estimate_s, rel=1e-3)
    assert results[2][1] >= 1 and joint.actual_s < greedy.actual_s * 0.75


def test_scheduling_is_deterministic():
    def run():
        _, result = contended(joint=2)
        return [(p.op.name, p.devices, p.choice.mode, round(p.finish, 15)) for p in result.plans]

    assert run() == run()


# -- critical path -----------------------------------------------------------------------
def test_upward_ranks_and_priority_order():
    g = Graph()
    for tid in range(1, 8):
        g.add_tensor(TensorSpec(tid, f"t{tid}", TensorKind.ACTIVATION, (1,), DType.FP16, 2))
    g.add_op(OpNode(1, OpCode.ADD, "src", inputs=(1,), outputs=(2,)))
    g.add_op(OpNode(2, OpCode.ADD, "long", inputs=(2,), outputs=(3,)))
    g.add_op(OpNode(3, OpCode.ADD, "short", inputs=(2,), outputs=(4,)))
    g.add_op(OpNode(4, OpCode.ADD, "join", inputs=(3, 4), outputs=(5,)))
    cost = {1: 1.0, 2: 5.0, 3: 1.0, 4: 2.0}
    ranks = upward_ranks(g, cost, cost.__getitem__)
    assert ranks == {4: 2.0, 2: 7.0, 3: 3.0, 1: 8.0}
    assert priority_order(g, cost, cost.__getitem__) == [1, 2, 3, 4]  # critical branch first
    assert critical_path(g, cost, cost.__getitem__) == ([1, 2, 4], 8.0)
    zero = dict.fromkeys(cost, 0.0)
    assert priority_order(g, zero, zero.__getitem__) == [1, 2, 3, 4]  # still topological
    assert critical_path(g, [], cost.__getitem__) == ([], 0.0)
