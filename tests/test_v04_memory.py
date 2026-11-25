"""v0.4 memory domains: shared accounting, the budget invariant, admission control, safe
eviction and migration, expert-cache policies, and the unified-memory design repair."""

from __future__ import annotations

import numpy as np
import pytest

from basp.dse.hardware import DesignError, HardwareDesign, shares_host_memory
from basp.experiments.explore import design_space, evaluate_design, seed_designs
from basp.experiments.profiles import load_json, load_topology
from basp.memory import (
    POLICIES,
    AdmissionController,
    AdmissionRefused,
    Category,
    EvictionUnsafe,
    ExpertCost,
    MemoryDomain,
    MemoryHierarchy,
    MemoryMap,
    MigrationEngine,
    Prefetcher,
    RequestShape,
    Tier,
    compare,
    simulate,
    trace_steps,
)
from basp.memory.allocator import OutOfMemory
from basp.workloads.trace import synth_trace

MiB = 1 << 20
W, KV, ACT, WS = Category.WEIGHTS, Category.KV, Category.ACTIVATIONS, Category.WORKSPACE


# -- domains --------------------------------------------------------------------------------------
def test_devices_sharing_a_domain_share_one_pool():
    m2 = MemoryMap.from_topology(load_topology("macbook_m2"), reserved_fraction=0.25)
    host, gpu = m2.domain_of("host"), m2.domain_of("gpu")
    assert host is gpu and m2.shared("host", "gpu") and len(m2.domains) == 1
    assert m2.physical_bytes == host.capacity  # counted once, not once per device
    host.allocate(1 * MiB, W, "cpu weights", device="host")
    gpu.allocate(2 * MiB, KV, "gpu kv", device="gpu")
    assert host.used == gpu.used == 3 * MiB
    desktop = MemoryMap.from_topology(load_topology("desktop_gpu12"))
    assert not desktop.shared("host", "gpu") and len(desktop.domains) == 2
    with pytest.raises(ValueError, match="does not use memory domain"):
        desktop.domain_of("gpu").allocate(MiB, W, "x", device="host")


def test_the_budget_invariant_holds_under_random_churn():
    rng = np.random.default_rng(0)
    domain = MemoryDomain("gpu", 64 * MiB, reserved=8 * MiB, limits={KV: 24 * MiB}, alignment=4096)
    live = []
    for _ in range(2000):
        if live and rng.random() < 0.45:
            domain.free_block(live.pop(rng.integers(len(live))))
        else:
            category = list(Category)[rng.integers(4)]
            size = int(rng.integers(1, 4 * MiB))
            try:
                live.append(domain.allocate(size, category, f"t{len(live)}"))
            except (OutOfMemory, AdmissionRefused):
                pass
        domain.check_invariant()
        assert sum(domain.used_by(c) for c in Category) + domain.reserved <= domain.capacity
        assert domain.used_by(KV) <= 24 * MiB
        assert 0.0 <= domain.fragmentation <= 1.0
    assert domain.peak <= domain.usable
    with pytest.raises(ValueError):
        MemoryDomain("bad", MiB, reserved=2 * MiB)


def test_referenced_or_pinned_blocks_are_never_evicted():
    domain = MemoryDomain("gpu", 8 * MiB)
    busy = domain.retain(domain.allocate(3 * MiB, W, "expert-busy"))
    pinned = domain.allocate(2 * MiB, W, "embeddings", pinned=True)
    idle = domain.allocate(3 * MiB, W, "expert-idle")
    with pytest.raises(EvictionUnsafe, match="still referenced"):
        domain.free_block(busy)
    with pytest.raises(EvictionUnsafe, match="pinned"):
        domain.free_block(pinned)
    assert domain.evict(6 * MiB) == []  # unreachable without the busy block: evict nothing
    assert domain.used == 8 * MiB
    assert [b.owner for b in domain.evict(3 * MiB)] == ["expert-idle"]
    domain.release(busy, free_when_unused=True)
    assert domain.used == 2 * MiB and domain.blocks()[0] is pinned
    with pytest.raises(ValueError, match="not a live block"):
        domain.free_block(idle)
    domain.check_invariant()


def test_admission_reserves_memory_evicts_cache_first_and_never_overcommits():
    domain = MemoryDomain("gpu", 100 * MiB, reserved=10 * MiB)
    control = AdmissionController(domain)
    cache = [domain.allocate(10 * MiB, W, f"expert{i}") for i in range(6)]  # 60 MiB of cached experts
    shape = RequestShape(prompt_tokens=512, max_new_tokens=512, kv_bytes_per_token=20 * 1024,
                         activation_bytes=2 * MiB, workspace_bytes=MiB)  # 23 MiB
    first = control.admit("r1", shape)
    assert first.admitted and not first.evicted  # 30 MiB were free
    second = control.admit("r2", shape)
    assert second.admitted and second.evicted  # cached experts made room
    assert all(b.refs == 1 for b in second.blocks)  # reserved for the request's lifetime
    huge = control.admit("r3", RequestShape(0, 10_000, 20 * 1024))
    assert not huge.admitted and "needs" in huge.reason
    big = control.admit("r4", RequestShape(1024, 1024, 20 * 1024))  # 40 MiB: only by evicting more
    domain.check_invariant()
    assert domain.used <= domain.usable
    if big.admitted:
        control.finish("r4")
    control.finish("r1")
    control.finish("r2")
    assert domain.used_by(KV) == 0 and len(domain.blocks(W)) <= len(cache)
    assert control.state() == "normal"


def test_pressure_feedback_keeps_a_margin_under_pressure():
    domain = MemoryDomain("gpu", 100 * MiB)
    control = AdmissionController(domain, elevated=0.5, critical=0.9)
    domain.retain(domain.allocate(60 * MiB, W, "resident weights"))
    assert control.state() == "elevated"
    # 35 MiB would fit (40 free), but would leave less than the 10% critical margin
    refused = control.admit("r", RequestShape(0, 35, MiB))
    assert not refused.admitted and "margin" in refused.reason
    assert control.admit("ok", RequestShape(0, 25, MiB)).admitted


def test_migration_holds_its_source_and_hides_its_destination_until_done():
    host = MemoryDomain("host", 64 * MiB)
    gpu = MemoryDomain("gpu", 16 * MiB)
    engine = MigrationEngine(bandwidth_bytes_per_s=1e9, latency_s=1e-5)
    src = host.allocate(8 * MiB, W, "expert7")
    m = engine.start(host, src, gpu, now=0.0)
    assert src.refs == 1 and m.destination.refs == 1
    with pytest.raises(EvictionUnsafe):
        host.free_block(src)  # being read
    assert gpu.evict(16 * MiB) == []  # the half-written copy cannot be evicted
    assert engine.poll(m.done_s / 2) == [] and m.state == "in_flight"
    second = engine.start(host, host.allocate(4 * MiB, W, "expert8"), gpu, now=0.0)
    assert second.start_s == pytest.approx(m.done_s)  # one link: copies are serialised
    assert engine.poll(second.done_s) == [m, second]
    assert src.refs == 0 and m.destination.evictable and engine.bytes_moved == 12 * MiB
    assert len(gpu.evict(16 * MiB)) == 2  # complete copies are ordinary, evictable blocks
    moved = engine.start(host, src, gpu, now=1.0, move=True)
    with pytest.raises(OutOfMemory):
        engine.start(host, host.allocate(12 * MiB, W, "x"), gpu, now=1.0)
    engine.cancel(moved)
    assert src.refs == 0 and gpu.blocks() == [] and host.blocks()[0] is src
    done = engine.start(host, src, gpu, now=2.0, move=True)
    engine.wait(done)
    assert src not in host.blocks() and done.destination in gpu.blocks()  # moved, not copied
    host.check_invariant()
    gpu.check_invariant()


def test_prefetch_respects_its_budget_and_reports_waste():
    host, gpu = MemoryDomain("host", 64 * MiB), MemoryDomain("gpu", 16 * MiB)
    engine = MigrationEngine(bandwidth_bytes_per_s=1e9)
    pre = Prefetcher(engine, host, gpu, budget_bytes=8 * MiB)
    blocks = {f"e{i}": host.allocate(4 * MiB, W, f"e{i}") for i in range(4)}
    started = pre.issue(blocks.items(), resident={"e0"}, now=0.0)
    assert [m.destination.owner for m in started] == ["e1", "e2"] and pre.stats.skipped_budget == 1
    assert pre.claim("e1") is not None and pre.claim("e3") is None
    assert pre.discard_unused() == 4 * MiB and pre.stats.wasted_bytes == 4 * MiB
    assert all(b.refs == 0 for b in blocks.values())


def test_hierarchy_places_in_the_fastest_tier_with_room():
    vram, dram = MemoryDomain("vram", 8 * MiB), MemoryDomain("dram", 64 * MiB)
    h = MemoryHierarchy([Tier(vram, 900e9, 25e9), Tier(dram, 50e9)])
    assert h.fastest_fit(4 * MiB) == 0
    vram.allocate(6 * MiB, W, "resident")
    assert h.fastest_fit(4 * MiB) == 1 and h.fastest_fit(128 * MiB) is None
    assert h.promotion_time(25 * 10**6, 1, 0) == pytest.approx(1e-3)
    with pytest.raises(ValueError):
        MemoryHierarchy([Tier(vram, 900e9), Tier(dram, 50e9)])


# -- expert caching policies --------------------------------------------------------------------
def costs_for(size=6 * MiB):
    cost = ExpertCost.from_rates(size, gpu_bandwidth=600e9, cpu_bandwidth=40e9, link_bandwidth=20e9,
                                 link_latency_s=1e-5)  # fmt: skip
    return lambda key: cost


def test_cache_policies_are_bounded_by_cpu_only_and_an_unbounded_cache():
    trace = synth_trace(200, 4, 64, 4, zipf_s=1.1, stickiness=0.6, seed=2)
    steps = trace_steps(trace)
    costs = costs_for()
    capacity = 48 * 6 * MiB  # a fifth of the experts
    results = {r.policy: r for r in compare(steps, capacity, costs)}
    cpu, unbounded = results["none (CPU only)"], results["unbounded cache"]
    for name in POLICIES:
        r = results[name]
        assert r.activations == cpu.activations == sum(len(s) for s in steps)
        assert r.hits + r.migrations + r.cpu_executions == r.activations
        assert r.peak_bytes <= capacity
        assert r.time_s >= unbounded.time_s * 0.999
    # with a skewed, sticky trace, valuing experts beats caching every miss
    assert results["cost-aware"].time_s < results["LRU"].time_s
    assert results["predictive"].time_s <= results["cost-aware"].time_s * 1.05
    assert results["cost-aware"].bytes_migrated < results["LRU"].bytes_migrated


def test_cost_aware_never_migrates_when_the_cpu_is_cheaper():
    expensive_link = ExpertCost(6 * MiB, t_gpu=1e-5, t_cpu=1e-4, t_migration=1.0)
    steps = [[(0, i % 3)] for i in range(50)]
    r = simulate(steps, POLICIES["cost-aware"](horizon=8), 64 * MiB, lambda k: expensive_link)
    assert r.migrations == 0 and r.cpu_executions == 50
    lru = simulate(steps, POLICIES["LRU"](), 64 * MiB, lambda k: expensive_link)
    assert lru.migrations == 3 and lru.time_s > r.time_s


def test_a_step_never_evicts_an_expert_it_uses():
    steps = [[(0, 0), (0, 1)], [(0, 1), (0, 2)], [(0, 0), (0, 2)]]
    for name, policy in POLICIES.items():
        r = simulate(steps, policy() if name in ("LRU", "LFU") else policy(horizon=1000),
                     2 * 6 * MiB, costs_for())  # fmt: skip
        assert r.peak_bytes <= 2 * 6 * MiB


# -- unified-memory design repair ----------------------------------------------------------------
def test_designs_on_a_unified_memory_machine_share_one_pool():
    estimates = load_json("design_estimates")
    m2 = load_topology("macbook_m2")
    assert shares_host_memory(m2) and not shares_host_memory(load_topology("desktop_gpu12"))
    with pytest.raises(DesignError, match="set integrated_gpu"):
        HardwareDesign(gpu_gib=12).topology(m2, estimates)
    with pytest.raises(DesignError, match="gpu_gib must be 0"):
        HardwareDesign(gpu_gib=12, host_gib=64, integrated_gpu=True)
    design = HardwareDesign(gpu_gib=64, host_gib=64, integrated_gpu=True, modules=2)
    topo = design.topology(m2, estimates)  # v0.3: ValueError, the domain was split
    host, gpu = topo.devices[topo.host], next(d for d in topo.devices.values() if d.name == "gpu")
    assert host.memory_domain == gpu.memory_domain == "unified"
    assert host.mem_capacity == gpu.mem_capacity == 64 * 2**30 and host.mem_bandwidth == gpu.mem_bandwidth
    # an integrated GPU is not bought separately
    assert design.cost_usd(estimates) == HardwareDesign(gpu_gib=0, host_gib=64, modules=2).cost_usd(estimates)
    space = design_space(17.0, gpu_sizes=(0, 64))
    baseline = next(c for c in seed_designs(space) if c["arch"] == "A")
    assert evaluate_design(baseline, base_profile="macbook_m2") is not None
