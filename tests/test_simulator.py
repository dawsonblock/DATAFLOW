"""Discrete-event kernel, fluid bandwidth model, device executors and interconnect.

The v0.1 evidence requirement is "verified event and bandwidth accounting":
these tests check completion times against closed-form results and check that
work is conserved under arbitrary contention.
"""

from __future__ import annotations

import math
import random

import pytest

from basp.core.topology import Topology
from basp.simulator.compute import (
    DeviceExecutor,
    Kernel,
    Phase,
    compute_key,
    kernel_time,
    mem_key,
)
from basp.simulator.dram import (
    TileCache,
    ddr_bandwidth,
    ddr_channel_bandwidth,
    lane_mac_rate,
    stream_mac_rate,
)
from basp.simulator.energy import energy_report
from basp.simulator.engine import SimulationError, Simulator
from basp.simulator.event_queue import EventQueue
from basp.simulator.fluid import FluidNetwork
from basp.simulator.interconnect import Interconnect
from tests.conftest import GPU, HOST, MOD0, MOD1


# -- event queue / kernel ------------------------------------------------------------
def test_events_fire_in_time_then_scheduling_order():
    sim = Simulator()
    seen: list[str] = []
    sim.at(2.0, seen.append, "c")
    sim.at(1.0, seen.append, "a")
    sim.at(1.0, seen.append, "b")
    doomed = sim.at(1.5, seen.append, "x")
    sim.cancel(doomed)
    assert sim.pending == 3
    assert sim.run() == 2.0
    assert seen == ["a", "b", "c"] and sim.events_processed == 3


def test_simulator_rejects_time_travel_and_bounds_runaway_loops():
    sim = Simulator()
    sim.at(1.0, lambda: None)
    sim.run()
    with pytest.raises(SimulationError):
        sim.at(0.5, lambda: None)
    with pytest.raises(SimulationError):
        sim.after(-1.0, lambda: None)

    def again():
        sim.after(0.0, again)

    sim.after(0.0, again)
    with pytest.raises(SimulationError, match="budget"):
        sim.run(max_events=100)


def test_run_until_stops_the_clock_between_events():
    sim = Simulator()
    seen = []
    sim.at(5.0, seen.append, 1)
    assert sim.run(until=2.0) == 2.0 and not seen
    assert sim.run() == 5.0 and seen == [1]
    queue = EventQueue()
    assert queue.pop() is None and queue.peek_time() is None


# -- fluid model -----------------------------------------------------------------------
def net() -> tuple[Simulator, FluidNetwork]:
    sim = Simulator()
    return sim, FluidNetwork(sim)


def test_single_flow_takes_the_roofline_time():
    """Section 6: W = 94.5 MiB at 54 GB/s, C = 176.2 M MAC at 128 GMAC/s -> 1.835 ms."""
    sim, n = net()
    mem, mac = n.resource("mem", 54e9), n.resource("mac", 128e9, "mac")
    weight_bytes, macs = 99_090_432, 176_160_768
    done = []
    n.start({mem: weight_bytes, mac: macs}, lambda f: done.append(sim.now))
    sim.run()
    assert done == [pytest.approx(max(weight_bytes / 54e9, macs / 128e9), rel=1e-12)]
    assert done[0] == pytest.approx(1.835e-3, rel=1e-3)
    assert mem.served == weight_bytes and mac.served == macs
    # the memory channel is the bottleneck (fully used); the MAC array has headroom
    assert mem.utilisation(sim.now) == pytest.approx(1.0)
    assert mac.utilisation(sim.now) == pytest.approx((macs / 128e9) / (weight_bytes / 54e9))


def test_equal_flows_share_a_resource_fairly():
    sim, n = net()
    link = n.resource("link", 10.0)
    times = {}
    n.start({link: 10.0}, lambda f: times.setdefault("small", sim.now))
    n.start({link: 30.0}, lambda f: times.setdefault("large", sim.now))
    sim.run()
    # both run at 5/s until the small one finishes at t=2; the large one then gets 10/s
    assert times == {"small": pytest.approx(2.0), "large": pytest.approx(4.0)}
    assert link.served == pytest.approx(40.0) and link.busy_time == pytest.approx(4.0)


def test_strict_priority_and_rate_caps_bound_background_traffic():
    sim, n = net()
    link = n.resource("link", 10.0)
    times = {}
    n.start({link: 20.0}, lambda f: times.setdefault("demand", sim.now))
    n.start({link: 5.0}, lambda f: times.setdefault("background", sim.now), priority=1)
    # 4 units of work capped at 0.25 progress/s -> 1 unit/s
    n.start({link: 4.0}, lambda f: times.setdefault("capped", sim.now), priority=1, max_rate=0.25)
    sim.run()
    assert times["demand"] == pytest.approx(2.0)  # never slowed by lower classes
    assert times["background"] > 2.0 and times["capped"] >= 2.0 + 4.0 - 1e-9
    assert link.served == pytest.approx(29.0)


def test_coupled_flows_use_dominant_resource_fairness():
    """Flows are throttled equally relative to their own roofline on a saturated resource."""
    sim, n = net()
    a, b = n.resource("a", 10.0), n.resource("b", 10.0)
    done = {}
    n.start({a: 10.0, b: 5.0}, lambda f: done.setdefault("x", sim.now))  # alone: 1 s, 5/s of b
    n.start({b: 10.0}, lambda f: done.setdefault("y", sim.now))  # alone: 1 s, 10/s of b
    sim.run()
    # together they ask 15/s of b; both run at 2/3 of their solo rate until b's work is done
    assert done["x"] == pytest.approx(1.5) and done["y"] == pytest.approx(1.5)
    assert b.served == pytest.approx(15.0) and b.utilisation(sim.now) == pytest.approx(1.0)
    assert a.utilisation(sim.now) == pytest.approx(2 / 3)

    sim, n = net()
    a, b, c = n.resource("a", 10.0), n.resource("b", 10.0), n.resource("c", 6.0)
    done = {}
    n.start({a: 10.0, b: 2.0}, lambda f: done.setdefault("x", sim.now))  # 2/s of b
    n.start({b: 6.0, c: 6.0}, lambda f: done.setdefault("y", sim.now))  # 6/s of b
    sim.run()
    # 2/s + 6/s fits within b: nothing is saturated, both keep their roofline
    assert done == {"x": pytest.approx(1.0), "y": pytest.approx(1.0)}


def test_cancel_and_release_keep_accounting_exact():
    sim, n = net()
    link, budget = n.resource("link", 10.0), n.resource("budget", 2.0, "budget")
    slow = n.start({link: 20.0, budget: 20.0}, tag="bg")  # alone: 2/s (budget bound)
    victim = n.start({link: 40.0}, tag="victim")  # alone: 10/s
    sim.run(until=1.0)
    # they ask 12/s of the link, so each runs at 10/12 of its solo rate
    assert n.cancel(victim) == pytest.approx((10.0 * 10 / 12) / 40.0)
    n.release(slow, [budget])  # promotion: the budget no longer applies
    sim.run()
    slow_first_second = 2.0 * 10 / 12
    assert slow.finished_at == pytest.approx(1.0 + (20.0 - slow_first_second) / 10.0)
    assert link.served == pytest.approx(20.0 + 10.0 * 10 / 12)
    assert link.by_tag == {"bg": pytest.approx(20.0), "victim": pytest.approx(100 / 12)}
    assert budget.served == pytest.approx(slow_first_second)
    assert n.cancel(victim) == pytest.approx((100 / 12) / 40.0)  # idempotent


def test_degenerate_flows_and_invalid_arguments():
    sim, n = net()
    free = n.resource("free", math.inf)
    link = n.resource("link", 1.0)
    done = []
    n.start({free: 123.0}, lambda f: done.append("free"))
    n.start({}, lambda f: done.append("empty"))
    sim.run()
    assert done == ["free", "empty"] and sim.now == 0.0 and free.served == 123.0
    with pytest.raises(ValueError):
        n.start({link: -1.0})
    with pytest.raises(ValueError):
        n.start({link: 1.0}, max_rate=0.0)
    with pytest.raises(ValueError):
        n.resource("link", 2.0)
    with pytest.raises(ValueError):
        n.resource("zero", 0.0)


@pytest.mark.parametrize("seed", range(5))
def test_work_is_conserved_under_random_contention(seed: int):
    """Total work served equals total work requested and never exceeds capacity x time."""
    rng = random.Random(seed)
    sim, n = net()
    resources = [n.resource(f"r{i}", rng.uniform(1.0, 50.0)) for i in range(5)]
    requested = {r: 0.0 for r in resources}
    finished = []
    flows = []

    def launch():
        demands = {r: rng.uniform(0.1, 20.0) for r in rng.sample(resources, rng.randint(1, 3))}
        for r, w in demands.items():
            requested[r] += w
        cap = rng.choice([None, None, rng.uniform(0.05, 2.0)])
        flows.append(n.start(demands, finished.append, priority=rng.randint(0, 2), max_rate=cap))

    for _ in range(40):
        sim.at(rng.uniform(0.0, 5.0), launch)
    sim.run()
    assert len(finished) == 40 and n.active_flows == 0
    for r in resources:
        assert r.served == pytest.approx(requested[r], rel=1e-9)
        assert r.served <= r.capacity * sim.now * (1 + 1e-9)
        assert r.busy_time <= sim.now * (1 + 1e-9)
        assert 0.0 <= r.utilisation(sim.now) <= 1.0 + 1e-9
    for flow in flows:
        # no flow can beat its uncontended roofline
        roofline = max(w / r.capacity for r, w in flow.demands)
        assert flow.finished_at - flow.started_at >= roofline * (1 - 1e-9)


def test_simulation_is_deterministic():
    def run() -> list[float]:
        rng = random.Random(42)
        sim, n = net()
        res = [n.resource(f"r{i}", rng.uniform(1, 9)) for i in range(3)]
        out: list[float] = []
        for _ in range(25):
            demands = {r: rng.uniform(1, 5) for r in rng.sample(res, 2)}
            sim.at(rng.uniform(0, 3), lambda d=demands: n.start(d, lambda f: out.append(sim.now)))
        sim.run()
        return out

    assert run() == run()


# -- memory helpers ----------------------------------------------------------------------
def test_ddr_and_lane_arithmetic_matches_the_specification():
    assert ddr_channel_bandwidth(5600) == 44.8e9
    assert ddr_bandwidth(5600, 2) == 89.6e9
    assert stream_mac_rate(89.6e9, 4.5) == pytest.approx(159.3e9, rel=1e-3)
    assert lane_mac_rate(1024, 250e6) == 256e9
    with pytest.raises(ValueError):
        ddr_bandwidth(5600, 2, efficiency=1.5)


def test_tile_cache_is_lru_within_capacity():
    cache = TileCache(100)
    assert cache.insert(1, 40) == [] and cache.insert(2, 40) == []
    cache.touch(1)
    assert cache.insert(3, 40) == [2]  # 2 is least recently used
    assert 1 in cache and 3 in cache and 2 not in cache and cache.used == 80
    assert cache.insert(4, 500) == [] and 4 not in cache  # larger than the cache: not stored
    cache.discard(1)
    assert cache.used == 40 and cache.contains_all([3]) and not cache.contains_all([1, 3])


# -- device executor ---------------------------------------------------------------------
def executor(topology: Topology, device: int):
    sim = Simulator()
    n = FluidNetwork(sim)
    dev = topology.devices[device]
    res = {mem_key(device): n.resource("mem", dev.mem_bandwidth),
           compute_key(device): n.resource("mac", dev.compute_rate, "mac")}  # fmt: skip
    return sim, DeviceExecutor(sim, n, topology, dev, res.get), res


def test_kernels_queue_fifo_and_staged_kernels_add_their_phases(topology: Topology):
    sim, ex, res = executor(topology, MOD0)
    fused = Kernel(MOD0, (Phase(((mem_key(MOD0), 54e6), (compute_key(MOD0), 128e6))),), startup_s=1e-6)
    staged = Kernel(MOD0, (Phase(((mem_key(MOD0), 54e6),)), Phase(((compute_key(MOD0), 512e6),))))
    assert kernel_time(topology, fused) == pytest.approx(1e-6 + 1e-3)  # max(1 ms, 0.5 ms)
    assert kernel_time(topology, staged) == pytest.approx(1e-3 + 2e-3)  # sum of the two passes
    done = []
    ex.submit(fused, lambda j: done.append(("fused", sim.now)))
    ex.submit(staged, lambda j: done.append(("staged", sim.now)))
    assert ex.queue_length == 2 and ex.backlog_s() == pytest.approx(1e-6 + 4e-3)
    sim.run()
    assert done[0] == ("fused", pytest.approx(1.001e-3))
    assert done[1] == ("staged", pytest.approx(1.001e-3 + 3e-3))
    assert ex.kernels_run == 2 and ex.max_queue == 1 and ex.wait_time == pytest.approx(1.001e-3)
    assert res[mem_key(MOD0)].served == pytest.approx(108e6)


def test_sram_hit_skips_external_memory(topology: Topology):
    sim, ex, res = executor(topology, MOD0)
    miss = (Phase(((mem_key(MOD0), 54e6), (compute_key(MOD0), 25.6e6))),)
    hit = (Phase(((compute_key(MOD0), 25.6e6),)),)
    kernel = Kernel(MOD0, miss, sram_tensors=((7, 1 << 20),), sram_hit_phases=hit)
    times = []
    ex.submit(kernel, lambda j: times.append(sim.now))
    sim.run()
    ex.submit(kernel, lambda j: times.append(sim.now))
    sim.run()
    assert times[0] == pytest.approx(1e-3)  # memory bound
    assert times[1] - times[0] == pytest.approx(1e-4)  # compute bound from SRAM
    assert (ex.tile_cache.hits, ex.tile_cache.misses) == (1, 1)
    assert res[mem_key(MOD0)].served == pytest.approx(54e6)


def test_executor_cancel_and_on_start_abort(topology: Topology):
    sim, ex, _ = executor(topology, MOD0)
    kernel = Kernel(MOD0, (Phase(((compute_key(MOD0), 256e6),)),))
    done = []
    first = ex.submit(kernel, lambda j: done.append("first"))
    second = ex.submit(kernel, lambda j: done.append("second"))
    ex.submit(kernel, lambda j: done.append("third"), on_start=lambda: False)
    ex.submit(kernel, lambda j: done.append("fourth"))
    ex.cancel(second)
    sim.run(until=0.5e-3)
    ex.cancel(first)
    sim.run()
    assert done == ["fourth"] and ex.kernels_run == 1
    with pytest.raises(ValueError):
        ex.submit(Kernel(GPU, ()), lambda j: None)


# -- interconnect ------------------------------------------------------------------------
def interconnect(topology: Topology):
    sim = Simulator()
    n = FluidNetwork(sim)
    mem = {d: n.resource(f"mem{d}", dev.mem_bandwidth) for d, dev in topology.devices.items()}
    return sim, n, Interconnect(sim, n, topology, mem), mem


def test_transfer_time_is_latency_plus_bottleneck_bandwidth(topology: Topology):
    sim, _, ic, mem = interconnect(topology)
    done = []
    ic.transfer(GPU, MOD0, 16e6, lambda h: done.append(sim.now), tag="weights")
    sim.run()
    assert done == [pytest.approx(7e-6 + 1e-3)]  # two hops of latency, 16 GB/s bottleneck
    assert mem[GPU].served == mem[MOD0].served == pytest.approx(16e6)
    assert ic.link_res[(0, False)].served == ic.link_res[(1, True)].served == pytest.approx(16e6)
    assert ic.link_res[(0, True)].served == 0.0


def test_link_directions_are_independent(topology: Topology):
    sim, _, ic, _ = interconnect(topology)
    done = {}
    ic.transfer(HOST, MOD0, 16e6, lambda h: done.setdefault("down", sim.now))
    ic.transfer(MOD0, HOST, 16e6, lambda h: done.setdefault("up", sim.now))
    sim.run()
    # opposite directions do not share link bandwidth; they do share both memories (54 GB/s)
    assert done["down"] == done["up"] == pytest.approx(2e-6 + 1e-3)


def test_dma_queue_bounds_concurrent_transfers():
    from tests.conftest import make_topology

    topology = make_topology(dma_depth=2)
    sim, _, ic, _ = interconnect(topology)
    done = []
    for i in range(5):
        ic.transfer(HOST, MOD1, 16e6, lambda h, i=i: done.append(i))
    sim.run()
    stats = ic.stats[(2, True)]
    assert done == [0, 1, 2, 3, 4]
    assert stats.max_active == 2 and stats.max_waiting == 3 and stats.wait_time > 0
    assert stats.transfers == 5 and stats.active == 0


def test_cancelled_transfer_reports_partial_bytes_and_frees_its_slot(topology: Topology):
    sim, _, ic, _ = interconnect(topology)
    done = []
    handle = ic.transfer(HOST, GPU, 24e6, lambda h: done.append("first"))
    ic.transfer(HOST, GPU, 24e6, lambda h: done.append("second"))
    sim.run(until=5e-6 + 0.5e-3)
    moved = ic.cancel(handle)
    assert moved == pytest.approx(6e6, rel=1e-6)  # half the link for half a millisecond
    assert ic.cancel(handle) == moved  # idempotent
    sim.run()
    assert done == ["second"] and handle.state == "cancelled"
    with pytest.raises(ValueError):
        ic.transfer(HOST, HOST, 1.0, lambda h: None)


def test_energy_report_separates_static_and_dynamic_terms():
    from basp.experiments.profiles import load_topology
    from tests.conftest import make_topology

    topology = load_topology("basp_4module")
    sim = Simulator()
    n = FluidNetwork(sim)
    module = topology.by_name("mod0")
    mem = n.resource("mem", module.mem_bandwidth)
    mac = n.resource("mac", module.compute_rate, "mac")
    n.start({mem: 54e6, mac: 100e6})
    sim.run()
    report = energy_report(topology, {("mem", module.device_id): mem, ("compute", module.device_id): mac},
                           sim.now)  # fmt: skip
    idle = sum(d.power.idle_w for d in topology.devices.values()) * sim.now
    assert report.static_j == pytest.approx(idle)
    assert report.compute_j == pytest.approx(100e6 * module.power.joules_per_mac)
    assert report.memory_j == pytest.approx(54e6 * module.power.joules_per_mem_byte)
    assert report.total_j == pytest.approx(report.static_j + report.compute_j + report.memory_j)
    assert make_topology().devices[0].power.idle_w == 0.0
