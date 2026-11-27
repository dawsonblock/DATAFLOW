"""v0.2 experiment layer: parametric hardware, the eight architectures, design-space
exploration with the simulator in the loop, calibration glue and the hardware gate."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from basp.backends.cpu import NumericPayload
from basp.calibration import Measurement, MeasurementSet
from basp.core.capabilities import DeviceKind
from basp.dse.hardware import CORNERS, DesignError, HardwareDesign, pick
from basp.dse.pareto import dominates
from basp.dse.search import search
from basp.experiments import explore
from basp.experiments.architectures import ARCHITECTURES, ArchitectureError, configure
from basp.experiments.calibrate import calibration_status, predict
from basp.experiments.profiles import load_json, load_topology
from basp.experiments.suite import REPLICAS, default_design
from basp.experiments.variants import (
    design_for,
    run_architecture,
    scale_latency,
    traces,
    with_policy,
)
from basp.simulator.dram import ddr_channel_bandwidth
from basp.verification import check_accounting, check_dependencies, compare_to_reference
from basp.workloads.model import tiny_spec
from basp.workloads.tiny import TinyModel
from basp.workloads.zoo import MODELS, SUITE

GiB = 1024**3
ESTIMATES = load_json("design_estimates")
BASE = load_topology("desktop_gpu12")
SPEC = tiny_spec(hidden=128, expert_inter=256, n_layers=4, n_experts=16, top_k=4, n_heads=4,
                 n_kv_heads=2, head_dim=32, vocab=1024)  # fmt: skip


def via(topology, src: int, dst: int) -> set[int]:
    """Devices a transfer from ``src`` to ``dst`` passes through."""
    nodes: set[int] = set()
    for hop in topology.route(src, dst):
        link = topology.links[hop.link]
        nodes |= {link.a, link.b}
    return nodes - {src, dst}


# -- parametric hardware ----------------------------------------------------------------------
def test_design_builds_modules_from_its_parameters_and_the_estimates():
    design = HardwareDesign(modules=4, module_gib=32, channels=2, lanes=1024, link_gbps=16.0)
    topology = design.topology(BASE, ESTIMATES)
    assert [d.name for d in topology.devices.values()] == ["host", "gpu", "mod0", "mod1", "mod2", "mod3"]
    module = topology.by_name("mod2")
    assert module.kind is DeviceKind.NEAR_MEMORY and module.mem_capacity == 32 * GiB
    assert module.mem_bandwidth == pytest.approx(2 * ddr_channel_bandwidth(5600) * 0.6)
    assert module.compute_rate == pytest.approx(1024 * 250e6)
    assert module.power.idle_w == pytest.approx(5.0 + 1024 * 0.005 + 32 * 0.1)
    assert topology.by_name("gpu").mem_capacity == 12 * GiB
    assert topology.by_name("host").mem_capacity == 64 * GiB
    gpu, mod = topology.by_name("gpu").device_id, module.device_id
    assert via(topology, gpu, mod) == {topology.host}  # star: modules hang off the host
    assert len(topology.links) == 5
    wide = replace(design, channels=4, lanes=2048).topology(BASE, ESTIMATES).by_name("mod0")
    assert wide.mem_bandwidth == pytest.approx(2 * module.mem_bandwidth)
    assert wide.compute_rate == pytest.approx(2 * module.compute_rate)
    no_gpu = replace(design, gpu_gib=0).topology(BASE, ESTIMATES)
    assert not no_gpu.of_kind(DeviceKind.GPU) and len(no_gpu.of_kind(DeviceKind.NEAR_MEMORY)) == 4


def test_every_estimate_moves_the_right_way_between_corners():
    design = HardwareDesign(modules=4)
    rate, bandwidth, idle, cost, latency = [], [], [], [], []
    for corner in CORNERS:
        topology = design.topology(BASE, ESTIMATES, corner)
        module = topology.by_name("mod0")
        rate.append(module.compute_rate)
        bandwidth.append(module.mem_bandwidth)
        idle.append(module.power.idle_w)
        cost.append(design.cost_usd(ESTIMATES, corner))
        latency.append(topology.path_latency(topology.host, module.device_id))
    assert rate == sorted(rate) and bandwidth == sorted(bandwidth)
    assert idle == sorted(idle, reverse=True) and cost == sorted(cost, reverse=True)
    assert latency == sorted(latency, reverse=True)
    assert cost[0] > 2 * cost[2]  # the price range is wide, and reported as such
    assert pick({"pessimistic": 1, "nominal": 2, "optimistic": 3}, "optimistic") == 3.0
    assert pick(7, "pessimistic") == 7.0
    with pytest.raises(DesignError, match="corner"):
        design.topology(BASE, ESTIMATES, "typical")


def test_local_fabric_connects_gpu_and_modules_without_the_host():
    star = HardwareDesign(modules=2).topology(BASE, ESTIMATES)
    local = HardwareDesign(modules=2, fabric="local").topology(BASE, ESTIMATES)
    fabric = local.by_name("fabric")
    assert fabric.kind is DeviceKind.SWITCH and not fabric.caps.can_compute and fabric.mem_capacity == 0
    gpu, mod = local.by_name("gpu").device_id, local.by_name("mod1").device_id
    assert via(local, gpu, mod) == {fabric.device_id}
    assert local.path_latency(gpu, mod) == pytest.approx(2 * 0.5e-6)
    s_gpu, s_mod = star.by_name("gpu").device_id, star.by_name("mod1").device_id
    assert local.path_latency(gpu, mod) < star.path_latency(s_gpu, s_mod)
    assert via(local, local.host, mod) == {fabric.device_id}  # the host reaches it over one uplink


def test_independent_channel_engines_divide_a_module():
    pooled = HardwareDesign(modules=2).topology(BASE, ESTIMATES)
    split = HardwareDesign(modules=2, split_channels=True).topology(BASE, ESTIMATES)
    whole, half = pooled.by_name("mod0"), split.by_name("mod0.c1")
    assert split.by_name("mod0").kind is DeviceKind.SWITCH
    for field in ("mem_capacity", "mem_bandwidth", "compute_rate", "sram_bytes"):
        assert getattr(half, field) == pytest.approx(getattr(whole, field) / 2)
    assert half.power.idle_w == pytest.approx(whole.power.idle_w / 2)
    engines = [d for d in split.of_kind(DeviceKind.NEAR_MEMORY) if d.caps.can_compute]
    assert len(engines) == 4
    # both engines share the module's one link to the host
    assert split.transfer_rate(split.host, half.device_id) <= 16e9


def test_impossible_designs_are_rejected():
    for bad in ({"fabric": "mesh"}, {"modules": 2, "module_gib": 512, "channels": 2},
                {"modules": 2, "channels": 1, "split_channels": True}, {"modules": -1},
                {"modules": 2, "lanes": 0}, {"gpu_gib": -4}):  # fmt: skip
        with pytest.raises(DesignError):
            HardwareDesign(**bad)
    HardwareDesign(modules=0, lanes=0)  # module parameters are irrelevant without modules
    with pytest.raises(DesignError, match="base profile"):
        HardwareDesign(gpu_gib=12).topology(HardwareDesign(gpu_gib=0).topology(BASE, ESTIMATES), ESTIMATES)


def test_cost_model_charges_for_what_a_design_contains():
    conventional = HardwareDesign(modules=0)
    assert conventional.cost_usd(ESTIMATES) == pytest.approx(800 + 64 * 3 + 700)
    assert HardwareDesign(gpu_gib=24).cost_usd(ESTIMATES) == pytest.approx(800 + 192 + 1800)
    assert HardwareDesign(gpu_gib=32).cost_usd(ESTIMATES) == pytest.approx(800 + 192 + 2600)
    with pytest.raises(DesignError, match="no 20 GiB GPU class"):
        HardwareDesign(gpu_gib=20).cost_usd(ESTIMATES)
    # the pessimistic corner is the one least favourable to modules: cheap host and GPU
    assert conventional.cost_usd(ESTIMATES, "pessimistic") < conventional.cost_usd(ESTIMATES, "optimistic")
    four_p, four_o = (HardwareDesign(modules=4).cost_usd(ESTIMATES, c) for c in ("pessimistic", "optimistic"))
    assert four_p > four_o
    four = HardwareDesign(modules=4)
    module = 600 + 1024 * 0.5 + 32 * 3.5 + 16 * 6
    # a GPU and four 8-lane modules need more PCIe lanes than a desktop board has
    assert four.pcie_lanes(ESTIMATES) == 16 + 4 * 8 > ESTIMATES["host"]["desktop_pcie_lanes"]
    assert four.cost_usd(ESTIMATES) == pytest.approx(conventional.cost_usd(ESTIMATES) + 1800 + 4 * module)
    local = replace(four, fabric="local")
    assert local.pcie_lanes(ESTIMATES) == 16  # one uplink: no workstation board needed
    assert local.cost_usd(ESTIMATES) == pytest.approx(
        conventional.cost_usd(ESTIMATES) + 4 * module + 500 + 6 * 100)  # fmt: skip
    costs = [HardwareDesign(modules=n).cost_usd(ESTIMATES) for n in (0, 1, 2, 4, 8)]
    assert costs == sorted(costs) and len(set(costs)) == 5
    assert replace(four, lanes=2048).cost_usd(ESTIMATES) > four.cost_usd(ESTIMATES)
    assert HardwareDesign(gpu_gib=0, modules=4).cost_usd(ESTIMATES) < four.cost_usd(ESTIMATES)


def test_design_for_sizes_the_system_to_the_model():
    small = design_for(MODELS["qwen3-8b"])
    assert (small.modules, small.module_gib, small.host_gib) == (4, 16, 64)
    large = design_for(MODELS["deepseek-v3"])
    weights = MODELS["deepseek-v3"].bytes_at(4.5) / GiB
    assert large.modules * large.module_gib >= weights and large.host_gib >= weights
    assert large.module_gib <= 128 * large.channels  # physically attachable
    doubled = design_for(MODELS["qwen3-30b-a3b"], replicas=2)
    assert doubled.module_gib >= 2 * design_for(MODELS["qwen3-30b-a3b"]).module_gib / 2 > 0
    assert doubled.modules * doubled.module_gib >= 2 * MODELS["qwen3-30b-a3b"].bytes_at(4.5) / GiB
    assert design_for(SPEC, modules=0).modules == 0


# -- model shapes ------------------------------------------------------------------------------
@pytest.mark.parametrize(("key", "total_b", "active_b"), [
    ("qwen3-4b", 4.02, 4.02), ("qwen3-8b", 8.19, 8.19), ("qwen3-14b", 14.77, 14.77),
    ("qwen3-30b-a3b", 30.53, None), ("llama-3.3-70b", 70.55, 70.55), ("deepseek-v3", 671.03, 37.55),
])  # fmt: skip
def test_model_shapes_reproduce_published_parameter_counts(key, total_b, active_b):
    spec = MODELS[key]
    assert spec.total_params / 1e9 == pytest.approx(total_b, abs=0.006)
    if active_b is not None:
        assert spec.active_params / 1e9 == pytest.approx(active_b, abs=0.006)
    assert spec.active_params <= spec.total_params
    assert spec.dense == (spec.n_experts == 1)


def test_the_suite_is_ordered_by_size_and_labels_hypothetical_shapes():
    sizes = [spec.total_params for _, spec, _ in SUITE]
    assert sizes == sorted(sizes) and len(SUITE) == 8
    assert 0.9e12 < MODELS["moe-1t"].total_params < 1.2e12
    assert 2.5e12 < MODELS["moe-3t"].total_params < 3.5e12
    assert MODELS["moe-1t"].source == MODELS["moe-3t"].source == "hypothetical"
    assert all(spec.source.startswith("published config") for key, spec, _ in SUITE[:6])
    assert 3.0e9 < MODELS["qwen3-30b-a3b"].active_params < 3.6e9  # "A3B": about 3 B active


# -- the eight architectures ----------------------------------------------------------------------
PRIOR, HELD = traces(SPEC, tokens=6, prior_steps=24)


def run(key: str, **kw):
    kw.setdefault("replicas", REPLICAS[key])
    design = kw.pop("design", None) or default_design(key, HardwareDesign(modules=2, module_gib=16))
    return run_architecture(key, design, SPEC, PRIOR, HELD, tokens=6, warmup_tokens=2, context_len=64,
                            keep_log=True, **kw)  # fmt: skip


@pytest.fixture(scope="module")
def outcomes():
    return {key: run(key) for key in ARCHITECTURES}


def test_all_eight_architectures_run_the_same_workload(outcomes):
    assert list(outcomes) == list("ABCDEFGH")
    for key, out in outcomes.items():
        result = out.result
        assert result.ok and result.tokens == 4 and result.tokens_per_s > 0, key
        assert check_dependencies(result.op_log, out.setup.system) == [], key
        assert result.tokens_per_joule > 0 and out.cost_usd > 0
        out.setup.residency.check_invariants()
        # every architecture executes exactly the experts the trace selected
        experts = sum(1 for r in result.op_log if r.kind == "expert")
        assert experts == 6 * SPEC.n_layers * SPEC.top_k, key


def test_each_architecture_computes_where_its_definition_says(outcomes):
    def devices(key, kind=None, *, embed=False):
        out = outcomes[key]
        names = {d.device_id: d.name for d in out.setup.system.topology.devices.values()}
        # the embedding gather is outside the layers: it runs where the table is stored
        return {names[d] for r in out.result.op_log if kind in (None, r.kind)
                and r.name.endswith(".embed") == embed for d in r.devices}  # fmt: skip

    for key in ARCHITECTURES:
        assert devices(key, embed=True) == {"host"}, key  # one row read on the host, never the table moved
    assert devices("A") <= {"host", "gpu"}
    assert devices("B") == {"mod0", "mod1"} and devices("C") == {"mod0", "mod1"}  # no GPU at all
    assert devices("D", "expert") == {"mod0", "mod1"} and devices("D", "trunk") == {"gpu"}
    tensor_parallel = [r for r in outcomes["E"].result.op_log if r.kind == "expert"]
    assert all(len(r.devices) == 2 for r in tensor_parallel)  # every expert on every module
    assert devices("F", "trunk") == {"gpu"} and devices("G", "trunk") == {"gpu"}
    assert "fabric" in {d.name for d in outcomes["H"].setup.system.topology.devices.values()}
    # C: a layer never leaves its stage, and stages follow the layer order
    names = {d.device_id: d.name for d in outcomes["C"].setup.system.topology.devices.values()}
    by_layer: dict[int, set[str]] = {}
    for r in outcomes["C"].result.op_log:
        if r.step == 2 and r.kind in ("trunk", "expert") and not r.name.endswith(".embed"):
            layer = int(r.name.split(".")[1][1:])  # "s2.l3.e7" -> 3
            by_layer.setdefault(layer, set()).update(names[d] for d in r.devices)
    assert [by_layer[layer] for layer in range(4)] == [{"mod0"}, {"mod0"}, {"mod1"}, {"mod1"}]


def test_architecture_policies_differ_in_the_intended_way(outcomes):
    policy = {key: out.setup.policy for key, out in outcomes.items()}
    assert policy["A"].cache.kind == policy["F"].cache.kind == "benefit" and policy["D"].cache.kind == "none"
    assert policy["F"].prefetch is not None and policy["H"].prefetch is not None
    assert not policy["G"].host_in_loop and policy["G"].eager_activations
    assert not policy["C"].host_in_loop and policy["D"].host_in_loop
    system = outcomes["G"].setup.system
    assert all(d.persistent for d in system.topology.devices.values() if d.kind is not DeviceKind.CPU)
    assert {len(parts) for parts in outcomes["E"].setup.program.expert_parts[0]} == {2}
    assert outcomes["A"].design.modules == 0 and outcomes["B"].design.gpu_gib == 0
    assert outcomes["A"].cost_usd < min(o.cost_usd for k, o in outcomes.items() if k not in "ABC")


def test_architectures_refuse_topologies_they_cannot_use():
    plain = HardwareDesign(modules=0).topology(BASE, ESTIMATES)
    star = HardwareDesign(modules=2).topology(BASE, ESTIMATES)
    headless = HardwareDesign(modules=2, gpu_gib=0).topology(BASE, ESTIMATES)
    for key, topology in (("D", plain), ("B", plain), ("H", star), ("A", headless), ("F", headless),
                          ("Z", star)):  # fmt: skip
        with pytest.raises(ArchitectureError):
            configure(key, topology, PRIOR)
    # D and E fall back to the host for the trunk when there is no GPU
    assert configure("D", headless, PRIOR).policy.compute_devices >= {headless.host}


@pytest.mark.parametrize("key", list(ARCHITECTURES))
def test_every_architecture_is_numerically_equivalent_to_the_reference(key):
    spec = tiny_spec()
    model = TinyModel.random(spec, seed=5)
    prior, _ = traces(spec, tokens=1, prior_steps=8)
    design = default_design(key, HardwareDesign(modules=2, module_gib=16))
    config = configure(key, design.topology(BASE, ESTIMATES), prior, replicas=REPLICAS[key])
    setup = config.build(spec, prior, batch=2, context_len=16, max_new_tokens=4)
    payload = NumericPayload(model, setup.program, model.initial_input(2))
    result = setup.run(None, tokens=4, payload=payload, keep_log=True)
    report = compare_to_reference(model, payload)
    assert result.ok and report.ok and report.routing_matches and report.tokens_match
    assert report.max_rel_l2 < 2e-3
    assert check_accounting(setup.system) == []


def test_variant_helpers_edit_topology_and_policy():
    base = run("D")
    slow = run("D", topology_edit=lambda t: scale_latency(t, 8.0))
    assert slow.tokens_per_s < base.tokens_per_s
    topology = slow.setup.system.topology
    assert topology.sync_latency_s == pytest.approx(8 * base.setup.system.topology.sync_latency_s)
    free = run("D", config_edit=with_policy(host_in_loop=False))
    assert free.tokens_per_s > base.tokens_per_s and not free.setup.policy.host_in_loop
    pessimistic, optimistic = run("D", corner="pessimistic"), run("D", corner="optimistic")
    assert pessimistic.tokens_per_s < base.tokens_per_s < optimistic.tokens_per_s
    assert pessimistic.cost_usd > base.cost_usd > optimistic.cost_usd
    assert base.per_kusd == pytest.approx(1e3 * base.tokens_per_s / base.cost_usd)


# -- design-space exploration ----------------------------------------------------------------------
TINY_WORKLOAD = {"model": "tiny", "tokens": 3, "warmup_tokens": 1, "context_len": 32,
                 "trace": {"zipf_s": 0.8, "stickiness": 0.6, "seed": 7, "prior_steps": 8}}  # fmt: skip


def test_design_space_contains_only_meaningful_candidates():
    model_gib = MODELS["qwen3-30b-a3b"].bytes_at() / GiB
    space = explore.design_space(model_gib)
    assert space.size == 8 * 4 * 4 * 3 * 3 * 4 * 3 * 2
    designs = list(space.enumerate())
    assert len(designs) == 10_803
    by_arch = {arch: [d for d in designs if d["arch"] == arch] for arch in "ABCDEFGH"}
    assert len(by_arch["A"]) == 3  # one point per GPU class: module parameters mean nothing there
    assert all(d["modules"] == 0 and d["gpu_gib"] > 0 for d in by_arch["A"])
    assert all(d["gpu_gib"] == 0 for d in by_arch["B"] + by_arch["C"])  # never pay for an unused GPU
    assert all(d["gpu_gib"] > 0 for d in by_arch["F"] + by_arch["H"])
    assert all(d["replicas"] == 1 for d in by_arch["C"] + by_arch["E"])
    for d in designs:
        if d["arch"] != "A":
            assert d["modules"] * d["module_gib"] >= model_gib * d["replicas"] * 1.1
            assert d["module_gib"] <= 128 * d["channels"] and d["replicas"] <= d["modules"]
        explore.to_design(d)  # every candidate is a constructible system
    # a model larger than host memory leaves no conventional candidate
    assert not [d for d in explore.design_space(200.0, module_sizes=(64,)).enumerate() if d["arch"] == "A"]


def test_seed_designs_cover_the_baseline_and_every_architecture():
    space = explore.design_space(MODELS["qwen3-30b-a3b"].bytes_at() / GiB)
    seeds = explore.seed_designs(space)
    assert all(space.feasible(s) for s in seeds)
    assert len({space.key(s) for s in seeds}) == len(seeds)
    assert {s["arch"] for s in seeds} == set("ABCDEFGH")
    assert sorted(s["gpu_gib"] for s in seeds if s["arch"] == "A") == [12, 24, 32]
    design = explore.to_design({"arch": "H", "gpu_gib": 12, "modules": 4, "module_gib": 32,
                                "channels": 2, "lanes": 512, "link_gbps": 8.0, "replicas": 2})  # fmt: skip
    assert design.fabric == "local" and not design.persistent and design.lanes == 512
    assert explore.to_design({**seeds[-1], "arch": "G"}).persistent


def test_evaluating_a_design_is_deterministic_and_reports_all_objectives():
    config = {"arch": "D", "gpu_gib": 12, "modules": 2, "module_gib": 16, "channels": 2,
              "lanes": 1024, "link_gbps": 16.0, "replicas": 1}  # fmt: skip
    metrics = explore.evaluate_design(config, workload=TINY_WORKLOAD)
    assert metrics == explore.evaluate_design(dict(config), workload=TINY_WORKLOAD)
    assert {o.name for o in explore.OBJECTIVES} <= set(metrics)
    assert metrics["tokens_per_s_per_kusd"] == pytest.approx(
        1e3 * metrics["tokens_per_s"] / metrics["cost_usd"])  # fmt: skip
    assert metrics["cost_usd"] == pytest.approx(explore.to_design(config).cost_usd(ESTIMATES))
    assert metrics["average_power_w"] == pytest.approx(
        metrics["tokens_per_s"] / metrics["tokens_per_joule"], rel=1e-6)  # fmt: skip
    corners = explore.evaluate_corners(config, workload=TINY_WORKLOAD)
    assert corners["nominal"] == metrics
    assert corners["pessimistic"]["tokens_per_s"] < metrics["tokens_per_s"] < corners["optimistic"]["tokens_per_s"]
    assert corners["pessimistic"]["cost_usd"] > metrics["cost_usd"] > corners["optimistic"]["cost_usd"]
    # designs that cannot be built or cannot hold the model are infeasible, not errors
    assert explore.evaluate_design({**config, "arch": "H", "gpu_gib": 0}, workload=TINY_WORKLOAD) is None
    assert explore.evaluate_design({**config, "arch": "D", "modules": 0}, workload=TINY_WORKLOAD) is None
    big = dict(TINY_WORKLOAD, model="llama-3.3-70b", tokens=1, warmup_tokens=0)
    assert explore.evaluate_design({**config, "module_gib": 16, "modules": 1}, workload=big) is None


def test_exhaustive_search_of_a_small_space_returns_the_true_pareto_front():
    space = explore.design_space(1e-3, gpu_sizes=(0, 12), module_counts=(0, 2), module_sizes=(16,),
                                 channels=(2,), lanes=(256, 1024), links=(16.0,))  # fmt: skip
    designs = list(space.enumerate())
    assert 10 < len(designs) < 40 and sum(d["arch"] == "A" for d in designs) == 1

    def evaluate(config):
        return explore.evaluate_design(config, workload=TINY_WORKLOAD)

    result = search(space, evaluate, explore.OBJECTIVES, method="exhaustive")
    assert result.n_evaluated == len(designs)
    feasible = [e for e in result.evaluations if e.feasible]
    assert {e.config["arch"] for e in feasible} == set("ABCDEFGH")
    front = result.front
    assert front and all(e.feasible for e in front)

    for e in front:
        assert not any(dominates(o.metrics, e.metrics, explore.OBJECTIVES) for o in feasible)
    for e in feasible:
        if e not in front:
            assert any(dominates(o.metrics, e.metrics, explore.OBJECTIVES) for o in front)
    # the conventional system is a candidate like any other -- and the cheapest point on the front
    cheapest = min(front, key=lambda e: e.metrics["cost_usd"])
    assert cheapest.config["arch"] == "A"
    assert cheapest.metrics["cost_usd"] == pytest.approx(HardwareDesign().cost_usd(ESTIMATES))


# -- calibration glue --------------------------------------------------------------------------------
def measurement(i: int, tokens_per_s: float = 1.0, **kw) -> Measurement:
    kw = {"machine": "i9-11900kf-3080ti", "split": "validation", "kind": "decode", "model": "qwen3-8b",
          "bits_per_weight": 4.85, "n_gpu_layers": 4 * i, **kw}  # fmt: skip
    return Measurement(id=f"m{i}", tokens_per_s=tokens_per_s, **kw)


def test_prediction_follows_the_benchmark_description():
    speeds = [predict(measurement(0, n_gpu_layers=n)) for n in (0, 12, 24, 36, 99, -1)]
    assert speeds[:4] == sorted(speeds[:4]) and speeds[0] < speeds[3]  # more layers on the GPU: faster
    assert speeds[3] == speeds[4] == speeds[5]  # -ngl beyond the model, or -1, means every layer
    # heavier quantisation formats read more bytes per token
    assert predict(measurement(0, bits_per_weight=8.5)) < predict(measurement(0, bits_per_weight=4.85))
    assert predict(measurement(0, machine="unknown-box")) is None
    assert predict(measurement(0, model="not-a-model")) is None
    # a 70B model does not fit this machine's 12 GiB GPU entirely
    assert predict(measurement(0, model="llama-3.3-70b", n_gpu_layers=-1)) is None
    m2 = predict(measurement(0, machine="macbook-m2", n_gpu_layers=-1))
    assert m2 is not None and m2 > 0
    prefill = predict(measurement(0, kind="prefill", batch=512, n_gpu_layers=-1))
    assert prefill > 10 * speeds[5]  # prompt tokens per second: the prompt is processed together
    assert predict(measurement(0, kind="prefill", n_gpu_layers=-1)) == prefill  # llama-bench's 512
    # a prompt after 4096 tokens of context attends to more and reads a larger cache
    assert predict(measurement(0, kind="prefill", batch=512, context_len=4096, n_gpu_layers=-1)) < prefill
    decode_deep = predict(measurement(0, n_gpu_layers=-1, context_len=4096 + 64))
    assert decode_deep < speeds[5]
    # experts of the first layers kept on the CPU (llama.cpp --n-cpu-moe): slower than all on the GPU
    moe = [predict(measurement(0, model="qwen3-30b-a3b", n_gpu_layers=99, n_cpu_moe=n)) for n in (24, 36, 48)]
    assert moe == sorted(moe, reverse=True) and all(v is not None for v in moe)
    custom = predict(measurement(0, machine="lab"), {"lab": load_topology("desktop_gpu12")})
    assert custom is not None and custom > 0


def test_calibration_verdict_depends_only_on_held_out_measurements(tmp_path):
    assert calibration_status(None).status == "not_evaluated"
    assert calibration_status(tmp_path / "absent.json").status == "not_evaluated"
    truth = {i: predict(measurement(i)) for i in range(9)}

    def write(scale, n=9, split="validation", extra=()):
        items = [measurement(i, truth[i] * scale(i), split=split) for i in range(n)]
        path = tmp_path / "m.json"
        MeasurementSet([*items, *extra]).save(path)
        return path

    exact = calibration_status(write(lambda i: 1.0))
    assert exact.status == "pass" and exact.validation.n == 9 and exact.validation.median < 1e-9
    assert calibration_status(write(lambda i: 1.3)).status == "fail"  # 30% off everywhere
    assert calibration_status(write(lambda i: 1.3 if i < 2 else 1.0)).status == "fail"  # p95 too large
    assert calibration_status(write(lambda i: 1.05)).status == "pass"
    few = calibration_status(write(lambda i: 1.0, n=5))
    assert few.status == "not_evaluated" and few.validation.n == 5
    # measurements used for tuning never count as evidence, however well they match
    tuned = calibration_status(write(lambda i: 1.0, split="calibration"))
    assert tuned.status == "not_evaluated" and tuned.validation.n == 0 and tuned.calibration.n == 9
    wrong = [measurement(20 + i, 1e-3, split="calibration", context_len=512 + i) for i in range(3)]
    mixed = calibration_status(write(lambda i: 1.0, extra=wrong))
    assert mixed.status == "pass" and mixed.calibration.median > 0.5  # reported, but not decisive
    json.dumps(exact.to_json())


def test_base_profile_selection_changes_the_machine_under_every_design():
    from basp.experiments.variants import base_profile, set_base_profile

    assert base_profile() == "desktop_gpu12"
    default = run("A")
    previous = set_base_profile("i9_11900kf_rtx3080ti")
    try:
        assert previous == "desktop_gpu12" and base_profile() == "i9_11900kf_rtx3080ti"
        other = run("A")
    finally:
        set_base_profile(previous)
    assert other.setup.system.topology.by_name("host").mem_bandwidth == 36e9
    assert default.setup.system.topology.by_name("host").mem_bandwidth == 54e9
    assert run("A", base_profile="i9_11900kf_rtx3080ti").tokens_per_s == other.tokens_per_s
    with pytest.raises(FileNotFoundError):
        set_base_profile("no_such_machine")
    assert base_profile() == "desktop_gpu12"
