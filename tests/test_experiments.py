"""Analytical model, standard configurations and report generation."""

from __future__ import annotations

import json

import pytest

from basp.core.capabilities import DeviceKind
from basp.dataflow.base import DataflowMode
from basp.experiments.analytic import compare_paths, size_arithmetic
from basp.experiments.configurations import standard_configurations, with_modes
from basp.experiments.profiles import load_topology, load_workload
from basp.experiments.report import (
    build_report,
    make_traces,
    predictive_experiment,
    run_config,
    summarise,
    table,
    vary_topology,
)
from basp.units import GiB, KiB, MiB
from basp.workloads.model import MoEModelSpec
from basp.workloads.trace import cyclic_trace

MID = MoEModelSpec("mid", hidden=512, n_layers=6, n_experts=32, top_k=4, expert_inter=1024,
                   n_heads=8, n_kv_heads=2, head_dim=64, vocab=4096)  # fmt: skip
WORKLOAD = {"model": "mid", "batch": 1, "context_len": 256, "tokens": 10, "warmup_tokens": 4,
            "trace": {"zipf_s": 0.8, "stickiness": 0.6, "seed": 7, "prior_steps": 24}}  # fmt: skip


def test_analytic_model_reproduces_the_specification_table():
    c = compare_paths()
    assert c.params == 176_160_768  # "176.2M"
    assert c.weight_bytes / MiB == pytest.approx(94.5)
    assert c.activation_bytes / KiB == 16.0
    assert c.traffic_reduction == pytest.approx(6048)
    assert c.cold_transfer_s * 1e3 == pytest.approx(4.129, abs=5e-4)
    assert c.near_memory_s * 1e3 == pytest.approx(1.836, abs=5e-4)
    assert c.near_memory_bound == "memory bandwidth"
    assert dict(c.as_rows())["Interconnect traffic reduction"] == "6,048x"
    # with a fast memory system the same kernel becomes compute-bound, and slow decoding dominates
    assert compare_paths(memory_bandwidth=400e9).near_memory_bound == "compute throughput"
    assert compare_paths(decode_rate=10e9).near_memory_bound == "decode throughput"
    assert compare_paths(bits_per_weight=2.5).weight_bytes < c.weight_bytes


def test_arithmetic_sizing_matches_section_10():
    s = size_arithmetic()
    assert s.memory_bandwidth == 89.6e9
    assert s.required_mac_rate / 1e9 == pytest.approx(159.3, abs=0.05)
    assert s.lane_mac_rate == 256e9 and s.headroom == pytest.approx(1.607, abs=1e-3)
    assert size_arithmetic(lanes=512).headroom < 1.0  # half the lanes cannot keep up


def scaled_profiles():
    """The shipped profiles with the GPU shrunk so the mid-sized model does not fit."""
    expert = MID.expert_params * 4.5 / 8
    gpu = int(0.5 * MID.n_layers * MID.n_experts * expert)
    return (vary_topology(load_topology("desktop_gpu12"), gpu_bytes=gpu),
            vary_topology(load_topology("basp_4module"), gpu_bytes=gpu))  # fmt: skip


@pytest.fixture(scope="module")
def results():
    desktop, basp = scaled_profiles()
    prior, held = make_traces(MID, WORKLOAD)
    out = {}
    for config in standard_configurations(desktop, basp, prior):
        out[config.key] = summarise(run_config(config, MID, prior, held, WORKLOAD))
    return out


def test_all_eight_configurations_run_cleanly(results):
    assert list(results) == ["gpu_only", "cpu_only", "static_split", "adaptive", "distributed",
                             "predictive", "compression", "complete"]  # fmt: skip
    for key, summary in results.items():
        assert summary["tokens_per_s"] > 0 and summary["tokens"] == 6, key
        assert summary["estimate_error"] < 0.1, key


def test_configuration_results_follow_from_the_dataflow(results):
    r = results
    assert r["cpu_only"]["link_mb_per_token"] == 0.0  # nothing crosses a link
    assert r["cpu_only"]["tokens_per_s"] < r["static_split"]["tokens_per_s"]
    assert r["adaptive"]["tokens_per_s"] > r["static_split"]["tokens_per_s"]
    assert r["complete"]["tokens_per_s"] >= r["distributed"]["tokens_per_s"]
    # near-memory execution moves activations, not weights
    assert r["distributed"]["link_mb_per_token"] < 0.2 * r["gpu_only"]["link_mb_per_token"]
    assert "weights" not in r["distributed"]["link_mb_by_kind"]
    # compressed cold copies shrink the weight traffic of GPU paging by the format ratio
    ratio = r["compression"]["link_mb_by_kind"]["weights"] / r["gpu_only"]["link_mb_by_kind"]["weights"]
    assert ratio == pytest.approx(4.12 / 4.5, rel=0.02)
    assert r["distributed"]["devices"]["mod0"]["mem_utilisation_busy"] > 0.6
    assert r["complete"]["mode_counts"]["output_stationary"] > 0


def test_restricting_modes_changes_only_the_policy():
    desktop, basp = scaled_profiles()
    prior, _ = make_traces(MID, WORKLOAD)
    complete = standard_configurations(desktop, basp, prior)[-1]
    limited = with_modes(complete, {DataflowMode.WEIGHT_STATIONARY}, "ws_only")
    assert limited.key == "ws_only" and limited.policy.modes == {DataflowMode.WEIGHT_STATIONARY}
    assert limited.topology is complete.topology and complete.policy.modes != limited.policy.modes
    with pytest.raises(ValueError, match="no compute-memory modules"):
        standard_configurations(desktop, desktop, prior)


def test_vary_topology_changes_capacity_and_module_count():
    basp = load_topology("basp_4module")
    small = vary_topology(basp, gpu_bytes=6 * GiB, modules=2)
    assert small.of_kind(DeviceKind.GPU)[0].mem_capacity == 6 * GiB
    assert len(small.of_kind(DeviceKind.NEAR_MEMORY)) == 2 and len(small.links) == 3
    assert len(basp.of_kind(DeviceKind.NEAR_MEMORY)) == 4  # the original is untouched


def test_prefetching_reduces_stalls_on_held_out_structured_routing():
    """v0.7 evidence, in simulation: fewer stalls on routing the predictor never saw."""
    rows = {(r["routing"], r["policy"]): r for r in predictive_experiment(
        load_topology("desktop_gpu12"), tokens=16, prior_steps=32)}  # fmt: skip
    reactive, prefetch = rows[("cyclic", "reactive cache")], rows[("cyclic", "with prefetch")]
    assert prefetch["prefetch_hits"] > 0 and prefetch["wasted_mb"] == 0.0
    assert prefetch["stall_ms_per_token"] < 0.9 * reactive["stall_ms_per_token"]
    assert prefetch["tokens_per_s"] > reactive["tokens_per_s"]
    sticky = rows[("sticky", "with prefetch")]
    # mispredictions cost only budgeted background bandwidth, never throughput
    assert sticky["tokens_per_s"] >= 0.98 * rows[("sticky", "reactive cache")]["tokens_per_s"]


def test_cyclic_trace_has_only_transitional_structure():
    trace = cyclic_trace(16, 2, 32, 4, period=4, seed=1)
    trace.validate()
    for layer in range(2):
        sets = [set(trace.at(step, layer).experts[0]) for step in range(8)]
        assert sets[0] == sets[4] and sets[1] == sets[5]  # period 4
        assert all(not (sets[i] & sets[i + 1]) for i in range(7))  # no repeats between steps
    with pytest.raises(ValueError):
        cyclic_trace(4, 1, 8, 4, period=4)


def test_report_is_deterministic_and_self_describing(tmp_path, monkeypatch):
    from basp.experiments import profiles, report

    monkeypatch.setitem(profiles.MODELS, "mid", MID)
    workload = tmp_path / "workload.json"
    workload.write_text(json.dumps(WORKLOAD))
    monkeypatch.setattr(report, "predictive_experiment", lambda *a, **k: [])
    first, data = build_report(str(workload), quick=True)
    second, _ = build_report(str(workload), quick=True)
    assert first == second
    assert "simulation" in first.split("\n")[0] and "not measurements" not in first[:10]
    for heading in ("Analytical dataflow comparison", "Assumptions", "Eight execution configurations",
                    "No dataflow is universally best", "Acceptance targets"):  # fmt: skip
        assert heading in first
    assert "6,048x" in first and "1.836 ms" in first and "4.129 ms" in first
    assert set(data["configurations"]) >= {"gpu_only", "complete"}
    assert all(run["ok"] for run in data["correctness"]["runs"])
    json.dumps(data, default=float)  # results are JSON-serialisable
    assert load_workload("workload_qwen3_decode")["model"] == "qwen3-30b-a3b"
    assert table(["a", "b"], [[1, 2.5]]).splitlines()[2] == "| 1 | 2.5 |"
