"""v0.2 investigations, suites and report pieces, exercised on a small model."""

from __future__ import annotations

import json

import pytest

from basp.dse.hardware import CORNERS
from basp.experiments import investigations as inv
from basp.experiments import report_v4
from basp.experiments.calibrate import calibration_status
from basp.experiments.gate import evaluate_gate
from basp.experiments.profiles import load_json
from basp.experiments.suite import architecture_table, default_design, model_suite, workload_suite
from basp.workloads.model import tiny_spec

SPEC = tiny_spec(hidden=128, expert_inter=256, n_layers=4, n_experts=16, top_k=4, n_heads=4,
                 n_kv_heads=2, head_dim=32, vocab=1024)  # fmt: skip


@pytest.fixture(scope="module")
def lab():
    return inv.Lab(SPEC, tokens=5, warmup_tokens=1, prior_steps=16, context_len=64)


@pytest.fixture(scope="module")
def results(lab):
    return inv.run_all(lab)


def by_variant(rows):
    return {r["variant"]: r for r in rows}


def test_every_investigation_reports_its_variants(results):
    assert set(results) == {"critical_path", "imbalance", "scaling", "synchronisation", "launch_overhead",
                            "memory_channels", "communication", "output_projection", "energy",
                            "scheduler", "compression"}  # fmt: skip
    counts = {name: len(rows) for name, rows in results.items() if isinstance(rows, list)}
    assert counts == {"imbalance": 6, "scaling": 11, "synchronisation": 6, "launch_overhead": 3,
                      "memory_channels": 4, "communication": 4, "output_projection": 3, "energy": 6,
                      "scheduler": 4}  # fmt: skip
    for name, rows in results.items():
        if isinstance(rows, list):
            assert all(r["tokens_per_s"] > 0 for r in rows), name
    json.dumps(results)  # plain data: goes straight into results.json


def test_critical_path_investigation_accounts_for_the_token(results):
    cp = results["critical_path"]
    assert cp["residual_ms"] < 1e-9
    assert sum(cp["by_component_ms"].values()) == pytest.approx(cp["token_time_ms"], rel=1e-9)
    assert cp["token_time_ms"] == pytest.approx(1e3 / cp["tokens_per_s"], rel=1e-6)
    assert cp["serial_ms"] + cp["expert_phase_ms"] == pytest.approx(cp["token_time_ms"])
    assert cp["expert_ops_per_token"] == SPEC.n_layers * SPEC.top_k
    ideal = cp["ideal_efficiency"]
    assert 1.0 >= ideal["2"] >= ideal["4"] >= ideal["8"] > 0  # the serial share grows with modules


def test_scaling_is_measured_against_each_architectures_smallest_system(results):
    rows = results["scaling"]
    for label in ("D expert-local", "E tensor-parallel"):
        own = [r for r in rows if r["architecture"] == label]
        assert [r["modules"] for r in own] == [1, 2, 4, 8]
        assert own[0]["speedup"] == 1.0 and own[0]["efficiency"] == 1.0
        for r in own:
            assert r["efficiency"] == pytest.approx(r["speedup"] / r["modules"])
            assert r["speedup"] == pytest.approx(r["tokens_per_s"] / own[0]["tokens_per_s"])
    replicated = [r for r in rows if r["architecture"] == "D with 2 replicas"]
    assert [r["modules"] for r in replicated] == [2, 4, 8]  # two copies need two modules
    single = next(r for r in rows if r["architecture"] == "D expert-local" and r["modules"] == 1)
    assert replicated[0]["speedup"] == pytest.approx(replicated[0]["tokens_per_s"] / single["tokens_per_s"])


def test_mechanism_experiments_move_in_the_expected_direction(results):
    launch = by_variant(results["launch_overhead"])
    assert launch["experts batched into one launch"]["speedup"] > 1.0
    assert launch["persistent kernels"]["speedup"] >= launch["experts batched into one launch"]["speedup"]
    sync = by_variant(results["synchronisation"])
    assert sync["dependency events only (latencies x1)"]["speedup"] > 1.0
    # removing host round trips matters more when they are slower
    assert (sync["dependency events only (latencies x8)"]["speedup"]
            > sync["dependency events only (latencies x1)"]["speedup"])  # fmt: skip
    comm = by_variant(results["communication"])
    assert comm["activations streamed eagerly (latencies x8)"]["speedup"] >= 1.0
    assert (comm["activations streamed eagerly (latencies x8)"]["link_mb_per_token"]
            >= comm["activations sent after routing (latencies x8)"]["link_mb_per_token"])  # fmt: skip
    energy = by_variant(results["energy"])
    assert energy["F: idle devices gated"]["tokens_per_joule"] >= energy["F: no power gating"]["tokens_per_joule"]
    aware, plain = energy["A: energy-aware scheduling (0.01 s per joule)"], energy["A: time-only scheduling"]
    assert aware["tokens_per_joule"] >= plain["tokens_per_joule"] * (1 - 1e-9)
    imbalance = results["imbalance"]
    assert imbalance[0]["modules"] == 1 and imbalance[0]["efficiency"] == 1.0
    assert all(r["efficiency"] == pytest.approx(r["speedup"] / 4) for r in imbalance[1:])


def test_scheduler_investigation_reports_search_effort_and_its_cost(results):
    rows = results["scheduler"]
    assert [r["variant"] for r in rows] == ["greedy earliest-finish", "one refinement pass",
                                            "two refinement passes", "four refinement passes"]  # fmt: skip
    plans = [r["plans_per_batch"] for r in rows]
    assert plans == sorted(plans) and plans[0] > 0  # more passes evaluate more plans
    assert rows[0]["joint_improvements"] == 0
    for r in rows:
        assert r["tokens_per_s"] >= rows[0]["tokens_per_s"] * (1 - 1e-9)  # refinement never hurts
        overhead = list(r["overhead"].values())
        assert overhead == sorted(overhead) and overhead[0] > 0
        assert list(r["overhead"]) == ["50 ns per plan evaluation", "200 ns per plan evaluation",
                                       "1000 ns per plan evaluation"]  # fmt: skip


def test_compression_estimate_is_labelled_by_weight_distribution(results):
    compression = results["compression"]
    assert set(compression) == {"gaussian", "laplace", "student-t (3 d.o.f.)"}
    for row in compression.values():
        assert 2.0 < row["entropy_bits_per_weight"] < 4.5
        assert row["saving_vs_int4"] == pytest.approx(1 - row["entropy_bits_per_weight"] / 4.5)
    # heavier tails concentrate the codes: more to gain
    assert compression["student-t (3 d.o.f.)"]["saving_vs_int4"] > compression["gaussian"]["saving_vs_int4"]
    assert inv.compression(seed=0) == compression


# -- suites ---------------------------------------------------------------------------------------
def test_architecture_table_carries_a_range_for_every_figure(lab):
    rows = architecture_table(lab)
    assert [r["key"] for r in rows] == list("ABCDEFGH")
    for row in rows:
        assert row["design"] == default_design(row["key"]).as_dict()
        speeds = [row[c]["tokens_per_s"] for c in CORNERS]
        costs = [row[c]["cost_usd"] for c in CORNERS]
        if row["key"] == "A":  # shared components only: the pessimistic corner prices them low
            assert costs == sorted(costs) and costs[0] < costs[2]
        else:  # the modules dominate: the pessimistic corner prices them high
            assert costs == sorted(costs, reverse=True) and costs[0] > costs[2]
        if row["key"] == "A":
            assert all(row[c]["vs_a"] == {"throughput": 1.0, "energy": 1.0, "cost_efficiency": 1.0}
                       for c in CORNERS)  # fmt: skip
            assert speeds[0] == speeds[1] == speeds[2]  # existing hardware: no estimate range
        else:
            assert speeds[0] <= speeds[2]  # hypothetical hardware: pessimistic to optimistic
            if row["key"] in "BCDEG":  # these run their experts on the modules whatever happens
                assert speeds[0] < speeds[1] < speeds[2]
            for c in CORNERS:
                own, ref = row[c], rows[0][c]
                assert own["vs_a"]["throughput"] == pytest.approx(own["tokens_per_s"] / ref["tokens_per_s"])
                assert own["vs_a"]["cost_efficiency"] == pytest.approx(
                    own["tokens_per_s_per_kusd"] / ref["tokens_per_s_per_kusd"])  # fmt: skip
    text = report_v4.render_architectures(rows)
    assert text.count("\n") == 9 and "H Hierarchical" in text and "throughput vs A" in text


def test_model_suite_sizes_each_system_and_prefers_the_baseline_on_ties():
    rows = model_suite(architectures=("A", "D", "E"), tokens=2, warmup_tokens=0,
                       keys=("qwen3-4b", "qwen3-14b-int8"))  # fmt: skip
    assert [r["key"] for r in rows] == ["qwen3-4b", "qwen3-14b-int8"]
    small, pressed = rows
    assert small["params_b"] == pytest.approx(4.02, abs=0.01) and small["bits_per_weight"] == 4.5
    assert small["gpu_gib"] == pressed["gpu_gib"] == 12
    assert set(small["tokens_per_s"]) == {"A", "D", "E"}
    assert small["fastest"] == "A" and small["speedup_vs_a"] <= 1.0  # fits the GPU: nothing to gain
    assert small["best_per_dollar"] == "A" and small["per_dollar_vs_a"] < 1.0
    assert small["cost_usd"]["A"] < small["cost_usd"]["D"]
    # at 8.5 bits per weight the 14B model no longer fits a 12 GiB GPU
    assert pressed["bits_per_weight"] == 8.5 and pressed["weights_gib"] > 12
    assert pressed["model"] == "Qwen3-14B (INT8)" and pressed["speedup_vs_a"] > 1.0
    assert pressed["best_module_arch"] in ("D", "E") and pressed["fastest"] == pressed["best_module_arch"]


def test_model_suite_steps_up_the_gpu_when_the_trunk_alone_does_not_fit():
    # the 3T shape's attention blocks exceed 12 GiB: nothing can be placed on the smallest GPU
    rows = model_suite(architectures=("A",), tokens=1, warmup_tokens=0, keys=("moe-3t",))
    assert rows[0]["gpu_gib"] == 24 and rows[0]["tokens_per_s"]["A"] > 0
    assert rows[0]["source"] == "hypothetical" and "speedup_vs_a" not in rows[0]
    none = model_suite(architectures=("A",), tokens=1, warmup_tokens=0, keys=("moe-3t",), gpu_sizes=(12,))
    assert none[0]["tokens_per_s"] == {} and "fastest" not in none[0]


def test_workload_suite_reports_each_workload_type_separately():
    rows = workload_suite(SPEC, quick=True)
    names = [r["workload"] for r in rows]
    assert len(rows) == 6 and len(set(names)) == 6
    for prefix in ("single-user decoding", "batched inference", "concurrent requests",
                   "long-context attention", "prefill", "sustained decoding"):  # fmt: skip
        assert any(n.startswith(prefix) for n in names)
    for row in rows:
        assert set(row["tokens_per_s"]) == {"A", "H"} and row["fastest"] in ("A", "H")
        assert row["speedup_vs_a"] == pytest.approx(row["tokens_per_s"]["H"] / row["tokens_per_s"]["A"])
    metrics = {r["workload"].split(" (")[0]: r["metric"] for r in rows}
    assert metrics["prefill"] == "prompt tokens/s" and metrics["batched inference"] == "aggregate tokens/s"
    single, long_context = rows[0], rows[3]
    assert long_context["tokens_per_s"]["A"] < single["tokens_per_s"]["A"]  # attention over 32k tokens


# -- report pieces -----------------------------------------------------------------------------------
def test_numeric_verification_covers_three_dataflow_families():
    correct = report_v4.correctness()
    assert correct["all_ok"] and len(correct["runs"]) == 3
    # with the reduction order fixed, the graph form must not change a single bit
    assert correct["fused_equals_unfused"]
    assert correct["max_rel_l2"] < load_json("hardware_gate")["numeric_rel_l2_max"]
    # partial sums and expert shards add in a different order: equal within rounding only
    assert 0 <= correct["max_abs_diff_between_dataflows"] < 2e-3 * correct["logit_scale"]
    for run in correct["runs"]:
        assert run["routing_matches"] and run["tokens_match"]
        assert run["dependency_violations"] == 0 and run["accounting_violations"] == 0


def test_calibration_plan_predicts_the_requested_benchmarks():
    plan = report_v4.calibration_plan()
    assert len(plan) == 8 and {r["model"] for r in plan} == {"Qwen3-4B", "Qwen3-8B", "Qwen3-14B",
                                                             "Qwen3-30B-A3B"}  # fmt: skip
    speeds = {(r["model"], r["gpu_layers"]): r["predicted_tokens_per_s"] for r in plan}
    assert all(v is not None and v > 0 for v in speeds.values())
    assert speeds[("Qwen3-8B", 36)] > 5 * speeds[("Qwen3-8B", 0)]  # GPU against CPU-only
    assert speeds[("Qwen3-30B-A3B", 32)] > speeds[("Qwen3-30B-A3B", 16)] > speeds[("Qwen3-30B-A3B", 0)]
    # CPU-only decoding of a dense model is limited by memory bandwidth in the prior profile
    assert 4 < speeds[("Qwen3-8B", 0)] < 12


def dse_entry(arch, tokens_per_s, joule, cost, power, **config):
    config = {"arch": arch, "gpu_gib": 12, "modules": 0 if arch == "A" else 4, "module_gib": 32,
              "channels": 2, "lanes": 1024, "link_gbps": 16.0, "replicas": 1, **config}  # fmt: skip
    metrics = {"tokens_per_s": tokens_per_s, "tokens_per_joule": joule, "cost_usd": cost,
               "tokens_per_s_per_kusd": 1e3 * tokens_per_s / cost, "average_power_w": power}  # fmt: skip
    return {"config": config, "metrics": metrics, "feasible": True}


def test_exploration_summary_compares_at_equal_cost_and_equal_power():
    a12 = dse_entry("A", 150.0, 1.2, 1700.0, 125.0)
    a24 = dse_entry("A", 400.0, 2.5, 2800.0, 160.0, gpu_gib=24)
    cheap_e = dse_entry("E", 120.0, 0.9, 4500.0, 135.0, gpu_gib=0)
    fast_h = dse_entry("H", 420.0, 1.9, 9000.0, 220.0, replicas=2)
    data = {
        "evaluations": [a12, a24, cheap_e, fast_h,
                        {"config": {"arch": "D"}, "metrics": {}, "feasible": False}],
        "front": [{**fast_h, "corners": {}}, {**a12, "corners": {}}, {**a24, "corners": {}}],
        "evaluated": 5, "feasible_designs": 100, "space_size": 400, "method": "evolutionary",
    }  # fmt: skip
    summary = report_v4.summarise_dse(data)
    assert summary["feasible_evaluated"] == 4 and summary["front_size"] == 3
    assert [e["metrics"]["cost_usd"] for e in summary["front"]] == [1700.0, 2800.0, 9000.0]
    assert summary["front_archs"] == {"A": 2, "E": 0, "H": 1}
    assert summary["per_arch"]["A"]["fastest"] is a24 and summary["per_arch"]["A"]["evaluated"] == 2
    assert summary["per_arch"]["A"]["best_per_dollar"] is a24
    budgets = summary["equal_cost"]
    assert budgets["2000"]["best"] is a12 and budgets["5000"]["best"] is a24
    assert budgets["10000"]["best"] is fast_h and budgets["10000"]["best_conventional"] is a24
    power = summary["equal_power"]
    assert power["130"]["best"] is a12 and power["160"]["best"] is a24 and power["200"]["best"] is a24
    assert report_v4._design_text(a24["config"]) == "A, 24 GiB GPU"
    assert report_v4._design_text(fast_h["config"]) == (
        "H, 12 GiB GPU + 4 x 32 GiB modules, 2 ch, 1024 lanes, 16 GB/s, 2 replicas")  # fmt: skip
    assert report_v4._design_text(cheap_e["config"]).startswith("E, 4 x 32 GiB modules")


def test_gate_table_states_status_and_evidence_for_every_criterion():
    from basp.evidence import EvidenceLevel, ReceiptRegistry
    from basp.experiments.gate import Candidate, scheduler_receipt, simulation_receipts

    thresholds = load_json("hardware_gate")
    cand = Candidate("H", "Hierarchical", "30B MoE, single-user decoding", model="qwen3-30b-a3b")
    bindings = {"source": "s", "config": "c"}
    registry = ReceiptRegistry(simulation_receipts(cand, level=EvidenceLevel.E1, bindings=bindings, values={
        "throughput_ratio": 1.4, "energy_ratio": 0.8, "cost_efficiency_ratio": 0.3, "scaling_efficiency": 0.72}))
    registry.add(scheduler_receipt(cand, plans_per_batch=60.0, batch_time_s=1e-4,
                                   costs=thresholds["plan_evaluation_cost_s"], limit=0.05))  # fmt: skip
    decision = evaluate_gate(cand, thresholds, receipts=registry, calibration=calibration_status(None),
                             comparison_on_calibrated_profile=False, current=bindings)  # fmt: skip
    # three ratios fail even on uncalibrated evidence; nothing else can pass without better evidence
    assert decision.verdict == "NO-GO (3 criteria failed, 5 not evaluated)"
    text = report_v4.render_gate(decision)
    lines = text.splitlines()
    assert len(lines) == 2 + 8
    assert lines[0].startswith("| criterion | threshold | result | status | evidence | required")
    assert "| not evaluated |" in text and "| fail |" in text and "argument, not a measurement" in text
    assert "E1 uncalibrated simulation" in text and "E2 simulation" not in text and "| E3 |" in text


def test_report_compares_cache_policies_and_flags_stale_runtime_evidence(tmp_path):
    from basp.workloads.zoo import MODELS

    caching = report_v4.expert_cache_policies(MODELS["qwen3-30b-a3b"], "desktop_gpu12", steps=8,
                                              capacities_gib=(2,))  # fmt: skip
    policies = {r["policy"] for r in caching["rows"]}
    assert policies == {"LRU", "LFU", "cost-aware", "predictive", "none (CPU only)", "unbounded cache"}
    assert all(r["capacity_gib"] == 2 and r["ms_per_token"] > 0 for r in caching["rows"])
    # a recorded parity result produced by other runtime code is reported as stale
    folder = tmp_path / "verification" / "m"
    folder.mkdir(parents=True)
    record = {"model": "m", "file": {"name": "m.gguf"}, "verdict": "pass", "reasons": [], "numerical_error": 1e-6,
              "hidden": {"rel_l2_max": 1e-6, "layers_compared": 1}, "routing": None,
              "generation": {"identical": True, "tokens": 4}, "telemetry": {"decode": {"tokens_per_s": 1.0}},
              "runtime_fingerprint": "an older runtime"}  # fmt: skip
    (folder / "parity.json").write_text(json.dumps(record))
    [row] = report_v4.runtime_verification(tmp_path)
    assert row["stale"] and row["verdict"] == "pass"
    assert report_v4.runtime_verification(tmp_path / "nothing") == []
