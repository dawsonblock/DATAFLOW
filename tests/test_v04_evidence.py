"""v0.4 evidence system: receipts, metric-specific qualification, identity binding, and the
adversarial cases from the v0.4 plan (invalid values, repeated runs, per-machine scoping,
changed executables and weights, stale profiles, missing quality, zero plan evaluations)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from basp.calibration import Measurement, MeasurementSet, evaluate_by_machine, evaluate_calibration
from basp.evidence import (
    CONFIG_DIR,
    EvidenceLevel,
    Receipt,
    ReceiptRegistry,
    directory_manifest,
    file_identity,
    path_hash,
    profile_identity,
    provenance,
    qualify,
    stale_reasons,
    value_problems,
)
from basp.experiments.calibrate import (
    calibration_status,
    predict,
    profile_qualification,
    staleness,
)
from basp.experiments.gate import Candidate, evaluate_gate, scheduler_receipt, simulation_receipts
from basp.experiments.profiles import load_json
from benchmarks.run_calibration import run_key

E0, E1, E2, E3 = EvidenceLevel.E0, EvidenceLevel.E1, EvidenceLevel.E2, EvidenceLevel.E3
THRESHOLDS = load_json("hardware_gate")
MACHINE = "i9-11900kf-3080ti"
REGIME = "qwen3-30b-a3b, single-user decoding"
CURRENT = {"source": "src-1", "config": "cfg-1", "profile": "prof-1", "validation": "val-1"}
MEASURED = {"machine": "i9", "model": "qwen3-30b-a3b", "runtime": "rt-1", "benchmark": "bench-1"}
CAND = Candidate("H", "Hierarchical", REGIME, model="qwen3-30b-a3b")


@pytest.fixture(scope="module")
def passing_calibration(tmp_path_factory):
    path = tmp_path_factory.mktemp("gate") / "m.json"
    ms = [Measurement(id=f"m{i}", machine=MACHINE, split="validation", kind="decode", model="qwen3-8b",
                      tokens_per_s=1.0, bits_per_weight=4.85, n_gpu_layers=4 * i) for i in range(8)]  # fmt: skip
    MeasurementSet([replace(m, tokens_per_s=predict(m)) for m in ms]).save(path)
    report = calibration_status(path)
    assert report.status == "pass"
    return report


def full_registry(**overrides) -> ReceiptRegistry:
    """Receipts that would open the gate: E2 simulation bound to the current identities, E3
    measurements bound to the regime's model."""
    values = {"throughput_ratio": 2.5, "energy_ratio": 2.2, "cost_efficiency_ratio": 2.1, "scaling_efficiency": 0.75}
    values.update({k: v for k, v in overrides.items() if k in values})
    registry = ReceiptRegistry(simulation_receipts(CAND, level=E2, bindings=CURRENT, values=values))
    measured = {"scheduler_overhead": 0.01, "numerical_error": 5e-4, "model_quality": 1.002}
    measured.update({k: v for k, v in overrides.items() if k in measured})
    for metric, value in measured.items():
        registry.add(Receipt(metric, value, E3, "H", REGIME, None, MEASURED, repetitions=5))
    return registry


def gate(calibration, registry, cand=CAND, **kw):
    args = {"comparison_on_calibrated_profile": True, "current": CURRENT}
    args.update(kw)
    return evaluate_gate(cand, THRESHOLDS, receipts=registry, calibration=calibration, **args)


def by_name(decision):
    return {c.name: c for c in decision.criteria}


# -- the gate decides from receipts ------------------------------------------------------------
def test_gate_opens_only_on_e2_simulation_and_e3_measurements(passing_calibration):
    decision = gate(passing_calibration, full_registry())
    assert decision.go and decision.verdict == "GO"
    assert by_name(decision)["Throughput improvement"].level is E2
    # replace the measured scheduler overhead by an estimate: the gate closes
    registry = ReceiptRegistry(r for r in full_registry() if r.metric != "scheduler_overhead")
    registry.add(scheduler_receipt(CAND, plans_per_batch=40, batch_time_s=2e-3,
                                   costs=THRESHOLDS["plan_evaluation_cost_s"], limit=0.05))  # fmt: skip
    estimated = gate(passing_calibration, registry)
    overhead = by_name(estimated)["Scheduler overhead"]
    assert overhead.status == "not_evaluated" and overhead.level is E0 and "ns per evaluation" in overhead.result
    assert estimated.verdict == "NO-GO (0 criteria failed, 1 not evaluated)"


def test_uncalibrated_simulation_is_e1_and_cannot_pass(passing_calibration):
    values = {"throughput_ratio": 3.0, "energy_ratio": 3.0, "cost_efficiency_ratio": 3.0, "scaling_efficiency": 0.9}
    e1 = ReceiptRegistry(simulation_receipts(CAND, level=E1, bindings={"source": "src-1", "config": "cfg-1"},
                                             values=values))  # fmt: skip
    criteria = by_name(gate(passing_calibration, e1))
    for name in ("Throughput improvement", "Energy efficiency", "Cost efficiency", "Four-module scaling efficiency"):
        assert criteria[name].status == "not_evaluated" and criteria[name].level is E1
        assert "E2 required" in criteria[name].result
    # an E1 receipt that misses the threshold fails anyway
    low = ReceiptRegistry(simulation_receipts(CAND, level=E1, bindings={"source": "src-1", "config": "cfg-1"},
                                              values={"cost_efficiency_ratio": 0.3}))  # fmt: skip
    assert by_name(gate(passing_calibration, low))["Cost efficiency"].status == "fail"


def test_a_receipt_qualifies_only_its_own_metric_candidate_and_regime(passing_calibration):
    only_throughput = ReceiptRegistry(simulation_receipts(CAND, level=E2, bindings=CURRENT,
                                                          values={"throughput_ratio": 2.5}))  # fmt: skip
    criteria = by_name(gate(passing_calibration, only_throughput))
    assert criteria["Throughput improvement"].passed
    assert criteria["Energy efficiency"].status == "not_evaluated"
    assert "no energy_ratio receipt for H" in criteria["Energy efficiency"].result
    # scaling measured for E, or in another regime, never qualifies H here
    other = ReceiptRegistry(simulation_receipts(Candidate("E", "", REGIME), level=E2, bindings=CURRENT,
                                                values={"scaling_efficiency": 0.95}))  # fmt: skip
    other.extend(simulation_receipts(Candidate("H", "", "70B dense"), level=E2, bindings=CURRENT,
                                     values={"scaling_efficiency": 0.95}))  # fmt: skip
    assert by_name(gate(passing_calibration, other))["Four-module scaling efficiency"].status == "not_evaluated"


def test_receipts_bound_to_changed_identities_are_stale(passing_calibration):
    registry = full_registry()
    for change in ({"source": "src-2"}, {"profile": "prof-2"}, {"validation": "val-2"}, {"validation": None}):
        criteria = by_name(gate(passing_calibration, registry, current={**CURRENT, **change}))
        throughput = criteria["Throughput improvement"]
        assert throughput.status == "not_evaluated" and "stale" in throughput.result, change
    # changed weights invalidate measured evidence too
    rt = {**CURRENT, "model": "qwen3-30b-a3b-requantised"}
    assert by_name(gate(passing_calibration, registry, current=rt))["Numerical correctness"].status == "not_evaluated"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1e-3, True, "0.001"])
def test_invalid_numerical_error_is_rejected(passing_calibration, bad):
    assert value_problems("numerical_error", bad)
    registry = ReceiptRegistry(r for r in full_registry() if r.metric != "numerical_error")
    registry.add(Receipt("numerical_error", bad, E3, "H", REGIME, None, MEASURED))
    numeric = by_name(gate(passing_calibration, registry))["Numerical correctness"]
    assert numeric.status == "not_evaluated" and "rejected" in numeric.result
    assert not gate(passing_calibration, registry).go


def test_numerical_error_on_another_model_does_not_qualify_the_regime(passing_calibration):
    registry = ReceiptRegistry(r for r in full_registry() if r.metric != "numerical_error")
    registry.add(Receipt("numerical_error", 1e-6, E3, "H", REGIME, None, {**MEASURED, "model": "tiny-random"}))
    numeric = by_name(gate(passing_calibration, registry))["Numerical correctness"]
    assert numeric.status == "not_evaluated" and "about model 'tiny-random'" in numeric.result


def test_missing_model_quality_is_not_evaluated(passing_calibration):
    registry = ReceiptRegistry(r for r in full_registry() if r.metric != "model_quality")
    quality = by_name(gate(passing_calibration, registry))["Model quality"]
    assert quality.status == "not_evaluated" and quality.level is None and quality.required is E3
    assert "not measured" in quality.result and "argument, not a measurement" in quality.result
    worse = full_registry(model_quality=1.05)
    assert by_name(gate(passing_calibration, worse))["Model quality"].status == "fail"


def test_zero_plan_evaluations_give_a_finite_receipt():
    receipt = scheduler_receipt(CAND, plans_per_batch=0, batch_time_s=1e-3,
                                costs=THRESHOLDS["plan_evaluation_cost_s"], limit=0.05)  # fmt: skip
    assert receipt.value == 0.0 and receipt.raw["budget_ns"] is None and receipt.problems() == []
    json.dumps(receipt.to_json(), allow_nan=False)
    q = qualify("scheduler_overhead", ReceiptRegistry([receipt]), candidate="H", regime=REGIME,
                meets=lambda x: x <= 0.05, required=E3)  # fmt: skip
    assert q.status == "not_evaluated" and q.level is E0  # an estimate never passes
    with pytest.raises(ValueError):
        scheduler_receipt(CAND, plans_per_batch=10, batch_time_s=0.0, costs=THRESHOLDS["plan_evaluation_cost_s"],
                          limit=0.05)  # fmt: skip


def test_receipts_are_content_addressed_and_need_their_bindings(tmp_path):
    a = Receipt("throughput_ratio", 2.0, E2, "H", REGIME, "A", CURRENT)
    assert a.receipt_id == Receipt("throughput_ratio", 2.0, E2, "H", REGIME, "A", dict(CURRENT)).receipt_id
    assert a.receipt_id != replace(a, value=2.01).receipt_id
    assert a.unit == "x" or a.unit
    missing = Receipt("throughput_ratio", 2.0, E2, "H", REGIME, "A", {"source": "s", "config": "c"})
    assert "lacks binding(s) ['profile', 'validation']" in missing.problems()[0]
    assert Receipt("numerical_error", 1e-4, E3, "H", REGIME, None, {"model": "m"}).problems()
    registry = ReceiptRegistry([a, a, missing])
    assert len(registry) == 2
    registry.save(tmp_path / "receipts.json")
    assert [r.receipt_id for r in ReceiptRegistry.load(tmp_path / "receipts.json")] == [r.receipt_id for r in registry]


# -- calibration: repetitions, machines, executables, weights, profiles ---------------------------
def val(machine: str, i: int, tps: float, **kw) -> Measurement:
    return Measurement(id=f"{machine}-{i}-{kw.pop('rep', 0)}", machine=machine, split="validation", kind="decode",
                       model="qwen3-8b", tokens_per_s=tps, n_gpu_layers=i, **kw)  # fmt: skip


def test_repetitions_of_one_workload_count_once():
    eight = MeasurementSet([val("i9", 1, 10.0 + 0.1 * r, rep=r) for r in range(8)])
    report = evaluate_calibration(lambda m: 10.0, eight)
    assert report.validation.n == 1 and report.status == "not_evaluated"
    assert report.per_measurement[0]["repetitions"] == 8
    assert report.per_measurement[0]["measured"] == pytest.approx(10.35)  # median of the repeats
    assert any("merged" in n for n in report.notes)


def test_each_machine_is_evaluated_on_its_own():
    good = [val("i9", i, 10.0) for i in range(8)]
    bad = [val("m2", i, 10.0) for i in range(8)]
    both = MeasurementSet(good + bad)

    def predict_(m):
        return 10.0 if m.machine == "i9" else 15.0

    report = evaluate_by_machine(predict_, both)
    assert report.by_machine["i9"].status == "pass" and report.by_machine["m2"].status == "fail"
    assert report.status == "fail" and all(r.startswith("m2:") for r in report.reasons)
    # a machine with too few validation benchmarks does not borrow another machine's
    few = MeasurementSet(good + bad[:3])
    report = evaluate_by_machine(lambda m: 10.0, few)
    assert report.by_machine["i9"].status == "pass" and report.by_machine["m2"].status == "not_evaluated"
    assert report.status == "not_evaluated"
    # combined over machines, the old evaluation would have seen 11 benchmarks and passed
    assert evaluate_calibration(lambda m: 10.0, few).status == "pass"


def measured(i: int, exe: str, weights: str) -> Measurement:
    return val("i9", i, 10.0, provenance={"executable_sha256": exe, "model_file": {"sha256": weights}})


def test_a_modified_executable_or_weights_invalidate_earlier_measurements():
    identity = {"i9": {"executable_sha256": "exe-2", "models": {"qwen3-8b": "w-1"}}}
    stale = staleness(identity)
    assert stale(measured(1, "exe-2", "w-1")) is None
    assert "executable changed" in stale(measured(1, "exe-1", "w-1"))
    assert "weights changed" in stale(measured(1, "exe-2", "w-0"))
    ms = MeasurementSet([measured(i, "exe-1", "w-1") for i in range(8)] + [measured(9, "exe-2", "w-1")])
    report = evaluate_by_machine(lambda m: 10.0, ms, exclude=stale)
    assert report.validation.n == 1 and report.status == "not_evaluated" and len(report.excluded) == 8
    # and the runner re-runs a configuration with a changed executable, weights or repetitions
    run = {"split": "validation", "model": "qwen3-8b", "ngl": 12, "depth": 0, "n_cpu_moe": 0, "n_prompt": 512,
           "n_gen": 128, "repetitions": 3}  # fmt: skip
    base = run_key(run, identity["i9"])
    assert base != run_key(run, {**identity["i9"], "executable_sha256": "exe-3"})
    assert base != run_key(run, {**identity["i9"], "models": {"qwen3-8b": "w-9"}})
    assert base != run_key({**run, "repetitions": 5}, identity["i9"])


def test_file_identity_hashes_the_whole_content(tmp_path):
    path = tmp_path / "weights.gguf"
    path.write_bytes(b"GGUF" + bytes(1000))
    first = file_identity(path)
    data = bytearray(path.read_bytes())
    data[900] = 1  # change one weight byte, far from the header
    path.write_bytes(bytes(data))
    assert file_identity(path)["sha256"] != first["sha256"] and file_identity(path)["size"] == first["size"]


def test_a_profile_changed_after_validation_is_not_qualified(passing_calibration, tmp_path):
    profile = load_json("i9_11900kf_rtx3080ti")
    record = tmp_path / MACHINE / "calibration.json"
    record.parent.mkdir()
    assert profile_qualification(passing_calibration, tmp_path)["i9_11900kf_rtx3080ti"]["reason"].startswith(
        "no calibration record")  # fmt: skip
    record.write_text(json.dumps({"fitted_profile_sha": profile_identity(profile)}))
    assert profile_qualification(passing_calibration, tmp_path)["i9_11900kf_rtx3080ti"]["qualified"]
    tuned = json.loads(json.dumps(profile))
    tuned["devices"][0]["mem_bandwidth"] = tuned["devices"][0].get("mem_bandwidth", 1.0) * 1.1
    record.write_text(json.dumps({"fitted_profile_sha": profile_identity(tuned)}))
    q = profile_qualification(passing_calibration, tmp_path)["i9_11900kf_rtx3080ti"]
    assert not q["qualified"] and q["reason"] == "the profile changed after it was validated"
    # comments do not change a profile's identity
    assert profile_identity({**profile, "_note": "edited comment"}) == profile_identity(profile)


def test_no_plan_and_no_measurements_are_not_evaluated():
    empty = evaluate_by_machine(lambda m: 1.0, MeasurementSet())
    assert empty.status == "not_evaluated" and empty.validation.n == 0 and empty.by_machine == {}
    json.dumps(empty.to_json(), default=float)


# -- provenance of files and directories ---------------------------------------------------------
def test_directory_manifests_change_with_any_file(tmp_path):
    folder = tmp_path / "evidence"
    (folder / "m").mkdir(parents=True)
    (folder / "m" / "a.json").write_text("{}")
    first = directory_manifest(folder)
    assert [f["path"] for f in first["files"]] == ["m/a.json"]
    assert path_hash(folder) == first["root"] and path_hash(tmp_path / "missing") is None
    recorded = provenance(configs=["hardware_gate"], inputs={"evidence": folder}, parameters={})
    (folder / "m" / "b.json").write_text("[]")
    assert directory_manifest(folder)["root"] != first["root"]
    now = provenance(configs=["hardware_gate"], inputs={"evidence": folder}, parameters={})
    assert any("evidence" in r for r in stale_reasons(recorded, now))


# -- calibration coverage and performance curves ------------------------------------------------
def test_coverage_lists_the_regimes_each_validation_split_exercises():
    from basp.calibration import coverage, regime_covered, regimes_of
    from basp.calibration.plan import plan_runs

    plan = json.loads((CONFIG_DIR / "calibration_plan.json").read_text())
    desktop = coverage(plan_runs(plan, MACHINE))
    assert desktop["complete"] and desktop["calibration_only"] == []
    assert {"experts-on-cpu", "moe", "split-layers", "long-context"} <= set(desktop["validated"])
    colab = coverage(plan_runs(plan, "colab-t4"))
    assert "cpu-only" in colab["calibration_only"] and "cpu-only" in colab["missing"]  # in-sample is not validation
    assert regime_covered({"moe", "gpu-only"}, coverage(plan_runs(plan, "macbook-m2"))) == ["moe"]
    assert regimes_of({"model": "qwen3-30b-a3b", "ngl": 99, "n_cpu_moe": 24, "split": "validation"}) == {
        "gpu-only", "experts-on-cpu", "moe"}  # fmt: skip


def test_performance_curves_are_validated_before_use():
    from basp.calibration import PerformanceCurve, validate_curve

    good = PerformanceCurve("i9", "pcie-gpu", "link_transfer", "bytes", "bytes/s",
                            ((4096.0, 2e9), (1 << 20, 12e9), (1 << 26, 24e9)), ("hw.json",))  # fmt: skip
    assert validate_curve(good) == []
    assert good.interpolate(1 << 20) == pytest.approx(12e9)
    assert good.interpolate(1) == 2e9 and good.interpolate(1 << 40) == 24e9  # clamped, never extrapolated
    assert PerformanceCurve.from_json(json.loads(json.dumps(good.to_json()))) == good
    bad = [replace(good, points=((1.0, float("nan")), (2.0, 1.0))), replace(good, points=((2.0, 1.0), (1.0, 2.0))),
           replace(good, y_unit="furlongs"), replace(good, points=((1.0, 1.0),)), replace(good, kind="vibes")]
    for curve in bad:
        assert validate_curve(curve)
        with pytest.raises(ValueError, match="invalid curve"):
            curve.interpolate(1.5)
