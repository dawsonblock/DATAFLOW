"""v0.6 evidence repairs (audit findings E1 and S4), as behaviour checks.

E1: receipts are combined by metric-specific, conservative aggregation.  Notes, hashes and
    filing order cannot change a verdict, and an applicable failing measurement cannot disappear.
S4: validated simulation counts only within the scope successful validation observations
    covered: a dense-only validation does not qualify a mixture-of-experts comparison on the same
    machine and the same validated profile; timing does not qualify energy; a plan covers nothing.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace

import pytest

from basp.calibration import (
    Measurement,
    MeasurementSet,
    coverage_gaps,
    observed_coverage,
    required_regimes,
)
from basp.evidence import (
    METRICS,
    EvidenceLevel,
    Receipt,
    ReceiptRegistry,
    profile_identity,
    qualify,
)
from basp.experiments.calibrate import calibration_status, predict, profile_qualification
from basp.experiments.gate import Candidate, evaluate_gate, simulation_receipts
from basp.experiments.profiles import load_json
from basp.workloads.zoo import MODELS

E1, E2, E3 = EvidenceLevel.E1, EvidenceLevel.E2, EvidenceLevel.E3
THRESHOLDS = load_json("hardware_gate")
MACHINE, PROFILE = "i9-11900kf-3080ti", "i9_11900kf_rtx3080ti"
REGIME = "regime"
MEASURED = {"machine": "m", "model": "the-model", "runtime": "rt", "benchmark": "b"}
CURRENT = {"source": "s", "config": "c", "profile": "p", "validation": "v"}


def numeric(value, note="", **kw):
    return Receipt("numerical_error", value, E3, "H", REGIME, None, MEASURED, note=note, **kw)


def verdict(receipts, metric="numerical_error", meets=lambda x: x <= 0.002, required=E3):
    q = qualify(metric, ReceiptRegistry(receipts), candidate="H", regime=REGIME, meets=meets, required=required)
    return q.status, q.value, q.level, q.conflict, tuple(q.receipt_ids)


# -- E1 ------------------------------------------------------------------------------------------
def test_an_applicable_failing_measurement_cannot_be_outvoted_or_dropped():
    good = [numeric(5e-4, repetitions=50), numeric(4e-4, raw={"run": 2}), numeric(1e-6, raw={"run": 3})]
    bad = numeric(5e-2, raw={"run": "the failing one"})
    assert verdict(good)[0] == "pass" and verdict(good)[1] == 5e-4  # the worst of the passing ones is reported
    for order in itertools.permutations([*good, bad]):
        status, value, level, conflict, ids = verdict(order)
        assert (status, value, level, conflict) == ("fail", 5e-2, E3, True) and len(ids) == 4


def test_notes_hashes_and_filing_order_cannot_change_a_verdict():
    base = [numeric(5e-4), numeric(1.9e-3, raw={"run": 2}), numeric(2.5e-3, raw={"run": 3})]
    reference = verdict(base)[:4]
    assert reference == ("fail", 2.5e-3, E3, True)
    for i in range(30):  # every note changes every receipt id, and so any hash-based ranking
        annotated = [replace(r, note=f"note {i}-{j}") for j, r in enumerate(base)]
        for order in itertools.permutations(annotated):
            assert verdict(order)[:4] == reference
    # the reported text is as stable as the verdict
    decisions = []
    for notes in (("a", "b", "c"), ("zzz", "", "m")):
        registry = ReceiptRegistry(replace(r, note=n, bindings={**MEASURED, "model": "other"})
                                   for r, n in zip(base, notes))  # fmt: skip
        gate = evaluate_gate(Candidate("H", "Hier", REGIME, model="the-model"), THRESHOLDS, receipts=registry,
                             calibration=calibration_status(None), comparison_on_calibrated_profile=False)  # fmt: skip
        decisions.append([(c.status, c.level) for c in gate.criteria])
    assert decisions[0] == decisions[1]


def test_the_same_observation_counts_once_however_it_is_annotated():
    a = numeric(5e-4, note="first filing")
    b = numeric(5e-4, note="filed again with a comment")
    assert a.receipt_id != b.receipt_id and a.measurement_id == b.measurement_id
    status, value, _, conflict, ids = verdict([a, b, b, a])
    assert (status, value, conflict) == ("pass", 5e-4, False) and len(ids) == 1
    assert numeric(5e-4, raw={"run": 2}).measurement_id != a.measurement_id  # different data: another observation


def test_ratios_use_the_lowest_value_and_errors_the_highest():
    ratios = [Receipt("throughput_ratio", v, E2, "H", REGIME, "A", CURRENT, raw={"i": i}) for i, v in enumerate((2.6, 1.7, 3.1))]
    assert verdict(ratios, "throughput_ratio", lambda x: x >= 2.0, E2)[:2] == ("fail", 1.7)
    assert verdict(ratios[::2], "throughput_ratio", lambda x: x >= 2.0, E2)[:2] == ("pass", 2.6)
    assert METRICS["throughput_ratio"].worst([2.0, 3.0]) == 2.0 and METRICS["numerical_error"].worst([1e-3, 1e-2]) == 1e-2
    # a measurement supersedes a simulation of the same quantity; within a level nothing is superseded
    mixed = [Receipt("throughput_ratio", 0.5, E1, "H", REGIME, "A", {"source": "s", "config": "c"}),
             Receipt("throughput_ratio", 2.2, E3, "H", REGIME, "A", MEASURED)]  # fmt: skip
    assert verdict(mixed, "throughput_ratio", lambda x: x >= 2.0, E2)[:3] == ("pass", 2.2, E3)


# -- S4 ------------------------------------------------------------------------------------------
def scored(tmp_path, models, *, depth=0):
    """Validation measurements the simulator predicts exactly, with a recorded profile fingerprint."""
    ms = []
    for model in models:
        layers = MODELS[model].n_layers
        for i, ngl in enumerate((0, layers // 3, 2 * layers // 3, 99)):
            for kind in ("decode", "prefill"):
                m = Measurement(id=f"{model}-{ngl}-{kind}-{i}", machine=MACHINE, split="validation", kind=kind,
                                model=model, tokens_per_s=1.0, bits_per_weight=4.85, n_gpu_layers=ngl,
                                context_len=depth, batch=512 if kind == "prefill" else 1)  # fmt: skip
                predicted = predict(m)
                if predicted is not None:  # e.g. the whole 30B model on a 12 GiB GPU: does not fit, no run
                    ms.append(replace(m, tokens_per_s=predicted))
    folder = tmp_path / MACHINE
    folder.mkdir(parents=True, exist_ok=True)
    MeasurementSet(ms).save(folder / "measurements.json")
    (folder / "calibration.json").write_text(json.dumps({"fitted_profile_sha": profile_identity(load_json(PROFILE))}))
    report = calibration_status(tmp_path)
    assert report.status == "pass"
    return report


def e2_registry(cand, **values):
    defaults = {"throughput_ratio": 2.5, "energy_ratio": 2.4, "cost_efficiency_ratio": 2.3, "scaling_efficiency": 0.8}
    return ReceiptRegistry(simulation_receipts(cand, level=E2, bindings=CURRENT, values={**defaults, **values}))


def test_dense_only_validation_does_not_qualify_a_mixture_of_experts_comparison(tmp_path):
    report = scored(tmp_path, ["qwen3-8b"])
    qualified = profile_qualification(report, tmp_path)[PROFILE]
    assert qualified["qualified"]  # the machine's profile *is* validated ...
    observed = observed_coverage(report.by_machine[MACHINE].per_measurement)
    assert {"dense", "cpu-only", "split-layers", "gpu-only", "decode", "prefill"} <= set(observed)
    assert "moe" not in observed and "experts-on-cpu" not in observed  # ... but not for this
    moe = MODELS["qwen3-30b-a3b"]
    requires = required_regimes(moe, gpu_gib=12)
    assert requires == ("decode", "experts-on-cpu", "moe")
    cand = Candidate("H", "Hierarchical", "30B MoE on a 12 GiB GPU", model="qwen3-30b-a3b", requires=requires)
    gate = evaluate_gate(cand, THRESHOLDS, receipts=e2_registry(cand), calibration=report,
                         comparison_on_calibrated_profile=True, current=CURRENT, coverage=observed)  # fmt: skip
    by_name = {c.name: c for c in gate.criteria}
    calibration = by_name["Simulator calibration error"]
    assert calibration.status == "not_evaluated" and "covers experts-on-cpu, moe" in calibration.result
    for name in ("Throughput improvement", "Cost efficiency", "Four-module scaling efficiency"):
        assert by_name[name].status == "not_evaluated" and "validation does not cover experts-on-cpu, moe" in by_name[name].result
    # a miss is still a miss
    failing = evaluate_gate(cand, THRESHOLDS, receipts=e2_registry(cand, throughput_ratio=1.2), calibration=report,
                            comparison_on_calibrated_profile=True, current=CURRENT, coverage=observed)  # fmt: skip
    assert {c.name: c.status for c in failing.criteria}["Throughput improvement"] == "fail"
    # the same evidence does qualify a dense comparison that does not fit the GPU
    dense = MODELS["qwen3-14b"]
    dense_requires = required_regimes(dense, gpu_gib=4)
    assert dense_requires == ("decode", "dense", "split-layers")
    dense_cand = Candidate("E", "Tensor-parallel", "14B dense on a 4 GiB GPU", model="qwen3-14b", requires=dense_requires)
    ok = {c.name: c for c in evaluate_gate(
        dense_cand, THRESHOLDS, receipts=e2_registry(dense_cand), calibration=report,
        comparison_on_calibrated_profile=True, current=CURRENT, coverage=observed).criteria}  # fmt: skip
    assert ok["Simulator calibration error"].status == "pass" and ok["Throughput improvement"].status == "pass"
    # timing validation says nothing about energy
    assert ok["Energy efficiency"].status == "not_evaluated" and "does not cover energy" in ok["Energy efficiency"].result


def test_validation_including_the_regime_does_qualify_it(tmp_path):
    report = scored(tmp_path, ["qwen3-8b", "qwen3-30b-a3b"])
    observed = observed_coverage(report.by_machine[MACHINE].per_measurement)
    assert observed["moe"]["n"] >= 2 and "experts-on-cpu" not in observed  # layer offload only, no --n-cpu-moe run
    assert coverage_gaps(("decode", "moe"), observed, min_per_regime=2) == []
    assert coverage_gaps(("decode", "experts-on-cpu", "moe"), observed, min_per_regime=2) == ["experts-on-cpu"]
    assert coverage_gaps(("decode", "moe"), observed, min_per_regime=2, metric="energy_ratio") == ["energy"]
    assert coverage_gaps(("decode", "long-context"), observed) == ["long-context"]  # short contexts only
    deep = scored(tmp_path / "deep", ["qwen3-8b"], depth=4096)
    assert "long-context" in observed_coverage(deep.by_machine[MACHINE].per_measurement)


def test_plans_and_failed_runs_confer_no_coverage():
    rows = [{"split": "validation", "model": "qwen3-30b-a3b", "kind": "decode", "n_gpu_layers": 99, "n_cpu_moe": 24,
             "context_len": 0, "rel_error": None},  # planned or failed: never scored
            {"split": "calibration", "model": "qwen3-30b-a3b", "kind": "decode", "n_gpu_layers": 99, "n_cpu_moe": 24,
             "context_len": 0, "rel_error": 0.01}]  # in-sample: not validation
    assert observed_coverage(rows) == {} and observed_coverage([]) == {}
    one = observed_coverage([{**rows[0], "rel_error": 0.03}])
    assert one["experts-on-cpu"] == {"n": 1, "median": 0.03, "worst": 0.03}
    assert coverage_gaps(("moe",), one, min_per_regime=2) == ["moe"]  # one observation is not enough
    assert coverage_gaps(("moe",), one, min_per_regime=1) == []
    assert coverage_gaps(("moe",), None) == ["moe"]
    assert THRESHOLDS["min_validation_per_regime"] == 2
    with pytest.raises(KeyError):
        required_regimes(MODELS["no-such-model"], gpu_gib=12)
