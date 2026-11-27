"""Error statistics, the validation-only verdict, and recorded-routing helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from basp.calibration import (
    Measurement,
    MeasurementSet,
    build_trace,
    error_stats,
    evaluate_calibration,
    logits_to_routing,
    trace_statistics,
)
from basp.kernels import reference
from basp.workloads.trace import synth_trace


def test_error_stats_match_hand_computation():
    stats = error_stats([(110.0, 100.0), (90.0, 100.0), (100.0, 100.0), (150.0, 100.0)])
    # relative errors 0.1, 0.1, 0, 0.5 -> sorted 0, 0.1, 0.1, 0.5
    assert stats.n == 4 and stats.median == pytest.approx(0.1) and stats.worst == pytest.approx(0.5)
    assert stats.p95 == pytest.approx(0.1 + 0.85 * 0.4)  # linear interpolation at rank 2.85
    assert stats.mean_signed == pytest.approx(0.125)  # optimistic on average
    assert error_stats([(80.0, 100.0)]).mean_signed == pytest.approx(-0.2)  # pessimistic
    empty = error_stats([])
    assert empty.n == 0 and all(math.isnan(v) for v in (empty.median, empty.p95, empty.worst,
                                                         empty.mean_signed))  # fmt: skip
    with pytest.raises(ValueError, match="measured values must be positive"):
        error_stats([(1.0, 0.0)])


def measurements(n_validation: int = 8, n_calibration: int = 3) -> MeasurementSet:
    items = [Measurement(f"cal-{i}", "i9", "calibration", "decode", "qwen3-30b-a3b", 20.0 + i,
                         n_gpu_layers=i) for i in range(n_calibration)]  # fmt: skip
    items += [Measurement(f"val-{i}", "i9", "validation", "decode", "qwen3-30b-a3b", 40.0 + i,
                          n_gpu_layers=100 + i) for i in range(n_validation)]  # fmt: skip
    return MeasurementSet(items)


def scaled(factors: dict[str, float], default: float = 1.0):
    return lambda m: m.tokens_per_s * factors.get(m.id, factors.get(m.split, default))


def test_validation_split_alone_decides_the_status():
    ms = measurements()
    ok = evaluate_calibration(scaled({"validation": 1.05}), ms)
    assert ok.status == "pass" and ok.passed and ok.reasons == []
    assert ok.validation.n == 8 and ok.validation.median == pytest.approx(0.05)
    assert ok.thresholds == {"median_max": 0.10, "p95_max": 0.20, "min_validation": 8.0}

    # a perfect in-sample fit cannot rescue a failing validation split ...
    biased = evaluate_calibration(scaled({"validation": 1.15, "calibration": 1.0}), ms)
    assert biased.status == "fail" and biased.calibration.median == pytest.approx(0.0)
    assert any("median error 15.0% exceeds 10%" in r for r in biased.reasons)
    assert not any("p95" in r for r in biased.reasons)
    assert any("in-sample" in n and "does not affect" in n for n in biased.notes)
    # ... and a poor in-sample fit does not fail a good validation result
    reverse = evaluate_calibration(scaled({"validation": 0.97, "calibration": 2.0}), ms)
    assert reverse.status == "pass" and reverse.calibration.median == pytest.approx(1.0)


def test_one_bad_tail_point_fails_on_p95_only():
    report = evaluate_calibration(scaled({"validation": 1.02, "val-3": 1.60}), measurements())
    assert report.status == "fail"
    assert report.validation.median == pytest.approx(0.02)
    assert report.validation.p95 == pytest.approx(0.02 + 0.65 * 0.58)
    assert len(report.reasons) == 1 and "p95 error" in report.reasons[0]
    row = next(r for r in report.per_measurement if r["id"] == "val-3")
    assert row == {"id": "val-3", "machine": "i9", "model": "qwen3-30b-a3b", "kind": "decode",
                   "split": "validation", "measured": 43.0, "repetitions": 1, "n_gpu_layers": 103,
                   "n_cpu_moe": 0, "context_len": 0,
                   "predicted": pytest.approx(68.8), "rel_error": pytest.approx(0.6)}  # fmt: skip


def test_insufficient_evidence_is_not_evaluated():
    few = evaluate_calibration(scaled({}), measurements(n_validation=7))
    assert few.status == "not_evaluated" and "only 7 distinct validation" in few.reasons[0]
    assert "at least 8" in few.reasons[0]
    none = evaluate_calibration(scaled({}), measurements(n_validation=0))
    assert none.status == "not_evaluated" and none.validation.n == 0

    def partial(m):
        return None if m.id in ("val-2", "val-5") else m.tokens_per_s

    missing = evaluate_calibration(partial, measurements())
    assert missing.status == "not_evaluated"
    assert "['val-2', 'val-5']" in missing.reasons[0]
    assert next(r for r in missing.per_measurement if r["id"] == "val-2")["predicted"] is None
    relaxed = evaluate_calibration(scaled({}), measurements(n_validation=3), min_validation=3)
    assert relaxed.status == "pass"
    payload = none.to_json()
    assert payload["validation"]["median"] is None and payload["status"] == "not_evaluated"
    assert "calibration_in_sample" in payload  # labelled as in-sample


# -- recorded routing ------------------------------------------------------------------------
def test_logits_to_routing_matches_the_reference_router():
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((5, 16))
    step = logits_to_routing(logits, 4)
    idx, coeffs = reference.route_topk(logits, 4)
    assert step.experts == tuple(tuple(int(e) for e in row) for row in idx)
    np.testing.assert_allclose(step.coeffs, coeffs)
    assert all(sum(row) == pytest.approx(1.0) for row in step.coeffs)
    raw = logits_to_routing(logits, 4, normalize=False)
    assert all(sum(row) < 1.0 for row in raw.coeffs)
    tie = logits_to_routing(np.array([0.0, 2.0, 2.0, 1.0]), 2)  # 1-D input, tie -> lower index
    assert tie.experts == ((1, 2),)
    with pytest.raises(ValueError, match="non-finite"):
        logits_to_routing(np.array([[0.0, np.nan]]), 1)
    with pytest.raises(ValueError, match="rows, n_experts"):
        logits_to_routing(np.zeros((2, 3, 4)), 1)


def test_build_trace_validates_shapes():
    rng = np.random.default_rng(4)
    logits = [[rng.standard_normal((3, 8)) for _ in range(2)] for _ in range(5)]
    trace = build_trace(logits, 2)
    assert (trace.n_steps, trace.n_layers, trace.n_experts, trace.top_k, trace.batch) == (5, 2, 8, 2, 3)
    assert trace.at(4, 1) == logits_to_routing(logits[4][1], 2)
    with pytest.raises(ValueError, match="step 1 has 1 layers"):
        build_trace([logits[0], logits[1][:1]], 2)
    with pytest.raises(ValueError, match="logits shape"):
        build_trace([logits[0], [np.zeros((4, 8)), np.zeros((4, 8))]], 2)
    with pytest.raises(ValueError, match="at least one step"):
        build_trace([], 2)


def test_trace_statistics_expose_stickiness_skew_and_concentration():
    sticky = trace_statistics(synth_trace(40, 3, 64, 4, stickiness=1.0, zipf_s=0.0, seed=1))
    fresh = trace_statistics(synth_trace(400, 3, 64, 4, stickiness=0.0, zipf_s=0.0, seed=1))
    skewed = trace_statistics(synth_trace(400, 3, 64, 4, stickiness=0.0, zipf_s=1.5, seed=1))
    assert sticky["stickiness"] == 1.0
    assert fresh["stickiness"] == pytest.approx(4 / 64, abs=0.03)  # chance level for uniform routing
    assert skewed["stickiness"] > fresh["stickiness"]  # popular experts recur by chance
    assert fresh["usage_entropy_bits_max"] == 6.0
    assert fresh["usage_entropy_bits_mean"] == pytest.approx(6.0, abs=0.1)
    assert skewed["usage_entropy_bits_mean"] < fresh["usage_entropy_bits_mean"] - 0.5
    assert fresh["top_decile_share"] == pytest.approx(7 / 64, abs=0.04)  # 7 of 64 experts
    assert skewed["top_decile_share"] > 2 * fresh["top_decile_share"]
    assert len(skewed["usage_entropy_bits"]) == len(skewed["top_decile_share_per_layer"]) == 3
    single = trace_statistics(synth_trace(1, 2, 16, 2, seed=0))
    assert single["stickiness"] is None and single["n_steps"] == 1
