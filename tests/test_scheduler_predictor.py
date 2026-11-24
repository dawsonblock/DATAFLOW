import numpy as np
import pytest

from basp.scheduler.predictor import ExpertPredictor
from basp.workloads.trace import RoutingStep, RoutingTrace, synth_trace

FREQ_ONLY = {"use_recency": False, "use_transitions": False}


def _held_out_hit_rate(stickiness, flags, *, n=16, seed=3):
    trace = synth_trace(300, 4, 128, 8, stickiness=stickiness, seed=seed)
    train, held_out = trace.split(200)
    predictor = ExpertPredictor(4, 128, **flags)
    predictor.fit(train)
    predictor.reset_streams()
    return predictor.evaluate(held_out, n)["hit_rate"]


def _cycle_trace(n_experts=6, n_steps=300):
    steps = [[RoutingStep(((i % n_experts,),), ((1.0,),))] for i in range(n_steps)]
    return RoutingTrace(1, n_experts, 1, 1, steps)


def test_frequency_without_decay_equals_empirical_rate():
    trace = synth_trace(150, 3, 32, 4, batch=2, stickiness=0.3, seed=1)
    predictor = ExpertPredictor(3, 32, decay=1.0)
    assert predictor.frequency(0, 0) == 0.0
    predictor.fit(trace)
    empirical = trace.expert_frequencies()
    got = np.array([[predictor.frequency(layer, e) for e in range(32)] for layer in range(3)])
    np.testing.assert_allclose(got, empirical, atol=1e-12)


def test_decayed_frequency_converges_on_stationary_trace():
    trace = synth_trace(3000, 1, 16, 2, stickiness=0.0, seed=2)
    predictor = ExpertPredictor(1, 16, decay=0.999)
    predictor.fit(trace)
    got = np.array([predictor.frequency(0, e) for e in range(16)])
    np.testing.assert_allclose(got, trace.expert_frequencies()[0], atol=0.04)
    assert got.sum() == pytest.approx(2.0)


def test_probabilities_stay_within_unit_interval():
    trace = synth_trace(120, 2, 24, 6, batch=3, stickiness=0.8, zipf_s=1.5, seed=4)
    predictor = ExpertPredictor(2, 24)
    for layers in trace.steps:
        for layer, step in enumerate(layers):
            predictor.observe_step(layer, step)
            for stream in range(3):
                p = np.array([predictor.probability(layer, e, stream) for e in range(24)])
                assert ((p >= 0.0) & (p <= 1.0)).all()
            p_any = np.array([predictor.probability_any(layer, e) for e in range(24)])
            assert ((p_any >= 0.0) & (p_any <= 1.0)).all()


def test_recency_beats_frequency_on_held_out_sticky_trace():
    full = _held_out_hit_rate(0.7, {})
    freq = _held_out_hit_rate(0.7, FREQ_ONLY)
    assert full > freq + 0.15
    assert full > 0.75


def test_not_materially_worse_without_stickiness():
    for seed in (3, 11):
        full = _held_out_hit_rate(0.0, {}, seed=seed)
        freq = _held_out_hit_rate(0.0, FREQ_ONLY, seed=seed)
        assert full > freq - 0.02


def test_frequency_beats_uninformed_prior():
    freq = _held_out_hit_rate(0.0, FREQ_ONLY)
    uniform = _held_out_hit_rate(0.0, {"use_frequency": False, **FREQ_ONLY})
    assert freq > uniform + 0.2


def test_transitions_learn_deterministic_cycle():
    trace = _cycle_trace()
    with_transitions = ExpertPredictor(1, 6).evaluate(trace, 1)
    without = ExpertPredictor(1, 6, use_transitions=False).evaluate(trace, 1)
    assert with_transitions["hit_rate"] > 0.9
    assert with_transitions["coverage_full"] == with_transitions["hit_rate"]
    assert without["hit_rate"] < 0.3

    predictor = ExpertPredictor(1, 6)
    predictor.fit(trace)  # last observed expert is (300 - 1) % 6 == 5, so 0 comes next
    assert predictor.predict(0, 1, stream=0)[0][0] == 0
    assert predictor.probability(0, 0) > 0.7  # below 1: smoothing shrinks the decayed counts
    assert predictor.probability(0, 5) < 0.2


def test_transitions_alone_express_repeats():
    full = _held_out_hit_rate(0.7, {"use_recency": False})
    freq = _held_out_hit_rate(0.7, FREQ_ONLY)
    assert full > freq + 0.1


def test_predict_ordering_and_tie_breaking():
    predictor = ExpertPredictor(1, 8)
    assert predictor.predict(0, 3) == [(0, 0.0), (1, 0.0), (2, 0.0)]
    assert predictor.predict(0, 0) == []
    flat = ExpertPredictor(1, 8, decay=1.0, **FREQ_ONLY)
    flat.observe(0, [6, 2])
    flat.observe(0, [6, 4])
    ranked = flat.predict(0, 8, stream=0)
    assert [e for e, _ in ranked] == [6, 2, 4, 0, 1, 3, 5, 7]
    probs = [p for _, p in ranked]
    assert probs == sorted(probs, reverse=True)
    assert probs[0] == pytest.approx(flat.frequency(0, 6))
    assert all(isinstance(e, int) and isinstance(p, float) for e, p in ranked)
    assert len(flat.predict(0, 100)) == 8


def test_unknown_stream_falls_back_to_frequency():
    trace = synth_trace(60, 1, 16, 3, stickiness=0.8, seed=6)
    predictor = ExpertPredictor(1, 16)
    predictor.fit(trace)
    last = trace.at(59, 0).experts[0][0]
    assert predictor.probability(0, last, stream=0) > predictor.frequency(0, last)
    for e in range(16):
        assert predictor.probability(0, e, stream=99) == pytest.approx(predictor.frequency(0, e))


def test_probability_any_combines_streams():
    trace = synth_trace(80, 2, 16, 3, batch=3, stickiness=0.7, seed=8)
    predictor = ExpertPredictor(2, 16)
    assert predictor.probability_any(0, 0) == 0.0
    predictor.fit(trace)
    for e in range(16):
        per_stream = [predictor.probability(1, e, s) for s in range(3)]
        expected = 1.0 - np.prod([1.0 - p for p in per_stream])
        assert predictor.probability_any(1, e) == pytest.approx(expected)
        assert predictor.probability_any(1, e) >= max(per_stream) - 1e-12
    aggregated = predictor.predict(1, 16)
    assert aggregated[0][1] == pytest.approx(max(predictor.probability_any(1, e)
                                                 for e in range(16)))


def test_observe_step_maps_rows_to_streams():
    predictor = ExpertPredictor(1, 8, decay=1.0)
    predictor.observe_step(0, RoutingStep(((0, 1), (2, 3)), ((0.5, 0.5), (0.5, 0.5))))
    predictor.observe_step(0, RoutingStep(((0, 1), (2, 3)), ((0.5, 0.5), (0.5, 0.5))))
    assert predictor.frequency(0, 0) == pytest.approx(0.5)  # 2 of 4 stream visits
    assert predictor.probability(0, 0, stream=0) > predictor.probability(0, 0, stream=1)
    assert predictor.probability(0, 3, stream=1) > predictor.probability(0, 3, stream=0)


def test_reset_streams_keeps_statistics_and_drops_recency():
    trace = synth_trace(100, 1, 16, 3, batch=2, stickiness=0.9, seed=9)
    predictor = ExpertPredictor(1, 16)
    predictor.fit(trace)
    freq_before = [predictor.frequency(0, e) for e in range(16)]
    coeffs_before = predictor._coeffs(0)
    assert coeffs_before[0] > 0.5  # learned stickiness
    predictor.reset_streams()
    assert [predictor.frequency(0, e) for e in range(16)] == freq_before
    assert predictor._coeffs(0) == coeffs_before
    for e in range(16):
        f = predictor.frequency(0, e)
        assert predictor.probability(0, e, stream=0) == pytest.approx(f)
        assert predictor.probability_any(0, e) == pytest.approx(1.0 - (1.0 - f) ** 2)
    # Recency resumes as soon as the stream is observed again.
    predictor.observe(0, [1, 2, 3], stream=0)
    assert predictor.probability(0, 1, stream=0) > predictor.frequency(0, 1)


def test_uniform_prior_when_frequency_disabled():
    predictor = ExpertPredictor(1, 8, use_frequency=False, **FREQ_ONLY)
    predictor.observe(0, [0, 1])
    probs = [predictor.probability(0, e) for e in range(8)]
    assert probs == pytest.approx([0.25] * 8)
    assert predictor.frequency(0, 0) == 1.0  # the raw statistic is still tracked


def test_evaluate_is_online_and_reports_all_metrics():
    trace = synth_trace(40, 2, 16, 4, batch=2, stickiness=0.5, seed=10)
    predictor = ExpertPredictor(2, 16)
    result = predictor.evaluate(trace, 4)
    assert set(result) == {"hit_rate", "precision", "coverage_full"}
    assert all(0.0 <= v <= 1.0 for v in result.values())
    assert result["precision"] == pytest.approx(result["hit_rate"])  # n == top_k
    assert predictor.frequency(0, trace.at(0, 0).experts[0][0]) > 0.0  # it learned while scoring
    assert ExpertPredictor(2, 16).evaluate(trace.split(0)[0], 4)["hit_rate"] == 0.0


def test_argument_validation():
    with pytest.raises(ValueError):
        ExpertPredictor(0, 8)
    with pytest.raises(ValueError):
        ExpertPredictor(1, 8, decay=0.0)
    with pytest.raises(ValueError):
        ExpertPredictor(1, 8, smoothing=0.0)
    predictor = ExpertPredictor(1, 8)
    with pytest.raises(ValueError):
        predictor.observe(0, [8])
    with pytest.raises(ValueError):
        predictor.observe(0, [])
    with pytest.raises(ValueError):
        predictor.fit(synth_trace(2, 2, 8, 2))
    with pytest.raises(ValueError):
        predictor.evaluate(synth_trace(2, 1, 16, 2), 2)
