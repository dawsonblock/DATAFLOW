import json

import numpy as np
import pytest

from basp.workloads.trace import RoutingStep, RoutingTrace, synth_trace


def test_deterministic_per_seed_and_differs_across_seeds():
    a = synth_trace(20, 3, 32, 4, batch=2, seed=7)
    b = synth_trace(20, 3, 32, 4, batch=2, seed=7)
    c = synth_trace(20, 3, 32, 4, batch=2, seed=8)
    assert a == b
    assert a.steps != c.steps


def test_shapes_distinctness_and_coefficients():
    t = synth_trace(15, 4, 16, 5, batch=3, seed=1)
    assert (t.n_steps, t.n_layers, t.n_experts, t.top_k, t.batch) == (15, 4, 16, 5, 3)
    t.validate()
    for layers in t.steps:
        assert len(layers) == 4
        for st in layers:
            assert len(st.experts) == len(st.coeffs) == 3
            for ids, co in zip(st.experts, st.coeffs):
                assert len(set(ids)) == 5 and all(0 <= e < 16 for e in ids)
                assert all(type(e) is int for e in ids)
                assert sum(co) == pytest.approx(1.0, abs=1e-12)
                assert all(c > 0 for c in co) and list(co) == sorted(co, reverse=True)
    assert t.at(2, 1) is t.steps[2][1]


def test_top_k_equal_to_n_experts_selects_everything():
    t = synth_trace(5, 2, 4, 4, stickiness=0.5, seed=0)
    assert all(sorted(st.experts[0]) == [0, 1, 2, 3] for layers in t.steps for st in layers)


def test_stickiness_one_repeats_forever_and_zero_does_not():
    sticky = synth_trace(30, 2, 64, 4, batch=2, stickiness=1.0, seed=3)
    assert all(layers[layer].experts == sticky.at(0, layer).experts
               for layers in sticky.steps for layer in range(2))
    fresh = synth_trace(30, 2, 64, 4, batch=2, stickiness=0.0, zipf_s=0.0, seed=3)
    repeats = [len(set(fresh.at(s, 0).experts[0]) & set(fresh.at(s + 1, 0).experts[0]))
               for s in range(29)]
    assert np.mean(repeats) < 1.0  # uniform over 64 experts: expected overlap is 4 * 4 / 64


def test_repeat_rate_grows_with_stickiness():
    def repeat_rate(stickiness):
        t = synth_trace(200, 1, 128, 8, zipf_s=0.0, stickiness=stickiness, seed=5)
        same = [len(set(t.at(s, 0).experts[0]) & set(t.at(s + 1, 0).experts[0]))
                for s in range(199)]
        return np.mean(same) / 8
    low, mid, high = repeat_rate(0.1), repeat_rate(0.5), repeat_rate(0.9)
    assert low < mid < high
    assert mid == pytest.approx(0.5, abs=0.08)


def test_zipf_skew_visible_in_frequencies():
    skewed = synth_trace(400, 2, 64, 4, zipf_s=1.5, stickiness=0.0, seed=2).expert_frequencies()
    flat = synth_trace(400, 2, 64, 4, zipf_s=0.0, stickiness=0.0, seed=2).expert_frequencies()
    assert skewed.shape == flat.shape == (2, 64)
    np.testing.assert_allclose(skewed.sum(axis=1), 4.0)
    np.testing.assert_allclose(flat.sum(axis=1), 4.0)
    assert skewed.max(axis=1).min() > 0.7
    assert flat.max() < 0.2
    assert not np.array_equal(np.argsort(skewed[0]), np.argsort(skewed[1]))  # per-layer ranking


def test_expert_frequencies_counts_all_batch_rows():
    step = RoutingStep(((0, 1), (1, 2)), ((0.5, 0.5), (0.5, 0.5)))
    t = RoutingTrace(1, 4, 2, 2, [[step], [step]])
    np.testing.assert_allclose(t.expert_frequencies(), [[0.5, 1.0, 0.5, 0.0]])
    assert not RoutingTrace(2, 4, 2, 1).expert_frequencies().any()


def test_unique_experts_and_rows_for():
    step = RoutingStep(((5, 2, 9), (2, 7, 5), (1, 0, 9)), ((0.5, 0.3, 0.2),) * 3)
    assert step.unique_experts() == (0, 1, 2, 5, 7, 9)
    assert step.rows_for(5) == (0, 1)
    assert step.rows_for(9) == (0, 2)
    assert step.rows_for(4) == ()


def test_json_round_trip(tmp_path):
    t = synth_trace(6, 3, 16, 4, batch=2, seed=11)
    assert RoutingTrace.from_json(json.loads(json.dumps(t.to_json()))) == t
    path = tmp_path / "trace.json"
    t.save(path)
    loaded = RoutingTrace.load(path)
    assert loaded == t
    assert isinstance(loaded.at(0, 0).experts[0], tuple)


def _doc(**edits):
    d = synth_trace(2, 2, 8, 3, seed=0).to_json()
    d.update(edits)
    return d


@pytest.mark.parametrize("mutate, match", [
    (lambda d: d["steps"][0][0]["experts"][0].__setitem__(0, d["steps"][0][0]["experts"][0][1]),
     "duplicate"),
    (lambda d: d["steps"][1][1]["experts"][0].__setitem__(0, 8), "outside"),
    (lambda d: d["steps"][0][0]["experts"][0].__setitem__(0, 1.5), "integers"),
    (lambda d: d["steps"][0][0]["experts"][0].pop(), "top_k"),
    (lambda d: d["steps"][0].pop(), "layers"),
    (lambda d: d["steps"][0][0]["experts"].append([0, 1, 2]), "batch rows"),
    (lambda d: d["steps"][0][0]["coeffs"][0].__setitem__(0, 0.9), "sum to 1"),
    (lambda d: d["steps"][0][0].pop("coeffs"), "malformed"),
    (lambda d: d.pop("top_k"), "missing field"),
    (lambda d: d.__setitem__("version", 99), "unsupported"),
    (lambda d: d.__setitem__("batch", "1"), "integers"),
])
def test_from_json_validation_errors(mutate, match):
    d = _doc()
    mutate(d)
    with pytest.raises(ValueError, match=match):
        RoutingTrace.from_json(d)


def test_load_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        RoutingTrace.load(path)


def test_split():
    t = synth_trace(10, 2, 8, 2, seed=4)
    head, tail = t.split(7)
    assert (head.n_steps, tail.n_steps) == (7, 3)
    assert head.steps + tail.steps == t.steps
    assert (tail.n_layers, tail.n_experts, tail.top_k, tail.batch) == (2, 8, 2, 1)
    assert t.split(0)[0].n_steps == 0 and t.split(10)[1].n_steps == 0
    with pytest.raises(ValueError):
        t.split(11)


@pytest.mark.parametrize("kwargs", [
    {"top_k": 9}, {"stickiness": 1.5}, {"zipf_s": -1.0}, {"n_steps": -1}, {"batch": 0},
])
def test_synth_trace_argument_validation(kwargs):
    args = {"n_steps": 2, "n_layers": 1, "n_experts": 8, "top_k": 2}
    args.update(kwargs)
    with pytest.raises(ValueError):
        synth_trace(**args)


def test_generation_is_fast_at_qwen3_shape():
    import time
    start = time.perf_counter()
    t = synth_trace(200, 48, 128, 8, seed=0)
    assert time.perf_counter() - start < 2.0
    assert t.n_steps == 200
