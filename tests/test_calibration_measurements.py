"""Measurement records and the calibration/validation split rules."""

from __future__ import annotations

import json

import pytest

from basp.calibration import Measurement, MeasurementSet, find_problems


def m(mid: str, split: str = "calibration", **kw) -> Measurement:
    base = {"id": mid, "machine": "i9-3080ti", "split": split, "kind": "decode",
            "model": "qwen3-30b-a3b", "tokens_per_s": 21.5}  # fmt: skip
    base.update(kw)
    return Measurement(**base)


def test_signature_identifies_the_benchmark_not_the_run():
    a, b = m("run-1", n_gpu_layers=16), m("run-2", n_gpu_layers=16, tokens_per_s=22.0)
    assert a.signature == b.signature == ("i9-3080ti", "decode", "qwen3-30b-a3b", 4.5, 16, 0, 0, 1, 0)
    assert m("x", n_gpu_layers=17).signature != a.signature
    assert m("x", kind="prefill", n_gpu_layers=16).signature != a.signature
    assert m("x", machine="m2", n_gpu_layers=16).signature != a.signature


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"split": "test"}, "split must be one of"),
        ({"kind": "training"}, "kind must be one of"),
        ({"tokens_per_s": 0.0}, "tokens_per_s must be positive"),
        ({"tokens_per_s": float("nan")}, "tokens_per_s must be positive"),
        ({"machine": " "}, "machine must be a non-empty string"),
        ({"model": ""}, "model must be a non-empty string"),
        ({"bits_per_weight": 0.0}, "bits_per_weight must be positive"),
        ({"n_gpu_layers": -2}, "n_gpu_layers must be >= -1"),
        ({"batch": 0}, "batch must be >= 1"),
        ({"context_len": -1}, "context_len and threads"),
        ({"stddev_tokens_per_s": -0.1}, "stddev_tokens_per_s"),
    ],
)
def test_invalid_measurements_are_rejected_with_clear_messages(overrides, message):
    with pytest.raises(ValueError, match=message):
        m("bad", **overrides)
    with pytest.raises(ValueError, match="id must be a non-empty string"):
        m("")


def test_ids_must_be_unique():
    ms = MeasurementSet([m("a", n_gpu_layers=0)])
    with pytest.raises(ValueError, match="duplicate measurement id 'a'"):
        ms.add(m("a", n_gpu_layers=8))
    assert len(ms) == 1


def test_a_benchmark_cannot_be_in_both_splits():
    ms = MeasurementSet([m("cal-ngl16", "calibration", n_gpu_layers=16)])
    with pytest.raises(ValueError, match="'val-ngl16'.*'cal-ngl16'.*both splits"):
        ms.add(m("val-ngl16", "validation", n_gpu_layers=16))
    # a repeat of the same benchmark in the same split is fine, as is a different benchmark
    ms.add(m("cal-ngl16-again", "calibration", n_gpu_layers=16, tokens_per_s=21.9))
    ms.add(m("val-ngl24", "validation", n_gpu_layers=24))
    assert [x.id for x in ms.calibration] == ["cal-ngl16", "cal-ngl16-again"]
    assert [x.id for x in ms.validation] == ["val-ngl24"]
    assert ms.check() == [] and ms.machines() == ("i9-3080ti",) and "val-ngl24" in ms


def test_find_problems_reports_every_violation_without_raising():
    items = [m("a", "calibration", n_gpu_layers=4), m("a", "calibration", n_gpu_layers=5),
             m("b", "validation", n_gpu_layers=4), m("c", "validation", n_gpu_layers=9)]  # fmt: skip
    problems = find_problems(items)
    assert len(problems) == 2
    assert any("'a' appears 2 times" in p for p in problems)
    assert any("both splits" in p and "['b']" in p for p in problems)
    assert find_problems(items[2:]) == []


def test_json_round_trip_is_stable(tmp_path):
    ms = MeasurementSet([
        m("z-last", "validation", n_gpu_layers=24, notes="n_gen=128", threads=8),
        m("a-first", "calibration", n_gpu_layers=0, stddev_tokens_per_s=0.3, source="llama-bench"),
        m("m2-run", "calibration", machine="m2", kind="prefill", batch=512, tokens_per_s=310.0),
    ])  # fmt: skip
    path = tmp_path / "measurements.json"
    ms.save(path)
    loaded = MeasurementSet.load(path)
    assert sorted(loaded, key=lambda x: x.id) == sorted(ms, key=lambda x: x.id)
    assert [x["id"] for x in json.loads(path.read_text())["measurements"]] == ["a-first", "m2-run",
                                                                              "z-last"]  # fmt: skip
    loaded.save(tmp_path / "again.json")
    assert (tmp_path / "again.json").read_text() == path.read_text()  # byte-stable
    assert loaded.machines() == ("i9-3080ti", "m2")


def test_loading_rejects_inconsistent_or_malformed_files(tmp_path):
    good = MeasurementSet([m("a", "calibration", n_gpu_layers=4)]).to_json()
    clash = dict(good, measurements=[*good["measurements"],
                                     m("b", "validation", n_gpu_layers=4).to_json()])  # fmt: skip
    with pytest.raises(ValueError, match="inconsistent.*both splits"):
        MeasurementSet.from_json(clash)
    dup = dict(good, measurements=good["measurements"] * 2)
    with pytest.raises(ValueError, match="'a' appears 2 times"):
        MeasurementSet.from_json(dup)
    with pytest.raises(ValueError, match="unknown fields"):
        MeasurementSet.from_json(dict(good, measurements=[dict(good["measurements"][0], gpu="x")]))
    with pytest.raises(ValueError, match="unsupported format"):
        MeasurementSet.from_json(dict(good, version=99))
    with pytest.raises(ValueError, match="measurements"):
        MeasurementSet.from_json({"nothing": 1})
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        MeasurementSet.load(broken)
    with pytest.raises(ValueError, match="split must be one of"):
        MeasurementSet.from_json(dict(good, measurements=[dict(good["measurements"][0],
                                                               split="holdout")]))  # fmt: skip
