"""v0.6 measurement protocol: a protocol fixed before results, paired counterbalanced trials,
prompts as the independent units, explicit decisions -- and the harness validated end to end on
small random-weight models (no tokenizer, no download)."""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from basp.bench import Protocol, ProtocolError, analyze, decide, headroom, required_fraction
from basp.bench.arms import BaspArms, NativeArm
from basp.evidence import CONFIG_DIR, validity
from basp.models.gguf_writer import tiny_qwen3
from benchmarks import run_study

PROTOCOL = CONFIG_DIR / "benchmark_protocol.json"


# -- the protocol ---------------------------------------------------------------------------------
def test_the_proposed_protocol_is_well_formed_and_not_frozen():
    p = Protocol.load(PROTOCOL)
    held, dev = p.prompts("decision"), p.prompts("feasibility")
    assert Counter((x["category"], x["context"]) for x in held) == {
        (c, k): 4 for c in ("coding", "technical", "prose") for k in ("short", "long")}  # fmt: skip
    assert len(dev) == 6 and not {x["text"] for x in dev} & {x["text"] for x in held}
    assert (p.study("decision")["repetitions"], p.study("decision")["max_new_tokens"]) == (3, 128)
    assert p.comparison("internal")["min_speedup"] == 1.10 and p.comparison("competitive")["min_speedup"] == 1.20
    assert p.comparison("competitive")["baseline"] == "D" and p.comparison("competitive")["candidate"] == "E"
    assert p.data["regime"]["concurrency"] == 1 and not p.frozen("internal") and not p.frozen("competitive")
    with pytest.raises(ProtocolError, match="is not frozen"):
        p.check_runnable("decision", "internal")  # nothing decides before the protocol is frozen
    with pytest.raises(ProtocolError, match="not pinned: set endpoint, revision, binary_sha256, command"):
        p.check_runnable("feasibility", "competitive")  # a native arm must name its exact build
    p.check_runnable("feasibility", "internal")


def test_trials_are_paired_and_counterbalanced():
    p = Protocol.load(PROTOCOL)
    trials = p.schedule("decision", "internal")
    assert len(trials) == 24 * 3 * 2
    pairs: dict[int, list] = {}
    for i, t in enumerate(trials):
        pairs.setdefault(t.pair, []).append((i, t))
    for members in pairs.values():  # the two arms of a pair run back to back, on the same prompt
        (i, a), (j, b) = members
        assert j == i + 1 and a.prompt_id == b.prompt_id and {a.arm, b.arm} == {"A", "B"} and (a.position, b.position) == (0, 1)
    first = Counter(t.arm for t in trials if t.position == 0)
    assert first["A"] == first["B"] == 36  # no arm always goes first
    per_prompt = Counter((t.prompt_id, t.arm) for t in trials if t.position == 0)
    assert all(abs(per_prompt[(pid, "A")] - per_prompt[(pid, "B")]) == 1 for pid in {t.prompt_id for t in trials})
    assert trials == p.schedule("decision", "internal")  # deterministic
    order = [t.prompt_id for t in trials if t.repetition == 0 and t.position == 0]
    assert order != sorted(order) and order != [x["id"] for x in p.prompts("decision")]  # not file order


def test_freezing_binds_a_comparison_to_its_exact_content(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(PROTOCOL.read_text())
    p = Protocol.load(path)
    with pytest.raises(ProtocolError, match="pin setup"):
        p.freeze("internal")  # this runtime's arms: the model files and backend must be pinned first
    model = tmp_path / "target.gguf"
    model.write_bytes(b"GGUF-not-really-but-hashable")
    from basp.evidence import file_identity

    p.data["setup"].update(target="target.gguf", target_sha256=file_identity(model)["sha256"], backend="torch",
                           machine="this machine")  # fmt: skip
    fingerprint = p.freeze("internal")
    p.save()
    p = Protocol.load(path)
    assert p.frozen("internal") and p.fingerprint("internal") == fingerprint
    p.check_runnable("decision", "internal")
    with pytest.raises(ProtocolError, match="not pinned"):
        p.freeze("competitive")  # unpinned arms cannot be frozen in
    # pinning a native arm later does not unfreeze a comparison that does not use it ...
    p.arms["D"].update(endpoint="http://127.0.0.1:8080", revision="abc123", binary_sha256="0" * 64, command="server -m x")
    assert p.frozen("internal")
    # ... but touching anything the comparison depends on does
    for change in (lambda d: d["arms"]["B"].update(k=3), lambda d: d["prompts"]["heldout"].pop(),
                   lambda d: d["criteria"].update(ci_level=0.9), lambda d: d["studies"]["decision"].update(repetitions=5),
                   lambda d: d["comparisons"][0].update(min_speedup=1.05), lambda d: d["regime"].update(concurrency=4)):  # fmt: skip
        q = Protocol(json.loads(path.read_text()))
        assert q.frozen("internal")
        change(q.data)
        assert not q.frozen("internal")
        with pytest.raises(ProtocolError, match="changed after it was frozen"):
            q.check_runnable("decision", "internal")
    q = Protocol(json.loads(path.read_text()))
    q.data["_note"] = "a comment does not change the study"
    assert q.frozen("internal")
    q.check_setup(target=model, draft=None, backend="torch", expert_cache_gib=None, embedding="resident")
    other = tmp_path / "other.gguf"
    other.write_bytes(b"different weights")
    with pytest.raises(ProtocolError, match="target model is not the pinned file"):
        q.check_setup(target=other, draft=None, backend="torch", expert_cache_gib=None, embedding="resident")
    with pytest.raises(ProtocolError, match="backend is 'numpy'"):
        q.check_setup(target=model, draft=None, backend="numpy", expert_cache_gib=None, embedding="resident")
    bad = json.loads(path.read_text())
    bad["prompts"]["heldout"][0]["text"] = bad["prompts"]["development"][0]["text"]
    with pytest.raises(ProtocolError, match="disjoint"):
        Protocol(bad)


# -- the analysis ---------------------------------------------------------------------------------
PROMPTS = [{"id": f"{c}{i}", "category": c} for c in ("coding", "technical", "prose") for i in range(4)]


def observations(speedup_of, reps=3, base=10.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out, pair = [], 0
    for rep in range(reps):
        for p in PROMPTS:
            jitter = math.exp(noise * rng.standard_normal())
            out.append({"pair": pair, "prompt_id": p["id"], "arm": "A", "ok": True, "latency_s": base * jitter,
                        "tokens_returned": 64, "token_ids": [1, 2], "position": pair % 2})  # fmt: skip
            out.append({"pair": pair, "prompt_id": p["id"], "arm": "B", "ok": True,
                        "latency_s": base * jitter / speedup_of(p), "tokens_returned": 64, "token_ids": [1, 2],
                        "position": 1 - pair % 2})  # fmt: skip
            pair += 1
    return out


def run_analysis(obs, **kw):
    return analyze(obs, baseline="A", candidate="B", prompts=PROMPTS, expected_tokens=64, resamples=2000, **kw)


def test_speedup_is_the_equal_weight_geometric_mean_over_prompts():
    per_category = {"coding": 1.5, "technical": 1.2, "prose": 0.9}
    a = run_analysis(observations(lambda p: per_category[p["category"]]))
    assert a["speedup"] == pytest.approx((1.5 * 1.2 * 0.9) ** (1 / 3))
    assert a["time_saved_fraction"] == pytest.approx(1 - 1 / a["speedup"])
    assert a["ci_low"] <= a["speedup"] <= a["ci_high"]
    assert {c: round(v["speedup"], 6) for c, v in a["categories"].items()} == per_category
    assert a["categories"]["prose"]["median_latency_change"] == pytest.approx(1 / 0.9 - 1)
    assert a["order_balance"] == {"baseline_first": 18, "candidate_first": 18} and a["failures"] == []
    assert analyze([], baseline="A", candidate="B", prompts=PROMPTS)["speedup"] is None


def test_repetitions_are_not_independent_prompts():
    spread = lambda p: 1.0 + 0.1 * int(p["id"][-1])
    few, many = run_analysis(observations(spread, reps=1)), run_analysis(observations(spread, reps=30))
    assert many["valid_pairs"] == 30 * few["valid_pairs"]
    # thirty times the requests, the same twelve prompts: the interval does not shrink
    assert many["ci_high"] - many["ci_low"] == pytest.approx(few["ci_high"] - few["ci_low"], rel=0.05)
    assert many["speedup"] == pytest.approx(few["speedup"])


def test_failures_are_listed_and_make_the_result_inconclusive_not_better():
    obs = observations(lambda p: 1.3)
    slow = [o for o in obs if o["prompt_id"] == "prose0" and o["arm"] == "B"]
    for o in slow:
        o.update(ok=False, failure="MemoryError: out of memory")  # the candidate fails on one prompt
    next(o for o in obs if o["prompt_id"] == "coding1" and o["arm"] == "A")["tokens_returned"] = 31  # unequal work
    a = run_analysis(obs)
    assert a["prompts_missing"] == ["prose0"] and a["prompts_with_valid_pairs"] == 11
    problems = [p for f in a["failures"] for p in f["problems"]]
    assert sum("MemoryError" in p for p in problems) == 3 and any("returned 31 tokens, 64 were asked for" in p for p in problems)
    d = decide(a, min_speedup=1.10)
    assert d["verdict"] == "inconclusive" and not d["supports_decision"] and "coverage is incomplete" in " ".join(d["reasons"])
    # an output that differs is a failure where outputs must agree
    wrong = observations(lambda p: 1.3)
    next(o for o in wrong if o["arm"] == "B")["token_ids"] = [1, 9]
    assert "differs from the baseline's from token 1" in run_analysis(wrong, require_same_output=True)["failures"][0]["problems"][0]
    assert run_analysis(wrong)["failures"] == []


@pytest.mark.parametrize(("speedups", "noise", "verdict"), [
    ({"coding": 1.4, "technical": 1.3, "prose": 1.25}, 0.0, "meets_criterion"),
    ({"coding": 1.06, "technical": 1.05, "prose": 1.04}, 0.0, "below_criterion"),
    ({"coding": 0.8, "technical": 0.85, "prose": 0.9}, 0.0, "slower"),
    ({"coding": 2.6, "technical": 1.5, "prose": 0.9}, 0.0, "category_regression"),
])  # fmt: skip
def test_the_decision_rule(speedups, noise, verdict):
    a = run_analysis(observations(lambda p: speedups[p["category"]], noise=noise))
    d = decide(a, min_speedup=1.10)
    assert d["verdict"] == verdict, d
    assert d["supports_decision"] == (verdict == "meets_criterion")
    assert not decide(a, min_speedup=1.10, exploratory=True)["supports_decision"]  # an exploratory study never decides


def test_an_interval_that_includes_no_improvement_is_inconclusive():
    mixed = lambda p: 1.6 if int(p["id"][-1]) % 2 else 0.7
    a = run_analysis(observations(mixed))
    assert a["speedup"] > 1.0 and a["ci_low"] < 1.0 < a["ci_high"]
    assert decide(a, min_speedup=1.02)["verdict"] == "inconclusive"


def test_headroom_of_a_profiled_component():
    assert headroom(0.30, 0.50) == pytest.approx(1 / 0.85)  # 1.176x: misses a 1.20x target
    assert headroom(0.30, 0.50) < 1.20 < headroom(0.40, 0.50)
    assert required_fraction(1.20, 0.50) == pytest.approx(1 / 3)
    assert required_fraction(1.20, 0.10) > 1.0  # impossible: no component is larger than the whole
    assert headroom(0.0, 1.0) == 1.0 and headroom(1.0, 1.0) == math.inf
    assert 1 - 1 / 1.20 == pytest.approx(0.1667, abs=1e-4)  # 1.20x is 16.7% less time, not 20%
    with pytest.raises(ValueError):
        headroom(1.2, 0.5)
    assert run_study.main(["headroom", "--fraction", "0.30", "--reduction", "0.50"]) == 1
    assert run_study.main(["headroom", "--fraction", "0.50", "--reduction", "0.50"]) == 0


# -- the harness, end to end on small models -------------------------------------------------------
@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    tiny_qwen3(root / "target.gguf", moe=True, seed=3)
    tiny_qwen3(root / "draft.gguf", seed=5)
    return root


def test_a_feasibility_study_runs_resumes_and_is_analysed(tiny, tmp_path, capsys):
    out = tmp_path / "study"
    args = ["run", "--study", "feasibility", "--comparison", "internal", "--target", str(tiny / "target.gguf"),
            "--draft", str(tiny / "draft.gguf"), "--tokenizer", "bytes", "--tokens", "12", "--out", str(out)]  # fmt: skip
    assert run_study.main(args) == 0
    rows = [json.loads(line) for line in (out / "observations.jsonl").read_text().splitlines()]
    assert len(rows) == 6 * 2 * 2 and all(r["ok"] and r["tokens_returned"] == 12 for r in rows)
    for r in rows:  # logical demand, cache behaviour and physical traffic are separate numbers
        assert r["work"]["expert_logical_requests"] >= r["work"]["expert_cache_misses"] > 0
        assert r["work"]["expert_logical_stored_bytes"] > 0 and "storage_read_bytes" in r["io"]
        assert r["memory"]["peak_expert_resident_bytes"] > 0
    spec = [r for r in rows if r["arm"] == "B"]
    assert all(r["work"]["draft_rows"] > 0 and "acceptance" in r["speculation"] for r in spec)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["analysis"]["prompts_with_valid_pairs"] == 6 and summary["analysis"]["failures"] == []
    assert summary["decision"]["exploratory"] and not summary["decision"]["supports_decision"]
    assert summary["reduced"] and summary["harness_validation_only"]
    assert set(summary["projection"]) == {"feasibility", "decision"} and summary["projection"]["decision"]["seconds"] > 0
    assert validity.check(summary, tool_files=run_study._tool_files(),
                          models={"target": tiny / "target.gguf", "draft": tiny / "draft.gguf"})["valid"]  # fmt: skip
    # a rerun resumes: nothing is measured twice
    assert run_study.main(args) == 0
    assert len((out / "observations.jsonl").read_text().splitlines()) == 24
    assert run_study.main(["analyze", str(out)]) == 0 and "verdict:" in capsys.readouterr().out
    # changed weights invalidate the stored result
    tiny_qwen3(tmp_path / "other.gguf", moe=True, seed=4)
    stale = validity.check(summary, models={"target": tmp_path / "other.gguf"})
    assert not stale["valid"] and "model file 'target' changed" in stale["reasons"][0]
    assert "NOT APPLICABLE" in validity.note(stale) and validity.note({"valid": True, "reasons": []}) == ""


def test_the_runner_refuses_what_the_protocol_forbids(tiny, tmp_path):
    base = ["--target", str(tiny / "target.gguf"), "--draft", str(tiny / "draft.gguf"), "--tokenizer", "bytes",
            "--out", str(tmp_path / "x")]  # fmt: skip
    with pytest.raises(ProtocolError, match="is not frozen"):
        run_study.main(["run", "--study", "decision", *base])
    with pytest.raises(SystemExit, match="not among the protocol's tuning candidates"):
        run_study.main(["run", "--arm", json.dumps({"method": "speculative", "drafter": "model", "k": 64}), *base])
    with pytest.raises(SystemExit, match="never for a decision study"):
        run_study.main(["run", "--study", "decision", "--arm", "{}", *base])
    with pytest.raises(ProtocolError, match="not pinned"):
        run_study.main(["run", "--comparison", "competitive", *base])


def test_a_failed_request_is_recorded_and_the_runtime_recovers(tiny):
    arms = BaspArms(str(tiny / "target.gguf"), str(tiny / "draft.gguf"))
    ordinary = {"method": "ordinary"}
    prompt = list(range(1, 20))
    good = arms.request(ordinary, prompt, 8)
    assert good["ok"] and good["tokens_returned"] == 8
    domain = arms.target.domain
    hogs = []
    while domain._alloc.largest_free_block > 4096:  # other work takes all the memory (every free extent)
        hogs.append(domain.allocate(domain._alloc.largest_free_block - 1024, domain.blocks()[0].category, "x", refs=1))
    failed = arms.request(ordinary, prompt, 8)
    assert not failed["ok"] and "MemoryError" in failed["failure"] and failed["recovered"] and failed["latency_s"] >= 0
    for hog in hogs:
        domain.release(hog, free_when_unused=True)
    again = arms.request({"method": "speculative", "drafter": "model", "k": 3}, prompt, 8)
    assert again["ok"] and again["token_ids"] == good["token_ids"]  # same greedy tokens as ordinary decoding
    with pytest.raises(ValueError, match="unknown method"):
        arms.request({"method": "telepathy"}, prompt, 2)


# -- a native server arm, against a stand-in server -------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append((self.path, body))
        n = body["max_tokens"] if body["prompt"] != "short answer" else 3
        payload = json.dumps({"choices": [{"text": " text" * n}], "usage": {"completion_tokens": n},
                              "timings": {"predicted_per_second": 42.0}}).encode()  # fmt: skip
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


def test_native_arm_measures_requests_from_outside():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        arm = NativeArm({"endpoint": f"http://127.0.0.1:{server.server_port}", "request": {"ignore_eos": True},
                         "model_name": "m"}, timeout_s=5)  # fmt: skip
        obs = arm.request("a prompt", 16)
        assert obs["ok"] and obs["tokens_returned"] == 16 and obs["latency_s"] > 0 and obs["token_ids"] is None
        path, body = server.seen[0]
        assert path == "/v1/completions" and body == {"prompt": "a prompt", "max_tokens": 16, "temperature": 0,
                                                       "stream": False, "ignore_eos": True, "model": "m"}  # fmt: skip
        short = arm.request("short answer", 16)  # the engine stopped early: unequal work, flagged by the analysis
        rows = [{**obs, "pair": 0, "prompt_id": "p", "arm": "D"}, {**short, "pair": 0, "prompt_id": "p", "arm": "E"}]
        a = analyze(rows, baseline="D", candidate="E", prompts=[{"id": "p", "category": "c"}], expected_tokens=16)
        assert "returned 3 tokens, 16 were asked for" in a["failures"][0]["problems"][0]
    finally:
        server.shutdown()
    down = NativeArm({"endpoint": "http://127.0.0.1:9"}, timeout_s=1).request("x", 4)
    assert not down["ok"] and down["failure"] and down["latency_s"] >= 0
