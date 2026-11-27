"""v0.6 report pieces: request-latency studies rendered from their summaries under the one
validity rule, so a study measured by other code is shown as not applicable, never silently."""

from __future__ import annotations

import json

from basp.bench import analyze, decide
from basp.evidence import validity
from basp.experiments import report_v6
from benchmarks import run_study

PROMPTS = [{"id": f"p{i}", "category": c} for i, c in enumerate(["coding", "coding", "prose", "prose"])]


def _observations(speedup: float) -> list[dict]:
    out, pair = [], 0
    for p in PROMPTS:
        for rep in range(2):
            base = 1.0 + 0.1 * rep
            for arm, latency, position in (("A", base, rep % 2), ("B", base / speedup, 1 - rep % 2)):
                out.append({"pair": pair, "prompt_id": p["id"], "arm": arm, "ok": True, "latency_s": latency,
                            "tokens_returned": 8, "position": position})  # fmt: skip
            pair += 1
    return out


def _summary(tmp_path, name: str, *, identity: bool, speedup: float = 1.25) -> None:
    a = analyze(_observations(speedup), baseline="A", candidate="B", prompts=PROMPTS, expected_tokens=8,
                resamples=500)  # fmt: skip
    record = {
        "study": "feasibility", "comparison": "internal", "analysis": a,
        "decision": decide(a, min_speedup=1.10, exploratory=True),
        "arms": {"A": {"label": "ordinary decoding"}, "B": {"label": "speculative, k=2"}},
        "per_arm": {arm: {"returned_tokens": 64, "storage_read_bytes": 0,
                          "per_returned_token": {"expert_logical_requests": 48.0, "expert_cache_misses": 6.0,
                                                 "expert_decoded_stored_bytes": 3e6, "target_rows": 1.0,
                                                 "draft_rows": 0.0 if arm == "A" else 2.0}}
                    for arm in ("A", "B")},
    }  # fmt: skip
    if identity:
        model = tmp_path / "target.gguf"
        model.write_bytes(b"not really a model")
        record["identity"] = validity.measurement_identity(
            tool="benchmarks.run_study", tool_files=run_study._tool_files(), models={"target": model},
            parameters={"study": "feasibility"}, backend="numpy-cpu")  # fmt: skip
    out = tmp_path / "studies" / name
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(record))


def test_studies_are_rendered_with_their_verdict_and_validity(tmp_path):
    _summary(tmp_path, "current", identity=True)
    _summary(tmp_path, "unidentified", identity=False, speedup=0.8)
    studies = report_v6.study_evidence(tmp_path)
    by_path = {s["path"]: s for s in studies}
    assert by_path["studies/current"]["validity"]["valid"]
    assert not by_path["studies/unidentified"]["validity"]["valid"]
    text = report_v6._studies_section(studies)
    current = next(line for line in text.splitlines() if "`studies/current`" in line)
    other = next(line for line in text.splitlines() if "`studies/unidentified`" in line)
    assert "1.250x" in current and "NOT APPLICABLE" not in current
    assert "meets criterion (exploratory, protocol not frozen: decides nothing)" in current
    assert "slower (exploratory, protocol not frozen: decides nothing) -- NOT APPLICABLE (no identity recorded" in other
    assert "A vs B: speculative, k=2" in current and "target.gguf" in current
    # work per returned token is shown per arm, logical requests apart from misses
    assert "| `studies/current` | B | 48 | 6 | 3 |" in text
    # an exploratory study never supports a decision, whatever its speed-up
    assert by_path["studies/current"]["decision"]["supports_decision"] is False


def test_no_studies_says_so(tmp_path):
    assert report_v6.study_evidence(tmp_path) == []
    assert "No study has been run" in report_v6._studies_section([])
