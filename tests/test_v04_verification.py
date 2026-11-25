"""v0.4 verification: the NumPy runtime against transformers on the same GGUF weights.

Needs torch and transformers (skipped otherwise; CI runs them in the "verification" job).
Small random-weight Qwen3 and Qwen3-MoE models are written to GGUF here, so nothing is
downloaded.  The harness must also *fail* a broken runtime: one test breaks it on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("gguf")

from basp.backends.cpu import kernels
from basp.evidence import EvidenceLevel, Receipt
from basp.models.gguf_writer import tiny_qwen3
from basp.verification import compare_logits, compare_routing, perplexity
from basp.verification.parity_report import run_parity

PROMPT = [3, 14, 15, 92, 65, 35, 89, 79, 32, 38]
EVAL = list(np.random.default_rng(7).integers(0, 96, 48))


@pytest.mark.parametrize(("moe", "weights"), [(False, "F32"), (False, "Q8_0"), (True, "F32"), (True, "Q8_0")])
def test_runtime_matches_the_reference(tmp_path, moe, weights):
    path = tmp_path / "m.gguf"
    tiny_qwen3(path, moe=moe, weight_type=weights, seed=11)
    r = run_parity(path, PROMPT, model_name="tiny-random", eval_tokens=EVAL, n_generate=8)
    assert r["verdict"] == "pass", r["reasons"]
    assert r["numerical_error"] < 1e-4  # float32 on both sides: far inside the 0.002 target
    assert r["shapes_match"] and r["hidden"]["shapes_match"] and r["hidden"]["layers_compared"] == 1
    assert r["generation"]["identical"] and r["prefill_logits"]["argmax_agreement"] == 1.0
    assert abs(r["perplexity"]["relative"] - 1.0) < 1e-4
    if moe:
        assert r["routing"]["exact"] and r["routing"]["token_selections"] == 2 * len(PROMPT)
    receipts = [Receipt.from_json(x) for x in r["receipts"]]
    assert {x.metric for x in receipts} == {"numerical_error", "model_quality"}
    assert all(x.level is EvidenceLevel.E3 and not x.problems() for x in receipts)
    assert all(x.candidate == "runtime:numpy-cpu" and x.bindings["model"] == "tiny-random" for x in receipts)


def test_the_harness_fails_a_broken_runtime(tmp_path, monkeypatch):
    path = tmp_path / "m.gguf"
    tiny_qwen3(path, moe=True, seed=12)
    original = kernels.rope_neox
    monkeypatch.setattr(kernels, "rope_neox", lambda x, pos, theta: original(x, pos, theta * 2.0))
    r = run_parity(path, PROMPT, model_name="tiny-random", eval_tokens=EVAL, n_generate=8)
    assert r["verdict"] == "fail" and any("relative L2" in reason for reason in r["reasons"])


def test_comparison_helpers_reject_mismatches():
    a = np.array([[1.0, 2.0, 3.0]])
    assert compare_logits(a, a)["rel_l2_max"] == 0.0
    assert not compare_logits(np.array([[np.nan, 0, 0]]), a)["finite"]
    with pytest.raises(ValueError, match="shape"):
        compare_logits(a, np.ones((2, 3)))
    same = compare_routing([np.array([[1, 2]])], [np.array([[2, 1]])])
    assert same["exact"]  # order within the top-k does not matter
    differ = compare_routing([np.array([[1, 3]])], [np.array([[1, 2]])])
    assert not differ["exact"] and differ["mismatches"] == 1
    assert not compare_routing([], [np.array([[1, 2]])])["exact"]
    uniform = np.zeros((5, 10))
    assert perplexity(uniform, np.arange(5)) == pytest.approx(10.0)
