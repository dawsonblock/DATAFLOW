"""v0.5 speculative execution in the physical runtime: exact greedy output, preserved sampling
distribution, KV rollback, the adaptive controller, prompt lookup, and grouped expert reads."""

from __future__ import annotations

import numpy as np
import pytest

from basp.models.gguf_writer import tiny_qwen3
from basp.runtime import Engine, generate
from basp.runtime.speculative import (
    AdaptiveController,
    ModelDrafter,
    NgramDrafter,
    speculative_generate,
)

PROMPT = [5, 17, 3, 44, 9, 60, 2, 31]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("spec")
    paths = {}
    for name, moe, seed, vocab in (("target", False, 1, 96), ("draft", False, 2, 96), ("moe", True, 3, 96),
                                   ("small-target", False, 4, 16), ("small-draft", False, 5, 16)):  # fmt: skip
        paths[name] = root / f"{name}.gguf"
        tiny_qwen3(paths[name], moe=moe, seed=seed, vocab=vocab, scale=0.3 if vocab == 16 else 0.08)
    return paths


def sequential(engine, prompt, n):
    with engine.session(len(prompt) + n + 1) as s:
        return generate(s, prompt, n).tokens


@pytest.mark.parametrize("k", [1, 3, 6])
def test_greedy_speculative_output_equals_the_targets_greedy_output(models, k):
    target, draft = Engine(models["target"]), Engine(models["draft"])
    expected = sequential(target, PROMPT, 24)
    drafter = ModelDrafter(draft, 64)
    with target.session(64) as s:
        r = speculative_generate(s, drafter, PROMPT, 24, k=k)
    assert r.tokens == expected
    assert sum(x.returned for x in r.rounds) == len(r.tokens)
    assert all(x.proposed <= k and x.accepted <= x.proposed for x in r.rounds)
    # a drafter identical to the target is always right: k + 1 tokens per target call
    same = ModelDrafter(Engine(models["target"]), 64)
    with target.session(64) as s:
        r = speculative_generate(s, same, PROMPT, 24, k=k)
    assert r.tokens == expected and r.acceptance == 1.0
    assert r.rounds[0].returned == k + 1


def test_prompt_lookup_drafts_from_the_context_and_stays_exact(models):
    target = Engine(models["target"])
    expected = sequential(target, PROMPT * 3, 20)
    with target.session(64) as s:
        r = speculative_generate(s, NgramDrafter(), PROMPT * 3, 20, k=4)
    assert r.tokens == expected
    lookup = NgramDrafter(max_n=2)
    assert lookup.propose([1, 2, 3, 9, 1, 2], 3).tokens == [3, 9, 1]  # continues the earlier "1 2"
    assert lookup.propose([1, 2, 3], 3).tokens == []  # nothing repeats
    assert lookup.propose([1, 2, 3, 1, 2], 0).tokens == []


def test_sampling_follows_the_targets_distribution(models):
    """Chi-square test on the first two sampled tokens of a 16-token vocabulary: speculative
    sampling with a different drafter must reproduce the target's own distribution."""
    target, draft = Engine(models["small-target"]), Engine(models["small-draft"])
    prompt, temperature, n = [1, 2, 3], 1.0, 2000
    with target.session(8) as s:
        p1 = np.exp(s.prefill(prompt).logits[-1].astype(np.float64))
    p1 /= p1.sum()
    p2 = np.zeros(16)  # exact marginal of the second token: sum over the first
    for t1 in range(16):
        with target.session(8) as s:
            row = s.prefill(prompt + [t1]).logits[-1].astype(np.float64)
        q = np.exp(row - row.max())
        p2 += p1[t1] * q / q.sum()
    drafter = ModelDrafter(draft, 16)
    counts = np.zeros((2, 16))
    for seed in range(n):
        with target.session(16) as s:
            r = speculative_generate(s, drafter, prompt, 2, k=2, temperature=temperature, seed=seed)
        counts[0, r.tokens[0]] += 1
        counts[1, r.tokens[1]] += 1
    for observed, expected in ((counts[0], p1), (counts[1], p2)):
        mask = expected * n >= 5
        chi2 = float(((observed[mask] - n * expected[mask]) ** 2 / (n * expected[mask])).sum())
        assert chi2 < 37.7, chi2  # 15 degrees of freedom at most: p > 0.001
    # the drafter is not the target: some proposals must have been rejected
    with target.session(16) as s:
        r = speculative_generate(s, drafter, prompt, 8, k=3, temperature=temperature, seed=7)
    assert r.acceptance < 1.0


def test_rejected_proposals_leave_the_cache(models):
    target, draft = Engine(models["target"]), Engine(models["draft"])
    with target.session(64) as s:
        r = speculative_generate(s, ModelDrafter(draft, 64), PROMPT, 12, k=5)
        assert s.position == len(PROMPT) + len(r.tokens) - 1  # every committed token but the last
        after = s.step(r.tokens[-1]).logits[-1]
    with target.session(64) as s:
        fresh = s.prefill(PROMPT + r.tokens).logits[-1]
    np.testing.assert_allclose(after, fresh, rtol=0, atol=1e-5 * np.abs(fresh).max())
    with target.session(4) as s:
        s.prefill([1, 2, 3])
        with pytest.raises(ValueError, match="cannot truncate"):
            s.truncate(5)


def test_the_controller_drafts_more_when_proposals_are_accepted_and_none_when_rejected():
    def trained(accepted, slope: float) -> AdaptiveController:
        c = AdaptiveController(k_max=8)
        for i in range(30):
            k = 1 + i % 6
            c.update(proposed=k, accepted=accepted(k), draft_s=0.001 * k, rows=k + 1,
                     verify_s=0.010 + slope * (k + 1))  # fmt: skip
        return c

    good = trained(lambda k: k, 0.0005)  # always accepted, extra rows nearly free: draft far ahead
    assert good.alpha > 0.95 and good.choose() == 8 and good.choose(limit=3) == 3
    bad = trained(lambda k: 0, 0.0005)  # nothing is ever accepted: decode normally
    assert bad.alpha < 0.1 and bad.choose() == 0
    half = trained(lambda k: k // 2, 0.0005)  # about half accepted: draft a little
    assert 0.4 < half.alpha < 0.8 and 1 <= half.choose() <= 4
    costly = trained(lambda k: k // 2, 0.02)  # each verified row costs twice the fixed part: not worth it
    assert costly.choose() == 0
    assert AdaptiveController(k_max=5, k_start=3).choose() == 3  # before measurements: the starting guess
    with pytest.raises(ValueError):
        AdaptiveController(prior_alpha=1.0)


def test_adaptive_speculation_stays_exact(models):
    target, draft = Engine(models["target"]), Engine(models["draft"])
    expected = sequential(target, PROMPT, 30)
    controller = AdaptiveController(k_max=6)
    with target.session(64) as s:
        r = speculative_generate(s, ModelDrafter(draft, 64), PROMPT, 30, controller=controller)
    assert r.tokens == expected and controller.rounds == len(r.rounds)
    summary = r.summary()
    assert summary["returned_tokens"] == 30 and summary["target_calls"] == len(r.rounds)


def test_verifying_tokens_together_reads_each_expert_once(models):
    """Grouped execution on a mixture-of-experts target: with k proposals verified in one pass,
    an expert chosen by several of them is requested once, so expert reads per token fall."""
    target = Engine(models["moe"])
    seq_before = target.telemetry.expert_requests
    expected = sequential(target, PROMPT, 20)
    seq_requests = target.telemetry.expert_requests - seq_before
    with target.session(64) as s:
        r = speculative_generate(s, ModelDrafter(Engine(models["moe"]), 64), PROMPT, 20, k=4)
    assert r.tokens == expected and r.acceptance == 1.0
    spec_requests = sum(x.expert_requests for x in r.rounds)
    assert spec_requests < seq_requests  # same tokens, fewer expert reads
    assert r.summary()["expert_logical_bytes_per_token"] > 0


def test_a_shared_prefix_is_computed_once(models):
    from basp.memory import Category
    from basp.runtime.prefix_cache import PrefixCache

    engine = Engine(models["target"])
    cache = PrefixCache(engine, block=8, capacity_bytes=1 << 20)
    system = list(range(1, 33))  # 32 shared tokens
    questions = [[40, 41, 42], [50, 51], [60, 61, 62, 63]]
    for i, q in enumerate(questions):
        tokens = system + q
        s, reused = cache.session(tokens, 64)
        assert reused == (0 if i == 0 else 32)
        logits = s.prefill(tokens[reused:]).logits[-1]
        cache.store(s, tokens)
        with engine.session(64) as fresh:
            expected = fresh.prefill(tokens).logits[-1]
        np.testing.assert_allclose(logits, expected, rtol=0, atol=2e-5 * np.abs(expected).max())
        s.close()
    assert cache.stats.hits == 2 and cache.stats.reused_tokens == 64
    assert engine.domain.used_by(Category.KV) > 0  # the cached prefix is charged to the domain
    cache.clear()
    assert engine.domain.used_by(Category.KV) == 0
    tiny = PrefixCache(engine, block=8, capacity_bytes=1)  # too small for any prefix: never stores
    s, _ = tiny.session(system, 64)
    s.prefill(system)
    assert tiny.store(s, system) == 0
    with pytest.raises(ValueError):
        PrefixCache(engine, block=0)
