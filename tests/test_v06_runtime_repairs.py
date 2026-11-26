"""v0.6 runtime repairs (audit findings R1-R6), as behaviour checks.

R4/R5: owner-safe reclamation and transactional expert acquisition -- eviction by admission
       control followed by a numerically correct inference; an injected failure partway through
       an acquisition leaves no references or half-loaded experts and a retry succeeds.
R3:    ``expert_cache_bytes`` is a hard ceiling on resident expert arrays; a large expert union
       is computed in bounded groups with bit-identical results.
R1:    sampling with a deterministic proposal needs linear, not quadratic, memory and keeps the
       target's distribution.
R2:    after a stop token or the length limit the session holds exactly ``prompt + returned[:-1]``.
R6:    a prefix session is all or nothing.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from basp.memory import Category, EvictionUnsafe, MemoryDomain, RequestShape
from basp.memory.allocator import OutOfMemory
from basp.models.gguf import GGML_TYPES
from basp.models.gguf_writer import tiny_qwen3
from basp.models.quants import TRANSIENT_FACTOR, dequantize, transient_bytes
from basp.runtime import Engine, generate
from basp.runtime.prefix_cache import PrefixCache
from basp.runtime.speculative import (
    ModelDrafter,
    NgramDrafter,
    Proposal,
    accept_or_resample,
    speculative_generate,
)
from basp.runtime.weights import ExpertBudgetExceeded

TOKENS = list(range(1, 30))
ONE_EXPERT = 3 * 32 * 64 * 4  # float32 bytes of one expert of the tiny MoE model


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("repairs")
    paths = {"moe": root / "moe.gguf", "dense": root / "dense.gguf", "small": root / "small.gguf",
             "small-draft": root / "small-draft.gguf"}  # fmt: skip
    tiny_qwen3(paths["moe"], moe=True, seed=3)
    tiny_qwen3(paths["dense"], seed=1)
    tiny_qwen3(paths["small"], seed=4, vocab=16, scale=0.3)
    tiny_qwen3(paths["small-draft"], seed=5, vocab=16, scale=0.3)
    return paths


def logits_of(engine, tokens=TOKENS):
    with engine.session(len(tokens) + 1) as s:
        return s.prefill(tokens, all_logits=True).logits


def clean(engine) -> None:
    """No references, no stale or half-loaded entries, accounting exact."""
    engine.weights.check_invariant()
    engine.domain.check_invariant()
    assert all(entry.block.refs == 0 for entry in engine.weights._experts.values())


# -- R4: one reclamation path; the owner is told before capacity is freed ---------------------------
def test_blocks_reclaimed_by_anyone_invalidate_their_owner_first():
    domain = MemoryDomain("d", 1 << 20)
    owned: dict[str, object] = {}
    order: list[str] = []

    def forget(block):
        order.append(f"forget {block.owner} while live={block.live}")
        owned.pop(block.owner)

    for name in ("a", "b"):
        owned[name] = domain.allocate(1000, Category.WEIGHTS, name, on_reclaim=forget)
    busy = domain.retain(domain.allocate(1000, Category.WEIGHTS, "busy", on_reclaim=forget))
    owned["busy"] = busy
    assert [b.owner for b in domain.evict(domain.free + 1500, category=Category.WEIGHTS)] == ["a", "b"]
    assert owned == {"busy": busy} and order == ["forget a while live=True", "forget b while live=True"]
    with pytest.raises(EvictionUnsafe):
        domain.free_block(busy)  # refused: the hook must not have run
    assert "busy" in owned and busy.live
    domain.check_invariant()


def test_admission_driven_eviction_leaves_a_working_weight_cache(files):
    reference = logits_of(Engine(files["moe"]))
    engine = Engine(files["moe"], memory_bytes=3 << 20, reserved_fraction=0.0)
    np.testing.assert_array_equal(logits_of(engine), reference)
    cached = len(engine.weights._experts)
    assert cached > 0
    # a request that only fits if every cached expert is reclaimed
    need = engine.domain.free + cached * ONE_EXPERT - 4096
    admission = engine.admission.admit("request", RequestShape(0, 1, need))
    assert admission.admitted and len(admission.evicted) == cached
    assert len(engine.weights._experts) == 0  # the entries went with their blocks
    clean(engine)
    engine.admission.finish("request")
    np.testing.assert_array_equal(logits_of(engine), reference)  # reloads what was reclaimed
    clean(engine)


# -- R5: acquisition is a transaction ----------------------------------------------------------------
@pytest.mark.parametrize("failure", ["read", "dequantise", "allocate", "scratch"])
def test_a_failed_acquisition_leaves_no_trace_and_can_be_retried(files, failure, monkeypatch):
    engine = Engine(files["moe"])
    store = engine.weights
    with store.acquired(0, [0, 1]):
        pass
    before = (dict(store._experts), engine.domain.used, engine.telemetry.expert_loads)
    calls = {"n": 0}
    if failure in ("read", "dequantise"):
        original = engine.tensors.load_slice

        def load_slice(name, index):
            calls["n"] += 1
            if calls["n"] == 5:  # partway: the second of three missing experts
                raise OSError("injected read failure") if failure == "read" else ValueError("injected bad block")
            return original(name, index)

        monkeypatch.setattr(engine.tensors, "load_slice", load_slice)
    else:
        original = engine.domain.allocate

        def allocate(size, category, owner, **kw):
            calls["n"] += 1
            wanted = category is (Category.WORKSPACE if failure == "scratch" else Category.WEIGHTS)
            if wanted and (failure == "scratch" or calls["n"] == 2):
                raise OutOfMemory("injected allocation failure")
            return original(size, category, owner, **kw)

        monkeypatch.setattr(engine.domain, "allocate", allocate)
    with pytest.raises((OSError, ValueError, OutOfMemory), match="injected"):
        store.fetch(0, [0, 1, 4, 5, 6])
    monkeypatch.undo()
    assert dict(store._experts) == before[0] and engine.domain.used == before[1]  # nothing added, nothing leaked
    assert engine.domain.used_by(Category.WORKSPACE) == 0
    assert engine.telemetry.expert_failed_acquisitions == 1
    clean(engine)
    mats = store.fetch(0, [0, 1, 4, 5, 6])  # the retry
    assert sorted(mats) == [0, 1, 4, 5, 6] and all(m is not None for m in mats.values())
    store.release(0, mats)
    clean(engine)
    np.testing.assert_array_equal(logits_of(engine), logits_of(Engine(files["moe"])))


def test_a_failing_forward_pass_releases_its_experts(files, monkeypatch):
    engine = Engine(files["moe"])
    reference = logits_of(Engine(files["moe"]))
    real = engine.dispatcher.run
    state = {"n": 0}

    def run(group, kind, *inputs, **params):
        if kind == "moe_experts":
            state["n"] += 1
            if state["n"] == 2:
                raise RuntimeError("injected kernel failure")
        return real(group, kind, *inputs, **params)

    monkeypatch.setattr(engine.model.d, "run", run)
    session = engine.session(len(TOKENS) + 1)
    with pytest.raises(RuntimeError, match="injected kernel failure"):
        session.prefill(TOKENS)
    clean(engine)
    assert session.position == 0 and session.tokens == []  # the failed pass committed nothing
    monkeypatch.undo()
    np.testing.assert_array_equal(session.prefill(TOKENS, all_logits=True).logits, reference)
    session.close()


# -- R3: the expert-memory contract --------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["cpu", "torch"])
@pytest.mark.parametrize("experts_at_once", [1, 2, 3])
def test_a_large_expert_union_runs_within_the_ceiling_with_identical_results(files, experts_at_once, backend):
    if backend == "torch":
        pytest.importorskip("torch")
    unrestricted = Engine(files["moe"], prefer=(backend,))
    reference = logits_of(unrestricted)
    assert unrestricted.weights.peak_resident_bytes >= 6 * ONE_EXPERT  # the union really is large
    limit = experts_at_once * ONE_EXPERT + 100  # not a multiple of the expert size
    engine = Engine(files["moe"], expert_cache_bytes=limit, prefer=(backend,))
    logits = logits_of(engine)
    np.testing.assert_array_equal(logits, reference)  # same additions in the same order
    assert engine.weights.peak_resident_bytes <= limit
    assert engine.weights.resident_bytes <= limit
    assert limit < unrestricted.weights.peak_resident_bytes  # so the layer really ran in several groups
    clean(engine)
    report = engine.memory_report()
    assert report["expert_arrays"]["ceiling_bytes"] == limit
    assert report["expert_arrays"]["peak_resident_bytes"] <= limit
    assert report["dense_weights_bytes"] > 0 and report["expert_loading_workspace"]["peak_reserved_bytes"] > 0
    assert report["process"]["rss_bytes"] is None or report["process"]["rss_bytes"] > limit  # RSS is not the ceiling


def test_a_ceiling_below_one_expert_is_refused_before_anything_runs(files):
    with pytest.raises(ExpertBudgetExceeded, match="cannot hold one expert"):
        Engine(files["moe"], expert_cache_bytes=ONE_EXPERT - 1)
    with pytest.raises(ValueError):
        Engine(files["moe"], expert_cache_bytes=-1)
    engine = Engine(files["moe"], expert_cache_bytes=2 * ONE_EXPERT)
    with pytest.raises(ExpertBudgetExceeded, match="split the layer"):
        engine.weights.fetch(0, [0, 1, 2])  # one acquisition may not exceed the ceiling
    assert engine.weights.resident_bytes == 0
    # an expert in use counts against the ceiling: the next group shrinks, nothing is evicted under it
    held = engine.weights.fetch(0, [0])
    assert engine.weights.groups(1, [0, 1, 2]) == [[0], [1], [2]]
    engine.weights.release(0, held)
    assert engine.weights.groups(1, [0, 1, 2]) == [[0, 1], [2]]


def test_insufficient_memory_fails_before_partial_acquisition_and_recovers(files):
    reference = logits_of(Engine(files["moe"]))
    probe = Engine(files["moe"])
    dense = probe.domain.used
    engine = Engine(files["moe"], memory_bytes=dense + 6 * ONE_EXPERT, reserved_fraction=0.0)
    hog = engine.domain.allocate(engine.domain.free - ONE_EXPERT // 2, Category.ACTIVATIONS, "other work", refs=1)
    session = engine.session(len(TOKENS) + 1) if engine.domain.free > 40_000 else None
    if session is None:  # make room for the session's KV cache only
        engine.domain.release(hog, free_when_unused=True)
        session = engine.session(len(TOKENS) + 1)
        hog = engine.domain.allocate(engine.domain.free - ONE_EXPERT // 2, Category.ACTIVATIONS, "other work", refs=1)
    with pytest.raises(ExpertBudgetExceeded, match="none can be made resident"):
        session.prefill(TOKENS)
    assert engine.weights.resident_bytes == 0 and session.position == 0
    clean(engine)
    engine.domain.release(hog, free_when_unused=True)  # memory pressure ends
    np.testing.assert_array_equal(session.prefill(TOKENS, all_logits=True).logits, reference)
    session.close()
    clean(engine)


@pytest.mark.parametrize("name", sorted(TRANSIENT_FACTOR))
def test_declared_dequantisation_scratch_bounds_the_measured_peak(name):
    type_id = next(k for k, v in GGML_TYPES.items() if v[0] == name)
    _, block, size = GGML_TYPES[type_id]
    n = 256 * 1024 // block
    raw = np.random.default_rng(0).integers(0, 256, n * size, dtype=np.uint8)
    with np.errstate(all="ignore"):
        dequantize(raw[: size * 8], type_id, block * 8)  # warm up lazy tables
        tracemalloc.start()
        out = dequantize(raw, type_id, n * block)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    assert peak - out.nbytes <= transient_bytes(type_id, n * block)


# -- R1: deterministic proposals need linear memory and keep the distribution ---------------------------
def test_sampling_with_a_deterministic_drafter_uses_linear_memory(tmp_path):
    peaks = {}
    for vocab in (4096, 16384):
        path = tmp_path / f"v{vocab}.gguf"
        tiny_qwen3(path, seed=1, vocab=vocab)
        engine = Engine(path)
        prompt = [1, 2, 3, 1, 2, 3, 1, 2]
        with engine.session(64) as s:  # warm up
            speculative_generate(s, NgramDrafter(), prompt, 2, k=3, temperature=1.0)
        tracemalloc.start()
        with engine.session(64) as s:
            r = speculative_generate(s, NgramDrafter(), prompt, 6, k=3, temperature=1.0, seed=1)
        peaks[vocab] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert sum(x.verified for x in r.rounds) > 0  # the deterministic path really ran
        assert peaks[vocab] < 64 * vocab * 8  # a few vocabulary-sized vectors, nowhere near vocab**2 * 8
    assert peaks[16384] < 6 * peaks[4096]  # 4x the vocabulary: about 4x the memory, not 16x


def test_point_mass_proposals_keep_the_target_distribution():
    rng = np.random.default_rng(0)
    p = np.array([0.5, 0.25, 0.125, 0.125])
    n = 40_000
    for proposed in range(4):
        counts = np.zeros(4)
        accepted = 0
        for _ in range(n):
            ok, token = accept_or_resample(p, proposed, None, rng)
            counts[token] += 1
            accepted += ok
            assert token == proposed if ok else token != proposed  # a rejection never returns the proposal
        chi2 = float(((counts - n * p) ** 2 / (n * p)).sum())
        assert chi2 < 16.3, (proposed, chi2)  # 3 degrees of freedom, p > 0.001
        assert abs(accepted / n - p[proposed]) < 0.01  # accepted with probability p[proposed]
    # the general rule with q equal to a point mass gives the same answers as the direct one
    one_hot = np.eye(4)[2]
    a = [accept_or_resample(p, 2, None, np.random.default_rng(s)) for s in range(200)]
    b = [accept_or_resample(p, 2, one_hot, np.random.default_rng(s)) for s in range(200)]
    assert a == b
    assert accept_or_resample(p, 0, p.copy(), np.random.default_rng(1)) == (True, 0)  # q == p: always accepted


def test_speculative_sampling_with_a_deterministic_drafter_follows_the_target(files):
    target = Engine(files["small"])
    prompt, n = [1, 2, 3, 1, 2, 3, 1], 3000
    with target.session(8) as s:
        p = np.exp(s.prefill(prompt).logits[-1].astype(np.float64))
    p /= p.sum()
    likely = int(np.argmax(p))

    class Fixed:  # a deterministic drafter: always proposes the most likely token
        name = "fixed"

        def propose(self, committed, k, *, temperature=0.0, rng=None):
            return Proposal([likely] * k)

    counts = np.zeros(16)
    accepted = 0
    for seed in range(n):
        with target.session(16) as s:
            r = speculative_generate(s, Fixed(), prompt, 2, k=1, temperature=1.0, seed=seed)
        counts[r.tokens[0]] += 1
        accepted += r.rounds[0].accepted
    mask = p * n >= 5
    chi2 = float(((counts[mask] - n * p[mask]) ** 2 / (n * p[mask])).sum())
    assert chi2 < 37.7, chi2  # at most 15 degrees of freedom, p > 0.001
    assert abs(accepted / n - p[likely]) < 0.04  # accepted with probability p[proposal]


# -- R2: the session holds exactly what was returned -----------------------------------------------------
class Scripted:
    """Proposes the known continuation, wrong at chosen output positions."""

    name = "scripted"

    def __init__(self, prompt_len: int, continuation: list[int], wrong_at=(), vocab: int = 96) -> None:
        self.prompt_len, self.continuation, self.wrong_at, self.vocab = prompt_len, continuation, set(wrong_at), vocab

    def propose(self, committed, k, *, temperature=0.0, rng=None):
        done = len(committed) - self.prompt_len
        out = []
        for j in range(done, done + k):
            token = self.continuation[j] if j < len(self.continuation) else 0
            out.append((token + 1) % self.vocab if j in self.wrong_at else token)
        return Proposal(out)


@pytest.fixture(scope="module")
def story(tmp_path_factory):
    """A model and prompt whose greedy continuation starts with distinct tokens."""
    root = tmp_path_factory.mktemp("story")
    for seed in range(1, 40):
        path = root / f"m{seed}.gguf"
        tiny_qwen3(path, seed=seed, scale=0.2)
        engine = Engine(path)
        prompt = [5, 17, 3, 44, 9, 60, 2, 31]
        with engine.session(64) as s:
            ref = generate(s, prompt, 14).tokens
        if len(set(ref[:8])) == 8:
            return engine, prompt, ref
    raise AssertionError("no tiny model with a distinct continuation found")


def check_state(engine, session, prompt, returned):
    """The session holds prompt + returned[:-1]; continuing equals a fresh prefill of the returned context."""
    assert session.position == len(prompt) + len(returned) - 1
    assert session.tokens == prompt + returned[:-1]
    with engine.session(64) as fresh:
        expected = fresh.prefill(prompt + returned, all_logits=True)
        fresh_k = fresh.kv.k[:, :session.position].copy()
    np.testing.assert_allclose(session.kv.k[:, :session.position], fresh_k, rtol=0, atol=2e-5 * np.abs(fresh_k).max())
    resumed = session.step(returned[-1]).logits[-1]
    np.testing.assert_allclose(resumed, expected.logits[-1], rtol=0, atol=2e-5 * np.abs(expected.logits).max())


CASES = {
    "stop at the first accepted draft": ({}, 0),
    "stop at a middle accepted draft": ({}, 2),
    "stop at the last accepted draft": ({}, 3),
    "stop at the bonus token": ({}, 4),
    "stop at a rejection replacement": ({"wrong_at": {2}}, 2),
    "stop in a later round": ({"wrong_at": {1}}, 5),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_stopping_leaves_the_session_at_the_returned_sequence(story, case):
    engine, prompt, ref = story
    options, stop_index = CASES[case]
    drafter = Scripted(len(prompt), ref, **options)
    with engine.session(64) as s:
        r = speculative_generate(s, drafter, prompt, 12, k=4, stop={ref[stop_index]})
        assert r.tokens == ref[:stop_index + 1] and r.stopped == "stop_token"
        first = r.rounds[0]
        if not options and stop_index < 4:  # all four drafts were verified and accepted; fewer were returned
            assert (first.verified, first.accepted, first.returned) == (4, 4, stop_index + 1)
            assert first.rows == len(prompt) + 4  # the work beyond the stop is still counted
        assert sum(x.returned for x in r.rounds) == len(r.tokens)
        check_state(engine, s, prompt, r.tokens)


@pytest.mark.parametrize("length", [1, 2, 3, 5, 6])
@pytest.mark.parametrize("wrong_at", [(), (1,), (0, 1, 2)])
def test_the_length_limit_leaves_the_session_at_the_returned_sequence(story, length, wrong_at):
    engine, prompt, ref = story
    with engine.session(64) as s:
        r = speculative_generate(s, Scripted(len(prompt), ref, wrong_at), prompt, length, k=4)
        assert r.tokens == ref[:length] and r.stopped == "length"
        check_state(engine, s, prompt, r.tokens)


def test_generation_continues_from_the_state_a_call_left(story):
    engine, prompt, ref = story
    with engine.session(64) as s:
        first = speculative_generate(s, Scripted(len(prompt), ref), prompt, 12, k=4, stop={ref[1]})
        assert first.tokens == ref[:2]
        context = prompt + first.tokens
        second = speculative_generate(s, Scripted(len(context), ref[2:], wrong_at={1}), context, 5, k=3)
        assert second.tokens == ref[2:7]
        third = generate(s, context + second.tokens, 4)  # plain decoding continues too
        assert third.tokens == ref[7:11]
        check_state(engine, s, prompt, ref[:11])
    with engine.session(64) as s:
        s.prefill(prompt[:3])
        with pytest.raises(ValueError, match="proper prefix"):
            speculative_generate(s, None, [9, 9, 9, 9], 2)  # the session holds something else
        with pytest.raises(ValueError, match="proper prefix"):
            generate(s, prompt[:3], 2)  # nothing left to feed
        assert generate(s, prompt, 3).tokens == ref[:3]  # continuing a partly fed prompt is fine
    with engine.session(64) as s:
        assert speculative_generate(s, None, prompt, 0).tokens == [] and s.position == 0


def test_sampling_also_leaves_the_session_at_the_returned_sequence(files):
    target, draft = Engine(files["small"]), Engine(files["small-draft"])
    prompt = [1, 2, 3]
    for seed in range(12):
        with target.session(32) as s:
            r = speculative_generate(s, ModelDrafter(draft, 32), prompt, 7, k=3, temperature=1.0, seed=seed,
                                     stop={5})  # fmt: skip
            assert s.position == len(prompt) + len(r.tokens) - 1 and s.tokens == prompt + r.tokens[:-1]
            assert 5 not in r.tokens[:-1]


# -- R6: a prefix session is all or nothing ---------------------------------------------------------------
def test_prefix_session_failures_leave_nothing_behind(files, monkeypatch):
    engine = Engine(files["dense"])
    cache = PrefixCache(engine, block=8)
    tokens = list(range(1, 41))
    s, reused = cache.session(tokens, 64)
    assert reused == 0
    s.prefill(tokens)
    assert cache.store(s, tokens) == 40
    s.close()
    baseline = (len(engine.admission.active), engine.domain.used_by(Category.KV))
    with pytest.raises(ValueError, match="cannot hold a prompt"):
        cache.session(tokens, 16)  # validated before anything is admitted
    with pytest.raises(ValueError, match="non-empty"):
        cache.session([], 16)
    assert (len(engine.admission.active), engine.domain.used_by(Category.KV)) == baseline

    def boom(*_args, **_kwargs):
        raise MemoryError("injected failure while copying the prefix")

    monkeypatch.setattr(engine.domain, "touch", boom)
    with pytest.raises(MemoryError, match="injected"):
        cache.session([*tokens, 50, 51], 64)  # would reuse the 40 cached tokens
    monkeypatch.undo()
    assert (len(engine.admission.active), engine.domain.used_by(Category.KV)) == baseline  # the session was closed
    # a valid request still works, and reuse agrees with a fresh prefill
    question = [*tokens, 50, 51]
    s, reused = cache.session(question, 64)
    assert reused == 40 and s.tokens == tokens
    logits = s.prefill(question[reused:]).logits[-1]
    s.close()
    expected = logits_of(engine, question)[-1]
    np.testing.assert_allclose(logits, expected, rtol=0, atol=2e-5 * np.abs(expected).max())
    # a session too small for the whole cached prefix reuses a shorter one, never overflows
    cache.store(*_prefilled(engine, tokens[:16]))
    s, reused = cache.session(tokens, 40)
    assert reused == 16
    s.close()
    cache.check_invariant()


def _prefilled(engine, tokens):
    s = engine.session(64)
    s.prefill(tokens)
    return s, tokens


def test_prefix_entries_reclaimed_through_the_domain_disappear_from_the_cache(files):
    engine = Engine(files["dense"])
    cache = PrefixCache(engine, block=8)
    tokens = list(range(1, 33))
    s, _ = cache.session(tokens, 64)
    s.prefill(tokens)
    cache.store(s, tokens)
    s.close()
    assert cache.longest([*tokens, 99]) == 32
    evicted = engine.domain.evict(engine.domain.free + 1, category=Category.KV)
    assert len(evicted) == 1 and cache.longest([*tokens, 99]) == 0 and cache.stats.evicted == 1
    cache.check_invariant()
    with pytest.raises(ValueError, match="does not hold these tokens"):
        other = engine.session(64)
        other.prefill(tokens)
        cache.store(other, [9] * 32)
