"""Speculative decoding: propose several tokens cheaply, verify them together on the target.

A *drafter* proposes ``k`` next tokens -- a small model of the same tokenizer, or a lookup of
the context itself.  The target model then runs **one** forward pass over the pending token and
all ``k`` proposals.  On a mixture-of-experts target the proposals are processed as one batch,
so every expert selected by any of them is read once for all of them (grouped execution).

Acceptance keeps the target's output exact:

* **greedy** (temperature 0): a proposal is kept while it equals the target's own greedy choice;
  the first disagreement is replaced by the target's choice.  The tokens are the target's greedy
  tokens, up to floating-point differences between batched and single-row arithmetic (measured,
  not assumed: ``benchmarks.speculative_bench`` compares against sequential decoding);
* **sampling**: proposal ``x`` with drafter probability ``q(x)`` is accepted with probability
  ``min(1, p(x) / q(x))``; on rejection a token is drawn from ``max(0, p - q)`` renormalised; if
  every proposal is accepted, one more token is drawn from ``p``.  The output then follows the
  target's distribution exactly (Leviathan et al. 2023; Chen et al. 2023).  A deterministic
  drafter has ``q`` = one-hot.

Rejected proposals are discarded by truncating the KV cache.  ``AdaptiveController`` chooses
``k`` before each round from the measured acceptance rate and the measured cost of drafting and
of verifying ``n`` rows, maximising expected accepted tokens per second; ``k = 0`` is ordinary
one-token decoding.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .session import Session


def _probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits.astype(np.float64) / temperature
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


@dataclass
class Proposal:
    tokens: list[int]
    probs: list[np.ndarray] | None = None  # the drafter's distribution for each token (None: deterministic)


class Drafter(Protocol):
    name: str

    def propose(self, committed: list[int], k: int, *, temperature: float, rng: np.random.Generator) -> Proposal: ...


class NgramDrafter:
    """Prompt lookup: find the latest earlier occurrence of the context's last ``n`` tokens
    (longest ``n`` first) and propose the tokens that followed it.  Costs no model call."""

    name = "prompt-lookup"

    def __init__(self, max_n: int = 3, min_n: int = 1) -> None:
        if not 1 <= min_n <= max_n:
            raise ValueError("need 1 <= min_n <= max_n")
        self.max_n, self.min_n = max_n, min_n

    def propose(self, committed: list[int], k: int, *, temperature: float = 0.0, rng=None) -> Proposal:
        if k <= 0:
            return Proposal([])
        for n in range(self.max_n, self.min_n - 1, -1):
            if len(committed) <= n:
                continue
            suffix = committed[-n:]
            for start in range(len(committed) - n - 1, -1, -1):
                if committed[start:start + n] == suffix:
                    follow = committed[start + n:start + n + k]
                    if follow:
                        return Proposal(list(follow))
        return Proposal([])


class ModelDrafter:
    """A small model with the same tokenizer, kept in step with the committed tokens."""

    def __init__(self, engine, max_tokens: int, name: str = "draft model") -> None:
        if engine.config.vocab <= 0:
            raise ValueError("draft model without a vocabulary")
        self.engine = engine
        self.session: Session = engine.session(max_tokens)
        self.fed: list[int] = []
        self.name = name

    def propose(self, committed: list[int], k: int, *, temperature: float, rng: np.random.Generator) -> Proposal:
        if k <= 0:
            return Proposal([])
        common = 0
        for a, b in zip(self.fed, committed):
            if a != b:
                break
            common += 1
        common = min(common, len(committed) - 1)  # always feed at least one token to get logits
        self.session.truncate(common)
        del self.fed[common:]
        logits = self.session.prefill(committed[common:]).logits[-1]
        self.fed.extend(committed[common:])
        tokens, probs = [], []
        for i in range(k):
            if temperature > 0:
                p = _probs(logits, temperature)
                token = int(rng.choice(len(p), p=p))
                probs.append(p)
            else:
                token = int(np.argmax(logits))
            tokens.append(token)
            if i + 1 < k:
                logits = self.session.step(token).logits[-1]
                self.fed.append(token)
        return Proposal(tokens, probs if temperature > 0 else None)

    def close(self) -> None:
        self.session.close()


class AdaptiveController:
    """Choose the draft length from measured acceptance and measured costs.

    Acceptance is modelled as independent per proposal with probability ``alpha``, estimated as
    accepted / (accepted + rounds that ended in a rejection) with exponential forgetting.  The
    expected tokens a round yields with ``k`` proposals are ``(1 - alpha**(k+1)) / (1 - alpha)``.
    Costs: drafting is ``c_d`` per proposal; verifying ``n`` rows is ``a + b n``, fitted by least
    squares on recent rounds.  ``k`` maximises expected tokens per second.
    """

    def __init__(self, k_max: int = 8, *, k_start: int = 4, prior_alpha: float = 0.5, decay: float = 0.95,
                 window: int = 32) -> None:  # fmt: skip
        if k_max < 0 or not 0 < prior_alpha < 1:
            raise ValueError("need k_max >= 0 and 0 < prior_alpha < 1")
        self.k_max, self.k_start, self.decay = k_max, min(k_start, k_max), decay
        self.accepted, self.rejections = prior_alpha * 2, (1 - prior_alpha) * 2  # a weak prior
        self.draft_cost: float | None = None
        self.verify: deque[tuple[int, float]] = deque(maxlen=window)
        self.rounds = 0

    @property
    def alpha(self) -> float:
        return self.accepted / (self.accepted + self.rejections)

    def verify_cost(self, rows: int) -> float | None:
        if not self.verify:
            return None
        n = np.array([r for r, _ in self.verify], float)
        t = np.array([s for _, s in self.verify], float)
        if len(set(n)) < 2:
            return float(t.mean())  # one row count seen: no slope yet (assume flat, then explore)
        b, a = np.polyfit(n, t, 1)
        return float(max(a + b * rows, 1e-9))

    def expected_tokens(self, k: int) -> float:
        a = min(self.alpha, 1 - 1e-9)
        return (1 - a ** (k + 1)) / (1 - a)

    def choose(self, *, limit: int | None = None) -> int:
        k_max = self.k_max if limit is None else max(0, min(self.k_max, limit))
        if self.rounds < 2 or self.draft_cost is None or not self.verify:
            return min(self.k_start, k_max)
        best, best_rate = 0, -1.0
        for k in range(k_max + 1):
            cost = k * self.draft_cost + self.verify_cost(1 + k)
            rate = self.expected_tokens(k) / cost
            if rate > best_rate:
                best, best_rate = k, rate
        return best

    def update(self, *, proposed: int, accepted: int, draft_s: float, rows: int, verify_s: float) -> None:
        self.rounds += 1
        self.accepted = self.accepted * self.decay + accepted
        self.rejections = self.rejections * self.decay + (1 if accepted < proposed else 0)
        if proposed:
            per = draft_s / proposed
            self.draft_cost = per if self.draft_cost is None else 0.8 * self.draft_cost + 0.2 * per
        self.verify.append((rows, verify_s))


@dataclass
class Round:
    proposed: int
    accepted: int
    emitted: int
    rows: int
    draft_s: float
    verify_s: float
    expert_requests: int
    expert_bytes: int


@dataclass
class SpeculativeResult:
    prompt: list[int]
    tokens: list[int] = field(default_factory=list)
    rounds: list[Round] = field(default_factory=list)
    stopped: str = "length"
    seconds: float = 0.0

    @property
    def acceptance(self) -> float:
        proposed = sum(r.proposed for r in self.rounds)
        return sum(r.accepted for r in self.rounds) / proposed if proposed else 0.0

    @property
    def tokens_per_target_call(self) -> float:
        return len(self.tokens) / len(self.rounds) if self.rounds else 0.0

    def summary(self) -> dict[str, Any]:
        n = max(len(self.tokens), 1)
        return {"tokens": len(self.tokens), "target_calls": len(self.rounds), "acceptance": self.acceptance,
                "tokens_per_target_call": self.tokens_per_target_call, "seconds": self.seconds,
                "tokens_per_s": len(self.tokens) / self.seconds if self.seconds else 0.0,
                "draft_seconds": sum(r.draft_s for r in self.rounds),
                "verify_seconds": sum(r.verify_s for r in self.rounds),
                "expert_requests_per_token": sum(r.expert_requests for r in self.rounds) / n,
                "expert_bytes_per_token": sum(r.expert_bytes for r in self.rounds) / n,
                "k_histogram": dict(sorted(_count(r.proposed for r in self.rounds).items()))}  # fmt: skip


def _count(values: Iterable[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def speculative_generate(target: Session, drafter: Drafter | None, prompt: Iterable[int], max_new_tokens: int, *,
                         k: int = 4, controller: AdaptiveController | None = None, temperature: float = 0.0,
                         stop: Iterable[int] = (), seed: int = 0) -> SpeculativeResult:  # fmt: skip
    """Generate with ``drafter`` proposing and ``target`` verifying.  ``target`` must be a fresh
    session; with ``controller`` the draft length adapts, otherwise it is ``k``."""
    prompt = [int(t) for t in prompt]
    if not prompt:
        raise ValueError("generation needs a non-empty prompt")
    if target.position:
        raise ValueError("speculative generation starts from an empty session")
    stop = set(stop)
    rng = np.random.default_rng(seed)
    tel = target.engine.telemetry
    result = SpeculativeResult(prompt)
    committed = list(prompt)
    fed = 0  # tokens of ``committed`` in the target's KV cache
    start = time.perf_counter()
    while len(result.tokens) < max_new_tokens:
        room = target.capacity - len(committed) - 1
        remaining = max_new_tokens - len(result.tokens) - 1
        limit = max(0, min(room, remaining))
        want = (controller.choose(limit=limit) if controller else min(k, limit)) if drafter is not None else 0
        t0 = time.perf_counter()
        proposal = drafter.propose(committed, want, temperature=temperature, rng=rng) if want else Proposal([])
        drafts = proposal.tokens[:want]
        t1 = time.perf_counter()
        before = (tel.expert_requests, tel.expert_bytes)
        rows = committed[fed:] + drafts
        logits = target.prefill(rows, all_logits=True).logits
        t2 = time.perf_counter()
        base = len(committed) - fed - 1  # the row that predicts the first proposal
        accepted, new = 0, []
        for i, d in enumerate(drafts):
            row = logits[base + i]
            if temperature <= 0:
                choice = int(np.argmax(row))
                if choice == d:
                    accepted += 1
                    new.append(d)
                    continue
                new.append(choice)
                break
            p = _probs(row, temperature)
            q = proposal.probs[i] if proposal.probs is not None else np.eye(len(p))[d]
            if rng.random() < min(1.0, p[d] / q[d]):
                accepted += 1
                new.append(d)
                continue
            residual = np.maximum(p - q, 0.0)
            new.append(int(rng.choice(len(p), p=residual / residual.sum())))
            break
        else:  # every proposal accepted: the last row gives one more token
            row = logits[base + len(drafts)]
            new.append(int(np.argmax(row)) if temperature <= 0 else int(rng.choice(len(row), p=_probs(row, temperature))))
        fed = len(committed) + accepted
        target.truncate(fed)  # rejected proposals leave the cache
        round_ = Round(len(drafts), accepted, len(new), len(rows), t1 - t0, t2 - t1,
                       tel.expert_requests - before[0], tel.expert_bytes - before[1])  # fmt: skip
        result.rounds.append(round_)
        if controller is not None:
            controller.update(proposed=len(drafts), accepted=accepted, draft_s=t1 - t0, rows=len(rows),
                              verify_s=t2 - t1)  # fmt: skip
        for token in new:
            committed.append(token)
            result.tokens.append(token)
            if token in stop:
                result.stopped = "stop_token"
                break
            if len(result.tokens) >= max_new_tokens:
                break
        if result.stopped == "stop_token":
            break
    result.seconds = time.perf_counter() - start
    return result
