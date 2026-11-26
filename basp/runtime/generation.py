"""Token generation: greedy (deterministic) or temperature sampling with a fixed seed."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .session import Session


@dataclass
class GenerationResult:
    prompt: list[int]
    tokens: list[int] = field(default_factory=list)
    logits: list[np.ndarray] = field(default_factory=list)  # one row per generated token
    stopped: str = "length"  # length | stop_token


def generate(session: Session, prompt: Iterable[int], max_new_tokens: int, *, stop: Iterable[int] = (),
             temperature: float = 0.0, seed: int = 0, keep_logits: bool = False) -> GenerationResult:  # fmt: skip
    """Generate up to ``max_new_tokens`` tokens.

    ``session`` is fresh or already holds a proper prefix of ``prompt`` (continuation).  A call
    that returns ``N > 0`` tokens leaves the session holding ``prompt + returned[:-1]``: the last
    returned token is pending, so ``generate(session, prompt + returned, ...)`` continues.
    """
    prompt = [int(t) for t in prompt]
    if not prompt:
        raise ValueError("generation needs a non-empty prompt")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not session.holds_prefix_of(prompt):
        raise ValueError("the session must be fresh or hold a proper prefix of the prompt")
    stop = set(stop)
    rng = np.random.default_rng(seed)
    result = GenerationResult(prompt)
    if max_new_tokens == 0:
        return result
    logits = session.prefill(prompt[session.position:]).logits[-1]
    for _ in range(max_new_tokens):
        if keep_logits:
            result.logits.append(logits.copy())
        if temperature <= 0:
            token = int(np.argmax(logits))
        else:
            z = (logits - logits.max()) / temperature
            p = np.exp(z.astype(np.float64))
            token = int(rng.choice(len(p), p=p / p.sum()))
        result.tokens.append(token)
        if token in stop:
            result.stopped = "stop_token"
            break
        if len(result.tokens) == max_new_tokens:
            break
        logits = session.step(token).logits[-1]
    return result
