"""Perplexity from logits under teacher forcing."""

from __future__ import annotations

import numpy as np


def perplexity(logits: np.ndarray, tokens: np.ndarray) -> float:
    """``logits[t]`` predicts ``tokens[t + 1]``; float64 log-softmax."""
    logits = np.asarray(logits, dtype=np.float64)[:-1]
    targets = np.asarray(tokens)[1:]
    if len(targets) == 0:
        raise ValueError("perplexity needs at least two tokens")
    z = logits - logits.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    nll = -logp[np.arange(len(targets)), targets]
    return float(np.exp(nll.mean()))
