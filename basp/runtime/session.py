"""One sequence: its KV cache (admitted against the host memory budget) and its position."""

from __future__ import annotations

import time

import numpy as np

from basp.memory import RequestShape

from .model import ForwardOutput, KVCache


class Session:
    _ids = 0

    def __init__(self, engine, max_tokens: int) -> None:
        config = engine.config
        if max_tokens <= 0:
            raise ValueError("a session needs room for at least one token")
        if max_tokens > config.context_length:
            raise ValueError(f"{max_tokens} tokens exceed the model's context of {config.context_length}")
        Session._ids += 1
        self.session_id = f"session{Session._ids}"
        kv_per_token = 2 * config.n_layers * config.n_kv_head * config.head_dim * 4
        admission = engine.admission.admit(self.session_id, RequestShape(0, max_tokens, kv_per_token))
        if not admission.admitted:
            raise MemoryError(f"session refused: {admission.reason}")
        self.engine = engine
        self.kv = KVCache(config, max_tokens)
        self.position = 0
        #: the tokens whose keys and values the cache holds (``len(tokens) == position``)
        self.tokens: list[int] = []
        self.closed = False

    def _run(self, ids, *, all_logits: bool, capture: bool, phase: str) -> ForwardOutput:
        if self.closed:
            raise RuntimeError("session is closed")
        ids = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        positions = np.arange(self.position, self.position + len(ids))
        start = time.perf_counter()
        out = self.engine.model.forward(ids, positions, self.kv, all_logits=all_logits, capture=capture)
        elapsed = time.perf_counter() - start
        tel = self.engine.telemetry
        if phase == "prefill":
            tel.prefill_tokens += len(ids)
            tel.prefill_seconds += elapsed
        else:
            tel.decode_tokens += len(ids)
            tel.decode_seconds += elapsed
        self.position += len(ids)
        self.tokens.extend(int(t) for t in ids)
        return out

    def prefill(self, ids, *, all_logits: bool = False, capture: bool = False) -> ForwardOutput:
        return self._run(ids, all_logits=all_logits, capture=capture, phase="prefill")

    def step(self, token: int, *, capture: bool = False) -> ForwardOutput:
        return self._run([token], all_logits=False, capture=capture, phase="decode")

    def truncate(self, position: int) -> None:
        """Forget every token from ``position`` on (speculative tokens that were rejected).  The
        KV entries stay in memory and are overwritten; attention never reads past the position."""
        if not 0 <= position <= self.position:
            raise ValueError(f"cannot truncate to {position}: the session holds {self.position} tokens")
        self.position = position
        del self.tokens[position:]

    def holds_prefix_of(self, tokens) -> bool:
        """Whether the cache holds exactly a proper prefix of ``tokens`` (so the rest can be fed)."""
        return self.position < len(tokens) and self.tokens == [int(t) for t in tokens[:self.position]]

    @property
    def capacity(self) -> int:
        return self.kv.max_tokens

    def close(self) -> None:
        if not self.closed:
            self.engine.admission.finish(self.session_id)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
