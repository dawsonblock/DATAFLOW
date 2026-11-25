"""Shared-prefix KV cache: requests that begin with the same tokens (a system prompt, a
document, earlier turns of a conversation) reuse the keys and values computed for them instead
of recomputing the prefix.

Prefixes are cached at block boundaries (``block`` tokens).  Each cached prefix is charged to
the host memory domain as KV memory; entries not in use are evicted least recently used when
the cache exceeds its byte limit or the domain needs room.  Reuse copies the cached rows into
the new session's cache (no recomputation; a copy, not a shared page).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from basp.memory import Block, Category

from .session import Session


@dataclass
class PrefixStats:
    lookups: int = 0
    hits: int = 0
    reused_tokens: int = 0
    stored: int = 0
    evicted: int = 0


class PrefixCache:
    def __init__(self, engine, *, block: int = 16, capacity_bytes: int = 256 << 20) -> None:
        if block <= 0 or capacity_bytes <= 0:
            raise ValueError("block and capacity must be positive")
        self.engine, self.block, self.capacity = engine, block, capacity_bytes
        self._entries: OrderedDict[tuple[int, ...], tuple[np.ndarray, np.ndarray, Block]] = OrderedDict()
        self.stats = PrefixStats()

    def _bytes(self) -> int:
        return sum(b.size for _, _, b in self._entries.values())

    def longest(self, tokens: list[int]) -> int:
        """Length of the longest cached prefix of ``tokens`` that leaves at least one token to feed."""
        n = (len(tokens) - 1) // self.block * self.block
        while n > 0:
            if tuple(tokens[:n]) in self._entries:
                return n
            n -= self.block
        return 0

    def session(self, tokens: list[int], max_tokens: int) -> tuple[Session, int]:
        """A new session already holding the longest cached prefix of ``tokens``; returns the
        session and the number of tokens reused (feed ``tokens[n:]`` next)."""
        self.stats.lookups += 1
        session = self.engine.session(max_tokens)
        n = self.longest(list(tokens))
        if n:
            key = tuple(tokens[:n])
            k, v, _ = self._entries[key]
            self._entries.move_to_end(key)
            session.kv.k[:, :n] = k
            session.kv.v[:, :n] = v
            session.position = n
            self.stats.hits += 1
            self.stats.reused_tokens += n
        return session, n

    def store(self, session: Session, tokens: list[int]) -> int:
        """Cache the longest block-aligned prefix of the tokens ``session`` holds; returns its length."""
        n = min(len(tokens), session.position) // self.block * self.block
        key = tuple(tokens[:n])
        if n == 0 or key in self._entries:
            return 0
        k = session.kv.k[:, :n].copy()
        v = session.kv.v[:, :n].copy()
        nbytes = k.nbytes + v.nbytes
        if nbytes > self.capacity:
            return 0
        domain = self.engine.domain
        while self._entries and (self._bytes() + nbytes > self.capacity
                                 or not domain.can_allocate(nbytes, Category.KV)):  # fmt: skip
            _, (_, _, old) = self._entries.popitem(last=False)
            domain.free_block(old)
            self.stats.evicted += 1
        if not domain.can_allocate(nbytes, Category.KV):
            return 0
        block = domain.allocate(nbytes, Category.KV, f"prefix[{n}]")
        self._entries[key] = (k, v, block)
        self.stats.stored += 1
        return n

    def clear(self) -> None:
        for _, _, block in self._entries.values():
            self.engine.domain.free_block(block)
        self._entries.clear()
