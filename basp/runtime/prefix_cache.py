"""Shared-prefix KV cache: requests that begin with the same tokens (a system prompt, a
document, earlier turns of a conversation) reuse the keys and values computed for them instead
of recomputing the prefix.

Prefixes are cached at block boundaries (``block`` tokens).  Each cached prefix is charged to
the host memory domain as KV memory and registers the cache as its owner, so whoever reclaims
the block (this cache's own eviction, or anything else through the domain) drops the entry
first.  Entries not in use are evicted least recently used when the cache exceeds its byte
limit or the domain needs room.  Reuse copies the cached rows into the new session's cache (no
recomputation; a copy, not a shared page).

Creating a session is all or nothing: lengths are validated before the session is admitted,
and if copying the prefix fails the session is closed and its reservation released.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial

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

    def _forget(self, key: tuple[int, ...], _block: Block) -> None:
        """Owner's reclamation hook: drop the cached rows before the domain frees the block."""
        if self._entries.pop(key, None) is not None:
            self.stats.evicted += 1

    def longest(self, tokens: list[int], limit: int | None = None) -> int:
        """Length of the longest cached prefix of ``tokens`` that leaves at least one token to
        feed (and, with ``limit``, is at most ``limit`` tokens long)."""
        n = (len(tokens) - 1) // self.block * self.block
        if limit is not None:
            n = min(n, limit // self.block * self.block)
        while n > 0:
            if tuple(tokens[:n]) in self._entries:
                return n
            n -= self.block
        return 0

    def session(self, tokens: list[int], max_tokens: int) -> tuple[Session, int]:
        """A new session already holding the longest cached prefix of ``tokens``; returns the
        session and the number of tokens reused (feed ``tokens[n:]`` next).

        Validated before anything is admitted: the prompt must be non-empty and fit the session.
        If initialisation fails after admission, the session is closed (its KV reservation is
        released) before the error propagates.
        """
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise ValueError("a prefix session needs a non-empty prompt")
        if max_tokens < len(tokens):
            raise ValueError(f"a session of {max_tokens} tokens cannot hold a prompt of {len(tokens)}")
        self.stats.lookups += 1
        session = self.engine.session(max_tokens)
        try:
            n = self.longest(tokens, limit=max_tokens - 1)
            if n:
                key = tuple(tokens[:n])
                k, v, block = self._entries[key]
                self.engine.domain.touch(block)
                self._entries.move_to_end(key)
                session.kv.k[:, :n] = k
                session.kv.v[:, :n] = v
                session.position = n
                session.tokens = list(key)
        except BaseException:
            session.close()
            raise
        if n:
            self.stats.hits += 1
            self.stats.reused_tokens += n
        return session, n

    def store(self, session: Session, tokens: list[int]) -> int:
        """Cache the longest block-aligned prefix of the tokens ``session`` holds; returns its
        length (0 if nothing was stored).  ``tokens`` must be what the session holds."""
        tokens = [int(t) for t in tokens]
        n = min(len(tokens), session.position) // self.block * self.block
        if session.tokens[:n] != tokens[:n]:
            raise ValueError("the session does not hold these tokens")
        key = tuple(tokens[:n])
        if n == 0 or key in self._entries:
            return 0
        domain = self.engine.domain
        nbytes = 2 * session.kv.k[:, :n].nbytes
        if nbytes > self.capacity:
            return 0
        # make room first (least recently used entries not in use), then reserve, then copy
        for old in list(self._entries):
            if self._bytes() + nbytes <= self.capacity and domain.can_allocate(nbytes, Category.KV):
                break
            block = self._entries[old][2]
            if block.evictable:
                domain.free_block(block)  # single reclamation path; the hook drops the entry
        if self._bytes() + nbytes > self.capacity or not domain.can_allocate(nbytes, Category.KV):
            return 0
        block = domain.allocate(nbytes, Category.KV, f"prefix[{n}]", on_reclaim=partial(self._forget, key))
        try:
            self._entries[key] = (session.kv.k[:, :n].copy(), session.kv.v[:, :n].copy(), block)
        except BaseException:
            self._entries.pop(key, None)
            block.on_reclaim = None
            domain.free_block(block)
            raise
        self.stats.stored += 1
        return n

    def clear(self) -> None:
        for _, _, block in list(self._entries.values()):
            self.engine.domain.free_block(block)

    def check_invariant(self) -> None:
        for key, (_, _, block) in self._entries.items():
            assert block.live, f"prefix of {len(key)} tokens keeps rows for a freed block"
