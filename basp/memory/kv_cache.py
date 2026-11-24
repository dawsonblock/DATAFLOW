"""Paged KV cache: fixed-size blocks, per-sequence block tables, copy-on-write prefixes.

Key/value state is stored in physical blocks of ``block_tokens`` tokens. Each sequence owns a
block table mapping logical block index to physical block id, and blocks are allocated only
when a sequence grows into them, so memory follows actual sequence length rather than the
maximum context. ``fork`` shares every parent block by reference count; the first append into
a shared, partially filled last block copies that block (copy-on-write), while full shared
blocks stay shared for the lifetime of both sequences. Free blocks are reused lowest id first,
so placement is deterministic. This models allocation and traffic only; it holds no tensor data.
"""

from __future__ import annotations

import heapq
from collections.abc import Hashable
from dataclasses import dataclass, field


class KVCacheError(Exception):
    """Invalid KV cache operation."""


class KVCacheFull(KVCacheError):
    """Not enough free blocks to satisfy an append."""


@dataclass(frozen=True)
class KVLayout:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2

    @property
    def bytes_per_token(self) -> int:
        """Keys and values for every layer."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes


@dataclass
class _Sequence:
    tokens: int = 0
    blocks: list[int] = field(default_factory=list)


class PagedKVCache:
    def __init__(self, layout: KVLayout, capacity_bytes: int, block_tokens: int = 16) -> None:
        if block_tokens <= 0:
            raise ValueError("block_tokens must be > 0")
        if layout.bytes_per_token <= 0:
            raise ValueError("layout must have a positive bytes_per_token")
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be >= 0")
        self._layout = layout
        self._block_tokens = int(block_tokens)
        self._block_bytes = self._block_tokens * layout.bytes_per_token
        self._n_blocks = int(capacity_bytes) // self._block_bytes
        self._free: list[int] = list(range(self._n_blocks))  # min-heap: lowest id first
        self._ref = [0] * self._n_blocks
        self._seqs: dict[Hashable, _Sequence] = {}
        self._cow_copies = 0

    # -- properties ---------------------------------------------------------------------

    @property
    def layout(self) -> KVLayout:
        return self._layout

    @property
    def block_tokens(self) -> int:
        return self._block_tokens

    @property
    def block_bytes(self) -> int:
        return self._block_bytes

    @property
    def n_blocks(self) -> int:
        return self._n_blocks

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_bytes(self) -> int:
        return (self._n_blocks - len(self._free)) * self._block_bytes

    @property
    def cow_copies(self) -> int:
        """Number of block copies performed by copy-on-write so far."""
        return self._cow_copies

    # -- sequences ----------------------------------------------------------------------

    def _seq(self, seq_id: Hashable) -> _Sequence:
        try:
            return self._seqs[seq_id]
        except KeyError:
            raise KVCacheError(f"unknown sequence {seq_id!r}") from None

    def create_sequence(self, seq_id: Hashable) -> None:
        if seq_id in self._seqs:
            raise KVCacheError(f"sequence {seq_id!r} already exists")
        self._seqs[seq_id] = _Sequence()

    def append(self, seq_id: Hashable, n_tokens: int = 1) -> tuple[int, ...]:
        if n_tokens < 0:
            raise ValueError("n_tokens must be >= 0")
        seq = self._seq(seq_id)
        if n_tokens == 0:
            return ()
        room = len(seq.blocks) * self._block_tokens - seq.tokens
        copy_last = room > 0 and self._ref[seq.blocks[-1]] > 1
        grow = -(-max(0, n_tokens - room) // self._block_tokens)
        need = grow + copy_last
        if need > len(self._free):
            raise KVCacheFull(f"sequence {seq_id!r} needs {need} blocks, {len(self._free)} free")
        new: list[int] = []
        if copy_last:
            block = self._take()
            self._ref[seq.blocks[-1]] -= 1
            seq.blocks[-1] = block
            self._cow_copies += 1
            new.append(block)
        for _ in range(grow):
            block = self._take()
            seq.blocks.append(block)
            new.append(block)
        seq.tokens += n_tokens
        return tuple(new)

    def fork(self, parent_seq: Hashable, child_seq: Hashable) -> None:
        parent = self._seq(parent_seq)
        if child_seq in self._seqs:
            raise KVCacheError(f"sequence {child_seq!r} already exists")
        for block in parent.blocks:
            self._ref[block] += 1
        self._seqs[child_seq] = _Sequence(parent.tokens, list(parent.blocks))

    def free_sequence(self, seq_id: Hashable) -> int:
        seq = self._seq(seq_id)
        del self._seqs[seq_id]
        released = 0
        for block in seq.blocks:
            self._ref[block] -= 1
            if self._ref[block] == 0:
                heapq.heappush(self._free, block)
                released += 1
        return released

    def _take(self) -> int:
        block = heapq.heappop(self._free)
        self._ref[block] = 1
        return block

    # -- introspection ------------------------------------------------------------------

    def tokens(self, seq_id: Hashable) -> int:
        return self._seq(seq_id).tokens

    def blocks(self, seq_id: Hashable) -> tuple[int, ...]:
        return tuple(self._seq(seq_id).blocks)

    def read_bytes(self, seq_id: Hashable) -> int:
        """Bytes read when attention covers the whole sequence."""
        return self._seq(seq_id).tokens * self._layout.bytes_per_token

    def refcount(self, block_id: int) -> int:
        if not 0 <= block_id < self._n_blocks:
            raise KVCacheError(f"block {block_id} out of range")
        return self._ref[block_id]

    def sequences(self) -> tuple[Hashable, ...]:
        return tuple(self._seqs)

    def check_invariants(self) -> None:
        refs = [0] * self._n_blocks
        for seq_id, seq in self._seqs.items():
            assert seq.tokens >= 0, f"negative token count in {seq_id!r}"
            assert len(seq.blocks) == -(-seq.tokens // self._block_tokens), (
                f"block table of {seq_id!r} inconsistent with its token count"
            )
            assert len(set(seq.blocks)) == len(seq.blocks), f"{seq_id!r} maps a block twice"
            for block in seq.blocks:
                assert 0 <= block < self._n_blocks, "block id out of range"
                refs[block] += 1
        assert refs == self._ref, "refcounts do not match block-table references"
        free = set(self._free)
        assert len(free) == len(self._free), "duplicate block in free list"
        for block in range(self._n_blocks):
            assert (block in free) != (refs[block] > 0), "block must be free xor referenced"
