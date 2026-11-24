import random

import pytest

from basp.memory.kv_cache import KVCacheError, KVCacheFull, KVLayout, PagedKVCache

SMALL = KVLayout(n_layers=2, n_kv_heads=2, head_dim=4)  # 64 bytes per token


def make(n_blocks: int, block_tokens: int = 4) -> PagedKVCache:
    return PagedKVCache(SMALL, n_blocks * block_tokens * SMALL.bytes_per_token, block_tokens)


def test_qwen3_layout_bytes_per_token():
    layout = KVLayout(n_layers=48, n_kv_heads=4, head_dim=128)
    assert layout.bytes_per_token == 98_304
    cache = PagedKVCache(layout, capacity_bytes=1 << 30, block_tokens=16)
    assert cache.block_bytes == 16 * 98_304
    assert cache.n_blocks == (1 << 30) // (16 * 98_304)
    assert cache.layout is layout


def test_capacity_rounds_down_to_whole_blocks():
    cache = PagedKVCache(SMALL, capacity_bytes=4 * 64 * 3 + 100, block_tokens=4)
    assert cache.n_blocks == 3 and cache.free_blocks == 3 and cache.used_bytes == 0


def test_block_allocation_boundaries():
    cache = make(8)
    cache.create_sequence("s")
    assert cache.tokens("s") == 0 and cache.blocks("s") == ()
    assert cache.append("s") == (0,)  # first token allocates the first block
    assert cache.append("s", 3) == ()  # fills block 0 exactly
    assert cache.append("s") == (1,)  # token 5 crosses the boundary
    assert cache.append("s", 0) == ()
    assert cache.append("s", 3) == ()  # fills block 1 exactly
    assert cache.append("s", 4) == (2,)  # a whole block's worth lands in one new block
    assert cache.tokens("s") == 12 and cache.blocks("s") == (0, 1, 2)
    assert cache.free_blocks == 5
    cache.check_invariants()


def test_append_block_counts_match_ceiling():
    cache = make(8)
    cache.create_sequence("s")
    assert cache.append("s", 5) == (0, 1)
    assert cache.append("s", 8) == (2, 3)  # 3 fit in block 1, the other 5 need two blocks
    assert cache.tokens("s") == 13 and cache.blocks("s") == (0, 1, 2, 3)
    assert cache.used_bytes == 4 * cache.block_bytes and cache.free_blocks == 4
    assert cache.read_bytes("s") == 13 * SMALL.bytes_per_token
    cache.check_invariants()


def test_sequence_errors():
    cache = make(4)
    cache.create_sequence("a")
    with pytest.raises(KVCacheError):
        cache.create_sequence("a")
    for call in (cache.tokens, cache.blocks, cache.read_bytes, cache.append, cache.free_sequence):
        with pytest.raises(KVCacheError):
            call("missing")
    with pytest.raises(KVCacheError):
        cache.fork("missing", "b")
    with pytest.raises(KVCacheError):
        cache.fork("a", "a")
    with pytest.raises(KVCacheError):
        cache.refcount(99)
    with pytest.raises(ValueError):
        cache.append("a", -1)
    assert cache.sequences() == ("a",)


def test_fork_shares_blocks():
    cache = make(8)
    cache.create_sequence("p")
    cache.append("p", 6)
    cache.fork("p", "c")
    assert cache.blocks("c") == cache.blocks("p") == (0, 1)
    assert cache.tokens("c") == 6
    assert cache.refcount(0) == cache.refcount(1) == 2
    assert cache.free_blocks == 6  # forking allocates nothing
    assert cache.sequences() == ("p", "c")
    cache.check_invariants()


def test_copy_on_write_of_partial_last_block():
    cache = make(8)
    cache.create_sequence("p")
    cache.append("p", 6)  # blocks 0 (full), 1 (half full)
    cache.fork("p", "c")
    assert cache.append("c") == (2,)  # copy of block 1, no further growth
    assert cache.blocks("p") == (0, 1)  # parent untouched
    assert cache.blocks("c") == (0, 2)
    assert (cache.refcount(0), cache.refcount(1), cache.refcount(2)) == (2, 1, 1)
    assert cache.cow_copies == 1
    assert cache.append("p") == ()  # parent is sole owner again: writes in place
    assert cache.append("c", 3) == (3,)  # fills the copy, then grows normally
    assert cache.cow_copies == 1
    assert cache.tokens("p") == 7 and cache.tokens("c") == 10
    cache.check_invariants()


def test_copy_on_write_plus_growth_in_one_append():
    cache = make(8)
    cache.create_sequence("p")
    cache.append("p", 2)
    cache.fork("p", "c")
    assert cache.append("c", 7) == (1, 2, 3)  # copy, then two more blocks for 5 tokens
    assert cache.blocks("c") == (1, 2, 3) and cache.blocks("p") == (0,)
    cache.check_invariants()


def test_shared_full_block_needs_no_copy():
    cache = make(8)
    cache.create_sequence("p")
    cache.append("p", 8)  # two full blocks
    cache.fork("p", "c")
    assert cache.append("c") == (2,)
    assert cache.append("p") == (3,)
    assert cache.blocks("c") == (0, 1, 2) and cache.blocks("p") == (0, 1, 3)
    assert cache.refcount(0) == cache.refcount(1) == 2
    assert cache.cow_copies == 0
    cache.check_invariants()


def test_free_with_shared_blocks():
    cache = make(8)
    cache.create_sequence("p")
    cache.append("p", 8)
    cache.fork("p", "c")
    cache.append("c", 2)  # private block 2
    assert cache.free_sequence("p") == 0  # every parent block is still referenced
    assert cache.refcount(0) == cache.refcount(1) == 1
    assert cache.blocks("c") == (0, 1, 2)
    assert cache.free_sequence("c") == 3
    assert cache.free_blocks == 8 and cache.used_bytes == 0 and cache.sequences() == ()
    cache.check_invariants()


def test_full_is_atomic():
    cache = make(3)
    cache.create_sequence("a")
    cache.append("a", 6)  # blocks 0, 1; one free block left
    before = (cache.tokens("a"), cache.blocks("a"), cache.free_blocks)
    with pytest.raises(KVCacheFull):
        cache.append("a", 7)  # needs two more blocks
    assert (cache.tokens("a"), cache.blocks("a"), cache.free_blocks) == before
    assert cache.append("a", 6) == (2,)  # exactly fills the cache
    with pytest.raises(KVCacheFull):
        cache.append("a")
    cache.check_invariants()


def test_full_is_atomic_for_copy_on_write():
    cache = make(2)
    cache.create_sequence("p")
    cache.append("p", 2)
    cache.fork("p", "c")
    cache.create_sequence("x")
    cache.append("x", 1)  # takes the last free block
    with pytest.raises(KVCacheFull):
        cache.append("c")  # the copy has nowhere to go
    assert cache.blocks("c") == cache.blocks("p") == (0,)
    assert cache.refcount(0) == 2 and cache.tokens("c") == 2 and cache.cow_copies == 0
    assert isinstance(KVCacheFull("x"), KVCacheError)
    cache.check_invariants()


def test_free_blocks_reused_lowest_id_first():
    cache = make(6)
    for name in "abc":
        cache.create_sequence(name)
        cache.append(name, 8)
    assert cache.blocks("c") == (4, 5)
    cache.free_sequence("b")  # releases 2, 3
    cache.free_sequence("a")  # releases 0, 1
    cache.create_sequence("d")
    assert cache.append("d", 12) == (0, 1, 2)
    assert cache.append("c", 1) == (3,)
    cache.check_invariants()


def test_check_invariants_detects_corruption():
    cache = make(4)
    cache.create_sequence("a")
    cache.append("a", 5)
    cache._ref[0] = 2
    with pytest.raises(AssertionError):
        cache.check_invariants()


def test_randomized_stress():
    rng = random.Random(99)
    cache = make(64)
    tokens: dict[int, int] = {}
    next_id = 0
    for step in range(5000):
        r = rng.random()
        ids = list(tokens)
        if not ids or r < 0.08:
            cache.create_sequence(next_id)
            tokens[next_id] = 0
            next_id += 1
        elif r < 0.70:
            seq = rng.choice(ids)
            n = rng.randint(0, 9)
            before = (cache.tokens(seq), cache.blocks(seq), cache.free_blocks)
            try:
                new = cache.append(seq, n)
            except KVCacheFull:
                assert (cache.tokens(seq), cache.blocks(seq), cache.free_blocks) == before
                victim = max(ids, key=tokens.get)
                cache.free_sequence(victim)
                del tokens[victim]
            else:
                tokens[seq] += n
                assert cache.free_blocks == before[2] - len(new)
                assert set(new) <= set(cache.blocks(seq))
        elif r < 0.85:
            parent = rng.choice(ids)
            cache.fork(parent, next_id)
            tokens[next_id] = tokens[parent]
            next_id += 1
        else:
            seq = rng.choice(ids)
            cache.free_sequence(seq)
            del tokens[seq]
        if step % 100 == 0:
            cache.check_invariants()
            assert set(cache.sequences()) == set(tokens)
            for seq, n in tokens.items():
                assert cache.tokens(seq) == n
                assert len(cache.blocks(seq)) == -(-n // 4)
                assert cache.read_bytes(seq) == n * SMALL.bytes_per_token
            referenced = {b for seq in tokens for b in cache.blocks(seq)}
            assert cache.used_bytes == len(referenced) * cache.block_bytes
    for seq in list(tokens):
        cache.free_sequence(seq)
    cache.check_invariants()
    assert cache.free_blocks == cache.n_blocks == 64
