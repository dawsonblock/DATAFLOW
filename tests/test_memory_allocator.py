import random
import time
from itertools import pairwise

import pytest

from basp.memory.allocator import Allocation, Allocator, OutOfMemory


def test_alignment_rounds_size_and_offset():
    a = Allocator(1024, alignment=64)
    x = a.allocate(1, tag="x")
    y = a.allocate(65)
    z = a.allocate(64)
    assert (x.offset, x.size, x.requested, x.tag) == (0, 64, 1, "x")
    assert (y.offset, y.size, y.requested) == (64, 128, 65)
    assert (z.offset, z.size) == (192, 64)
    assert a.used == 256 and a.free_bytes == 768 and a.n_allocations == 3
    assert all(al.offset % 64 == 0 for al in a.allocations())
    a.check_invariants()


def test_first_fit_takes_lowest_offset_hole():
    a = Allocator(64 * 8)
    blocks = [a.allocate(64) for _ in range(8)]
    a.free(blocks[1])  # 64-byte hole at 64
    a.free(blocks[2])  # coalesces: 128-byte hole at 64
    a.free(blocks[5])  # exact-fit 64-byte hole at 320
    got = a.allocate(64)
    assert got.offset == 64  # lowest offset wins over the exact-fit hole further up
    assert a.allocate(64).offset == 128
    assert a.allocate(64).offset == 320
    a.check_invariants()


def test_first_fit_skips_holes_that_are_too_small():
    a = Allocator(64 * 8)
    blocks = [a.allocate(64) for _ in range(8)]
    a.free(blocks[0])
    a.free(blocks[3])
    a.free(blocks[4])
    assert a.allocate(100).offset == 192
    assert a.allocate(10).offset == 0


@pytest.mark.parametrize(
    "order, expected",
    [
        ((0, 1), [(0, 128)]),  # freeing 1 merges with its left neighbour
        ((1, 0), [(0, 128)]),  # freeing 0 merges with its right neighbour
        ((0, 2, 1), [(0, 192)]),  # freeing 1 merges both neighbours
        ((0, 2), [(0, 64), (128, 64)]),  # non-adjacent holes stay separate
    ],
)
def test_coalescing(order, expected):
    a = Allocator(256)
    blocks = [a.allocate(64) for _ in range(4)]
    for i in order:
        a.free(blocks[i])
        a.check_invariants()
    assert a.free_extents() == expected
    assert a.largest_free_block == max(size for _, size in expected)


def test_free_everything_restores_single_extent():
    a = Allocator(4096)
    blocks = [a.allocate(s) for s in (10, 200, 64, 1000, 7)]
    for b in (blocks[3], blocks[0], blocks[4], blocks[1], blocks[2]):
        a.free(b)
    assert a.free_extents() == [(0, 4096)]
    assert a.used == 0 and a.n_allocations == 0 and a.fragmentation == 0.0


def test_oom_and_try_allocate():
    a = Allocator(256)
    a.allocate(200)  # reserves 256
    assert a.try_allocate(1) is None
    assert not a.can_fit(1)
    with pytest.raises(OutOfMemory):
        a.allocate(1)
    assert a.used == 256 and a.free_bytes == 0 and a.largest_free_block == 0
    a.check_invariants()


def test_oom_when_free_bytes_suffice_but_are_fragmented():
    a = Allocator(256)
    blocks = [a.allocate(64) for _ in range(4)]
    a.free(blocks[0])
    a.free(blocks[2])
    assert a.free_bytes == 128 and a.largest_free_block == 64
    assert a.can_fit(64) and not a.can_fit(65)
    assert a.try_allocate(128) is None
    assert a.fragmentation == pytest.approx(0.5)


def test_invalid_sizes_and_arguments():
    a = Allocator(256)
    for bad in (0, -5):
        with pytest.raises(ValueError):
            a.allocate(bad)
        with pytest.raises(ValueError):
            a.try_allocate(bad)
    with pytest.raises(ValueError):
        Allocator(-1)
    with pytest.raises(ValueError):
        Allocator(64, alignment=0)


def test_double_free_and_foreign_allocation():
    a, b = Allocator(256), Allocator(256)
    x = a.allocate(64)
    foreign = b.allocate(64)
    with pytest.raises(ValueError):
        a.free(foreign)  # same offset and size, but not ours
    with pytest.raises(ValueError):
        a.free(Allocation(128, 64, 64))
    a.free(x)
    with pytest.raises(ValueError):
        a.free(x)
    a.check_invariants()
    b.check_invariants()


def test_fragmentation_metric():
    a = Allocator(64 * 10)
    assert a.fragmentation == 0.0
    blocks = [a.allocate(64) for _ in range(10)]
    assert a.fragmentation == 0.0  # nothing free
    for i in (0, 2, 4, 6):
        a.free(blocks[i])
    assert a.fragmentation == pytest.approx(1 - 64 / 256)
    a.free(blocks[1])
    assert a.fragmentation == pytest.approx(1 - 192 / 320)


def test_exhaust_and_refill():
    a = Allocator(64 * 100)
    blocks = [a.allocate(64) for _ in range(100)]
    assert a.try_allocate(1) is None
    assert [b.offset for b in blocks] == [64 * i for i in range(100)]
    for b in blocks[::2]:
        a.free(b)
    refill = [a.allocate(33) for _ in range(50)]
    assert [b.offset for b in refill] == [64 * i for i in range(0, 100, 2)]
    assert a.free_bytes == 0
    a.check_invariants()


def test_capacity_not_multiple_of_alignment():
    a = Allocator(200, alignment=64)
    blocks = [a.allocate(64) for _ in range(3)]
    assert a.try_allocate(1) is None  # 8-byte tail cannot hold an aligned block
    assert a.used == 192 and a.free_bytes == 8
    a.check_invariants()
    for b in blocks:
        a.free(b)
    assert a.free_extents() == [(0, 200)]


def test_zero_capacity():
    a = Allocator(0)
    assert a.try_allocate(1) is None
    assert a.largest_free_block == 0 and a.fragmentation == 0.0
    a.check_invariants()


def test_allocations_sorted_and_tags_kept():
    a = Allocator(1024)
    blocks = {name: a.allocate(64, tag=name) for name in "abcd"}
    a.free(blocks["a"])
    blocks["e"] = a.allocate(64, tag="e")
    listed = a.allocations()
    assert [al.offset for al in listed] == sorted(al.offset for al in listed)
    assert [al.tag for al in listed] == ["e", "b", "c", "d"]


def test_check_invariants_detects_corruption():
    a = Allocator(256)
    a.allocate(64)
    a._used += 64
    with pytest.raises(AssertionError):
        a.check_invariants()
    b = Allocator(256)
    b.allocate(64)
    b._free_off[0] = 0  # free extent now overlaps the live allocation
    with pytest.raises(AssertionError):
        b.check_invariants()


def test_randomized_stress():
    rng = random.Random(1234)
    a = Allocator(1 << 20, alignment=64)
    live: list[Allocation] = []
    for step in range(6000):
        if live and (rng.random() < 0.45 or a.free_bytes < 4096):
            a.free(live.pop(rng.randrange(len(live))))
        else:
            size = rng.choice((1, 63, 64, 65, 500, 4096, 30000))
            got = a.try_allocate(size, tag=step)
            if got is None:
                assert a.largest_free_block < -(-size // 64) * 64
            else:
                assert got.requested == size and got.size >= size and got.tag == step
                live.append(got)
        if step % 200 == 0:
            a.check_invariants()
            spans = sorted((al.offset, al.offset + al.size) for al in live)
            assert all(e1 <= s2 for (_, e1), (s2, _) in pairwise(spans))
            assert a.n_allocations == len(live)
            assert a.used == sum(al.size for al in live)
    for al in live:
        a.free(al)
    a.check_invariants()
    assert a.free_extents() == [(0, 1 << 20)]


def test_performance_with_many_live_allocations():
    block = 2_654_208  # one Qwen3-30B-A3B expert at 4.5 bits per weight
    n = 10_000
    a = Allocator(block * n)
    rng = random.Random(7)
    start = time.perf_counter()
    live = [a.allocate(block) for _ in range(n)]
    for _ in range(20_000):
        i = rng.randrange(n)
        a.free(live[i])
        live[i] = a.allocate(block)
    elapsed = time.perf_counter() - start
    a.check_invariants()
    assert a.n_allocations == n
    assert elapsed < 2.0, f"allocator too slow: {elapsed:.2f}s"
