"""First-fit region allocator for one device's byte range.

The device memory is the half-open range ``[0, capacity)``. Free space is kept as a list of
disjoint, coalesced extents sorted by offset (two parallel lists, searched with ``bisect``);
live allocations are keyed by offset, so ``free`` is a dictionary lookup plus one bisect.
Allocation is lowest-offset first fit, which makes placement fully deterministic. Every
reserved size is rounded up to ``alignment``, so every offset is aligned as well.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from itertools import pairwise


class OutOfMemory(Exception):
    """No free extent is large enough for the request."""


@dataclass(frozen=True)
class Allocation:
    offset: int
    size: int  # aligned, reserved size
    requested: int
    tag: object = None

    @property
    def end(self) -> int:
        return self.offset + self.size


class Allocator:
    def __init__(self, capacity: int, alignment: int = 64) -> None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        if alignment <= 0:
            raise ValueError("alignment must be > 0")
        self._capacity = int(capacity)
        self._alignment = int(alignment)
        self._free_off: list[int] = [0] if capacity else []
        self._free_size: list[int] = [self._capacity] if capacity else []
        self._live: dict[int, Allocation] = {}
        self._used = 0

    # -- properties ---------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def alignment(self) -> int:
        return self._alignment

    @property
    def used(self) -> int:
        return self._used

    @property
    def free_bytes(self) -> int:
        return self._capacity - self._used

    @property
    def largest_free_block(self) -> int:
        return max(self._free_size, default=0)

    @property
    def n_allocations(self) -> int:
        return len(self._live)

    @property
    def fragmentation(self) -> float:
        free = self.free_bytes
        return 0.0 if free == 0 else 1.0 - self.largest_free_block / free

    # -- allocation ---------------------------------------------------------------------

    def _reserve(self, size: int) -> int:
        if size <= 0:
            raise ValueError("allocation size must be > 0")
        return -(-int(size) // self._alignment) * self._alignment

    def can_fit(self, size: int) -> bool:
        return self.largest_free_block >= self._reserve(size)

    def try_allocate(self, size: int, tag: object = None) -> Allocation | None:
        need = self._reserve(size)
        sizes = self._free_size
        for i, extent in enumerate(sizes):
            if extent >= need:
                break
        else:
            return None
        offset = self._free_off[i]
        if extent == need:
            del self._free_off[i]
            del sizes[i]
        else:
            self._free_off[i] = offset + need
            sizes[i] = extent - need
        alloc = Allocation(offset, need, int(size), tag)
        self._live[offset] = alloc
        self._used += need
        return alloc

    def allocate(self, size: int, tag: object = None) -> Allocation:
        alloc = self.try_allocate(size, tag)
        if alloc is None:
            raise OutOfMemory(
                f"cannot allocate {size} bytes: {self.free_bytes} free, "
                f"largest block {self.largest_free_block}"
            )
        return alloc

    def free(self, alloc: Allocation) -> None:
        if self._live.get(alloc.offset) is not alloc:
            raise ValueError(f"allocation at offset {alloc.offset} is not live in this allocator")
        del self._live[alloc.offset]
        self._used -= alloc.size
        offs, sizes = self._free_off, self._free_size
        offset, size = alloc.offset, alloc.size
        i = bisect_left(offs, offset)
        merge_left = i > 0 and offs[i - 1] + sizes[i - 1] == offset
        merge_right = i < len(offs) and offset + size == offs[i]
        if merge_left and merge_right:
            sizes[i - 1] += size + sizes[i]
            del offs[i]
            del sizes[i]
        elif merge_left:
            sizes[i - 1] += size
        elif merge_right:
            offs[i] = offset
            sizes[i] += size
        else:
            offs.insert(i, offset)
            sizes.insert(i, size)

    # -- introspection ------------------------------------------------------------------

    def allocations(self) -> list[Allocation]:
        return [self._live[o] for o in sorted(self._live)]

    def free_extents(self) -> list[tuple[int, int]]:
        """Free ``(offset, size)`` extents sorted by offset."""
        return list(zip(self._free_off, self._free_size))

    def check_invariants(self) -> None:
        live = self.allocations()
        for key, a in self._live.items():
            assert key == a.offset, "live allocation keyed by the wrong offset"
        for a in live:
            assert a.size > 0 and a.requested > 0, "empty allocation"
            assert a.size % self._alignment == 0 and a.offset % self._alignment == 0, "misaligned"
            assert a.requested <= a.size, "reserved size smaller than request"
            assert 0 <= a.offset and a.end <= self._capacity, "allocation leaves the range"
        assert len(self._free_off) == len(self._free_size), "free list arrays out of sync"
        free = self.free_extents()
        for (o1, s1), (o2, _) in pairwise(free):
            assert o1 + s1 < o2, "free list not sorted and coalesced"
        for o, s in free:
            assert s > 0 and 0 <= o and o + s <= self._capacity, "free extent leaves the range"
        cursor = 0
        for o, s in sorted([(a.offset, a.size) for a in live] + free):
            assert o == cursor, "allocations and free extents overlap or leave a gap"
            cursor = o + s
        assert cursor == self._capacity, "extents do not cover the whole range"
        assert self._used == sum(a.size for a in live), "used counter out of sync"
        assert self._used + sum(s for _, s in free) == self._capacity, "used + free != capacity"
