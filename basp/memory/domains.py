"""Physical memory domains with shared accounting.

A *memory domain* is one physical pool of memory: a discrete GPU's VRAM, a host's DRAM, or
the unified memory of an Apple-silicon machine, which the CPU and the GPU share.  Every byte
allocated by any device in the domain is charged to that one pool, so a unified machine can
never count its memory twice.

Each allocation belongs to one budget category::

    M_weights + M_kv + M_activations + M_workspace + M_reserved <= M_physical

``reserved`` is working memory set aside when the domain is created (for the operating
system, the driver, or memory a runtime must always be able to get); it is never handed out.
Optional per-category limits partition the rest.  ``check_invariant`` verifies the inequality
and that the accounting matches the underlying allocator to the byte.

Blocks are reference counted: a block that is in use (by a running kernel, or as the source
of a migration in flight) has a reference and cannot be evicted; releasing the last
reference frees it.  Eviction picks only unreferenced, unpinned blocks.

**Ownership.**  Whoever keeps data for a block (a weight cache holding the arrays, a prefix
cache holding KV rows) registers ``on_reclaim`` when it allocates.  There is exactly one
reclamation path, ``free_block``, and it calls the owner *before* the capacity is released, so
no owner is ever left holding arrays or a handle for memory the domain reports as free --
whoever triggers the reclamation (the owner itself, admission control, ``evict``).
"""

from __future__ import annotations

import enum
import itertools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .allocator import Allocation, Allocator, OutOfMemory


class Category(enum.Enum):
    WEIGHTS = "weights"  # model weights, including cached experts
    KV = "kv"  # key/value cache
    ACTIVATIONS = "activations"  # inter-layer activations
    WORKSPACE = "workspace"  # kernel scratch


class EvictionUnsafe(Exception):
    """An attempt to evict or free a block that is still referenced or pinned."""


class AdmissionRefused(OutOfMemory):
    """A request that would break the domain's budget (raised instead of over-committing)."""


@dataclass(eq=False)
class Block:
    """One live allocation in a domain."""

    block_id: int
    domain: str
    category: Category
    size: int  # bytes charged (aligned)
    requested: int
    owner: str  # what the bytes hold, e.g. "layer3.expert17"
    device: str | None = None  # the device that asked for it (accounting only: the pool is shared)
    pinned: bool = False
    refs: int = 0
    last_use: int = 0
    allocation: Allocation | None = field(default=None, repr=False)
    #: the owner's invalidation hook, called by ``free_block`` before the capacity is released
    on_reclaim: Callable[[Block], None] | None = field(default=None, repr=False)

    @property
    def live(self) -> bool:
        return self.allocation is not None

    @property
    def evictable(self) -> bool:
        return self.refs == 0 and not self.pinned


class MemoryDomain:
    """One physical pool, shared by the devices listed in ``devices``."""

    def __init__(self, name: str, capacity: int, *, devices: Iterable[str] = (), reserved: int = 0,
                 limits: Mapping[Category, int] | None = None, alignment: int = 256) -> None:  # fmt: skip
        if capacity <= 0:
            raise ValueError(f"domain {name!r}: capacity must be positive")
        if not 0 <= reserved <= capacity:
            raise ValueError(f"domain {name!r}: reserved memory must be within the capacity")
        limits = dict(limits or {})
        if any(v < 0 for v in limits.values()):
            raise ValueError(f"domain {name!r}: category limits must be non-negative")
        self.name = name
        self.capacity = int(capacity)
        self.reserved = int(reserved)
        self.devices = tuple(devices)
        self.limits = limits
        self._alloc = Allocator(self.capacity - self.reserved, alignment)
        self._blocks: dict[int, Block] = {}
        self._used = dict.fromkeys(Category, 0)
        self._ids = itertools.count()
        self._clock = itertools.count(1)
        self.peak = 0

    # -- accounting ------------------------------------------------------------------------
    @property
    def usable(self) -> int:
        return self.capacity - self.reserved

    @property
    def used(self) -> int:
        return sum(self._used.values())

    @property
    def free(self) -> int:
        return self.usable - self.used

    def used_by(self, category: Category) -> int:
        return self._used[category]

    def headroom(self, category: Category) -> int:
        """Bytes ``category`` could still get, ignoring fragmentation."""
        limit = self.limits.get(category)
        room = self.free
        return room if limit is None else min(room, limit - self._used[category])

    @property
    def pressure(self) -> float:
        """Fraction of the usable pool in use (0 to 1)."""
        return self.used / self.usable if self.usable else 1.0

    @property
    def fragmentation(self) -> float:
        """1 - largest free block / free bytes (0: one contiguous free region)."""
        return self._alloc.fragmentation

    def blocks(self, category: Category | None = None) -> list[Block]:
        return [b for b in self._blocks.values() if category is None or b.category is category]

    def budget(self) -> dict[str, int]:
        out = {c.value: self._used[c] for c in Category}
        out.update(reserved=self.reserved, physical=self.capacity, free=self.free)
        return out

    # -- allocation ------------------------------------------------------------------------
    def _rounded(self, size: int) -> int:
        a = self._alloc.alignment
        return -(-int(size) // a) * a

    def can_allocate(self, size: int, category: Category) -> bool:
        need = self._rounded(size)
        return need <= self.headroom(category) and self._alloc.can_fit(size)

    def allocate(self, size: int, category: Category, owner: str, *, device: str | None = None,
                 pinned: bool = False, refs: int = 0,
                 on_reclaim: Callable[[Block], None] | None = None) -> Block:  # fmt: skip
        """Allocate or refuse; never over-commit the pool or a category's limit.  ``on_reclaim``
        is the owner's hook: it must drop whatever the owner keeps for the block."""
        if size <= 0:
            raise ValueError("allocation size must be positive")
        if device is not None and self.devices and device not in self.devices:
            raise ValueError(f"device {device!r} does not use memory domain {self.name!r}")
        need = self._rounded(size)
        limit = self.limits.get(category)
        if limit is not None and self._used[category] + need > limit:
            raise AdmissionRefused(f"{self.name}: {category.value} limit {limit} bytes would be exceeded "
                                   f"({self._used[category]} used, {need} requested)")  # fmt: skip
        allocation = self._alloc.try_allocate(size, owner)
        if allocation is None:
            raise OutOfMemory(f"{self.name}: cannot allocate {size} bytes for {owner} ({self.free} free, "
                              f"largest block {self._alloc.largest_free_block})")  # fmt: skip
        block = Block(next(self._ids), self.name, category, allocation.size, int(size), owner, device, pinned,
                      refs, next(self._clock), allocation, on_reclaim)  # fmt: skip
        self._blocks[block.block_id] = block
        self._used[category] += block.size
        self.peak = max(self.peak, self.used)
        return block

    def retain(self, block: Block) -> Block:
        self._live(block)
        block.refs += 1
        block.last_use = next(self._clock)
        return block

    def release(self, block: Block, *, free_when_unused: bool = False) -> None:
        """Drop one reference; optionally free the block once nothing references it."""
        self._live(block)
        if block.refs <= 0:
            raise ValueError(f"{block.owner}: released more often than retained")
        block.refs -= 1
        if free_when_unused and block.evictable:
            self.free_block(block)

    def touch(self, block: Block) -> None:
        self._live(block)
        block.last_use = next(self._clock)

    def free_block(self, block: Block) -> None:
        """Free a block; refused while it is referenced or pinned.  The only reclamation path:
        the owner's ``on_reclaim`` runs first, so its entry is gone before the capacity is free."""
        self._live(block)
        if block.refs:
            raise EvictionUnsafe(f"{block.owner}: still referenced ({block.refs}); cannot free")
        if block.pinned:
            raise EvictionUnsafe(f"{block.owner}: pinned; unpin before freeing")
        assert block.allocation is not None
        if block.on_reclaim is not None:
            block.on_reclaim(block)  # may raise: then nothing was freed
        self._alloc.free(block.allocation)
        self._used[block.category] -= block.size
        del self._blocks[block.block_id]
        block.allocation = None

    def evict(self, nbytes: int, *, category: Category | None = None,
              order: Callable[[Block], Any] | None = None) -> list[Block]:  # fmt: skip
        """Free unreferenced, unpinned blocks (least recently used first, or by ``order``) until
        ``nbytes`` are free.  Returns what was evicted; frees nothing if the target is
        unreachable (an eviction that cannot help is not started)."""
        candidates = sorted((b for b in self.blocks(category) if b.evictable),
                            key=order or (lambda b: (b.last_use, b.block_id)))  # fmt: skip
        if self.free >= nbytes:
            return []
        reachable = self.free + sum(b.size for b in candidates)
        if reachable < nbytes:
            return []
        out = []
        for b in candidates:
            if self.free >= nbytes:
                break
            self.free_block(b)
            out.append(b)
        return out

    def _live(self, block: Block) -> None:
        if self._blocks.get(block.block_id) is not block:
            raise ValueError(f"{block.owner}: not a live block of domain {self.name!r}")

    # -- invariant ---------------------------------------------------------------------------
    def check_invariant(self) -> None:
        """M_weights + M_kv + M_act + M_workspace + M_reserved <= M_physical, accounting exact,
        every limit respected, no negative reference counts."""
        self._alloc.check_invariants()
        assert self.used + self.reserved <= self.capacity, "budget exceeds physical memory"
        assert self.used == self._alloc.used, "category accounting differs from the allocator"
        for category in Category:
            charged = sum(b.size for b in self._blocks.values() if b.category is category)
            assert charged == self._used[category], f"{category.value} accounting out of sync"
            limit = self.limits.get(category)
            assert limit is None or charged <= limit, f"{category.value} exceeds its limit"
        assert all(b.refs >= 0 for b in self._blocks.values()), "negative reference count"


class MemoryMap:
    """Devices to memory domains: devices that share a domain share one pool."""

    def __init__(self, domains: Iterable[MemoryDomain]) -> None:
        self.domains = {d.name: d for d in domains}
        self._of: dict[str, MemoryDomain] = {}
        for domain in self.domains.values():
            for device in domain.devices:
                if device in self._of:
                    raise ValueError(f"device {device!r} is in two memory domains")
                self._of[device] = domain

    def domain_of(self, device: str) -> MemoryDomain:
        try:
            return self._of[device]
        except KeyError:
            raise KeyError(f"device {device!r} has no memory domain") from None

    def shared(self, a: str, b: str) -> bool:
        """Whether two devices read the same physical memory (no copy needed between them)."""
        return self.domain_of(a) is self.domain_of(b)

    @property
    def physical_bytes(self) -> int:
        return sum(d.capacity for d in self.domains.values())

    def check_invariant(self) -> None:
        for domain in self.domains.values():
            domain.check_invariant()

    @classmethod
    def from_topology(cls, topology, *, reserved_fraction: float = 0.0) -> MemoryMap:
        """One domain per declared memory domain of a simulator ``Topology`` (devices without
        one get their own).  ``reserved_fraction`` of each pool is set aside as working memory."""
        groups: dict[str, list] = {}
        for dev in topology.devices.values():
            if dev.mem_capacity <= 0:
                continue
            groups.setdefault(dev.memory_domain or f"{dev.name}.mem", []).append(dev)
        domains = []
        for name, devs in sorted(groups.items()):
            capacity = devs[0].mem_capacity
            domains.append(MemoryDomain(name, capacity, devices=[d.name for d in devs],
                                        reserved=int(capacity * reserved_fraction)))  # fmt: skip
        return cls(domains)
