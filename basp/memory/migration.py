"""Asynchronous migration between memory domains, safe against eviction.

A migration copies a block from one domain to another over a link of known bandwidth.  It
proceeds in three steps:

1. ``start``: the destination is allocated (or the migration is refused) and the source block
   gains a reference, so it cannot be evicted while it is being read;
2. the copy is in flight until its completion time;
3. ``poll`` (or ``wait``) completes it: the destination becomes usable and the source
   reference is dropped (and, for a *move*, the source is freed).

A migration can be cancelled before completion: the destination is freed and the source
released.  A block that is the destination of an unfinished migration holds a reference too,
so it cannot be read as if it were complete or evicted half-written.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .domains import Block, Category, MemoryDomain


@dataclass
class Migration:
    migration_id: int
    source: Block
    destination: Block
    start_s: float
    done_s: float
    move: bool
    state: str = "in_flight"  # in_flight | complete | cancelled


class MigrationEngine:
    def __init__(self, *, bandwidth_bytes_per_s: float, latency_s: float = 0.0) -> None:
        if bandwidth_bytes_per_s <= 0 or latency_s < 0:
            raise ValueError("bandwidth must be positive and latency non-negative")
        self.bandwidth = float(bandwidth_bytes_per_s)
        self.latency = float(latency_s)
        self.in_flight: dict[int, Migration] = {}
        self.link_free_at = 0.0  # one copy at a time per engine (one link direction)
        self.bytes_moved = 0
        self._ids = itertools.count()

    def start(self, source_domain: MemoryDomain, source: Block, target: MemoryDomain, *, now: float,
              owner: str | None = None, category: Category | None = None, move: bool = False) -> Migration:  # fmt: skip
        """Begin copying ``source`` into ``target``; raises if the destination cannot be allocated."""
        if source_domain is target:
            raise ValueError("source and destination are the same memory domain: nothing to copy")
        destination = target.allocate(source.requested, category or source.category, owner or source.owner, refs=1)
        source_domain.retain(source)
        begin = max(now, self.link_free_at)
        done = begin + self.latency + source.requested / self.bandwidth
        self.link_free_at = done
        m = Migration(next(self._ids), source, destination, begin, done, move)
        m._domains = (source_domain, target)  # type: ignore[attr-defined]
        self.in_flight[m.migration_id] = m
        return m

    def poll(self, now: float) -> list[Migration]:
        """Complete every migration finished by ``now``."""
        done = [m for m in self.in_flight.values() if m.done_s <= now]
        for m in sorted(done, key=lambda m: m.done_s):
            self._finish(m)
        return done

    def wait(self, migration: Migration) -> float:
        """Complete one migration; returns its completion time."""
        if migration.state == "in_flight":
            self._finish(migration)
        return migration.done_s

    def cancel(self, migration: Migration) -> None:
        if migration.state != "in_flight":
            raise ValueError(f"migration {migration.migration_id} is {migration.state}")
        src_domain, dst_domain = migration._domains  # type: ignore[attr-defined]
        dst_domain.release(migration.destination, free_when_unused=True)
        src_domain.release(migration.source)
        migration.state = "cancelled"
        del self.in_flight[migration.migration_id]

    def _finish(self, m: Migration) -> None:
        src_domain, dst_domain = m._domains  # type: ignore[attr-defined]
        dst_domain.release(m.destination)  # now complete: unreferenced, evictable when unused
        src_domain.release(m.source)
        if m.move and m.source.evictable:
            src_domain.free_block(m.source)
        m.state = "complete"
        self.bytes_moved += m.source.requested
        del self.in_flight[m.migration_id]
