"""Prefetching under a bandwidth budget.

Before a layer runs, a predictor names experts likely to be selected; the prefetcher starts
migrations for those not yet resident, at most ``budget_bytes`` per step and never by
evicting a referenced block.  A prefetch that has not completed when its expert is needed is
simply waited for (its remaining time is on the critical path); a prefetched expert that is
not used is wasted traffic, which is reported, not hidden.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .allocator import OutOfMemory
from .domains import AdmissionRefused, Block, MemoryDomain
from .migration import Migration, MigrationEngine


@dataclass
class PrefetchStats:
    issued: int = 0
    used: int = 0
    wasted_bytes: int = 0
    skipped_budget: int = 0
    skipped_memory: int = 0
    pending: dict[str, Migration] = field(default_factory=dict)


class Prefetcher:
    def __init__(self, engine: MigrationEngine, source: MemoryDomain, target: MemoryDomain, *,
                 budget_bytes: int) -> None:  # fmt: skip
        if budget_bytes < 0:
            raise ValueError("the prefetch budget must be non-negative")
        self.engine, self.source, self.target = engine, source, target
        self.budget = budget_bytes
        self.stats = PrefetchStats()

    def issue(self, predicted: Iterable[tuple[str, Block]], *, resident: set[str], now: float) -> list[Migration]:
        """Start migrations for predicted (owner, source block) pairs, within the budget."""
        spent, started = 0, []
        for owner, block in predicted:
            if owner in resident or owner in self.stats.pending:
                continue
            if spent + block.requested > self.budget:
                self.stats.skipped_budget += 1
                continue
            try:
                m = self.engine.start(self.source, block, self.target, now=now, owner=owner)
            except (OutOfMemory, AdmissionRefused):
                self.stats.skipped_memory += 1
                continue
            spent += block.requested
            self.stats.pending[owner] = m
            self.stats.issued += 1
            started.append(m)
        return started

    def claim(self, owner: str) -> Migration | None:
        """The prefetch for ``owner`` (completed by waiting if it is still in flight)."""
        m = self.stats.pending.pop(owner, None)
        if m is not None:
            self.engine.wait(m)
            self.stats.used += 1
        return m

    def discard_unused(self) -> int:
        """Forget prefetches nobody claimed; their traffic is counted as wasted."""
        wasted = 0
        for m in self.stats.pending.values():
            if m.state == "in_flight":
                self.engine.cancel(m)
            wasted += m.source.requested
        self.stats.wasted_bytes += wasted
        self.stats.pending.clear()
        return wasted
