"""A memory hierarchy: ordered tiers (fast and small to slow and large) connected by links.

Each tier is a ``MemoryDomain``; the hierarchy answers where a tensor of a given size can be
placed (the fastest tier with room) and what it costs to bring it up a tier.  It is the
memory-side view the runtime and the expert cache share; placement *policy* (which tensors
deserve the fast tier) lives in ``expert_cache`` and the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass

from .domains import Category, MemoryDomain


@dataclass(frozen=True)
class Tier:
    domain: MemoryDomain
    read_bandwidth: float  # bytes/s the tier's compute device reads at
    link_to_next_bandwidth: float | None = None  # bytes/s to the next (slower) tier; None for the last


class MemoryHierarchy:
    def __init__(self, tiers: list[Tier]) -> None:
        if not tiers:
            raise ValueError("a hierarchy needs at least one tier")
        if any(t.read_bandwidth <= 0 for t in tiers):
            raise ValueError("tier bandwidths must be positive")
        for upper in tiers[:-1]:
            if not upper.link_to_next_bandwidth or upper.link_to_next_bandwidth <= 0:
                raise ValueError(f"tier {upper.domain.name!r} needs a link to the next tier")
        self.tiers = tiers

    def fastest_fit(self, size: int, category: Category = Category.WEIGHTS) -> int | None:
        """Index of the fastest tier that can take ``size`` bytes now, or None."""
        return next((i for i, t in enumerate(self.tiers) if t.domain.can_allocate(size, category)), None)

    def promotion_time(self, size: int, from_tier: int, to_tier: int) -> float:
        """Seconds to copy ``size`` bytes up from ``from_tier`` to ``to_tier`` (to < from)."""
        if not 0 <= to_tier <= from_tier < len(self.tiers):
            raise ValueError("promotion goes from a slower tier to a faster one")
        return sum(size / self.tiers[i].link_to_next_bandwidth for i in range(to_tier, from_tier))

    def read_time(self, size: int, tier: int) -> float:
        return size / self.tiers[tier].read_bandwidth
