"""Replica management: adaptive expert caching and eviction.

Replicas of immutable weights may be created on cache devices (typically a GPU)
when that is beneficial.  The manager answers two questions for the scheduler:
which replica should go when room is needed, and whether an incoming replica is
worth more than the victim.  Eviction examines a small seeded random sample of
idle replicas per decision (sampled LFU/LRU), which keeps decisions O(1) even
with thousands of cached experts.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from basp.core.tensor import CompressionFormat
from basp.core.topology import Topology
from basp.dataflow.base import CachePolicy
from basp.memory.residency import Copy, CopyState, ResidencyMap
from basp.memory.tensor_registry import TensorRegistry


@dataclass
class ReplicaAction:
    """A background or speculative replication chosen by the manager."""

    tensor: int
    src: int
    dst: int
    nbytes: float
    fmt: CompressionFormat | None
    evict: list[tuple[int, int]] = field(default_factory=list)
    speculative: bool = False
    max_rate: float | None = None  # bytes/s the background budget allows on this path
    value: float = 0.0
    tag: str = "replicate"
    store_fmt: CompressionFormat | None = None  # representation of the new replica
    dst_bytes: float | None = None


class ReplicaManager:
    def __init__(
        self,
        policy: CachePolicy,
        residency: ResidencyMap,
        registry: TensorRegistry,
        topology: Topology,
        *,
        freq_fn: Callable[[int], float] | None = None,
        seed: int = 0,
    ) -> None:
        self.policy = policy
        self.residency = residency
        self.registry = registry
        self.topology = topology
        self.freq_fn = freq_fn or (lambda _tensor: 0.0)
        self._rng = random.Random(seed)
        self._cached: dict[int, list[int]] = {d: [] for d in policy.devices}
        self._pos: dict[tuple[int, int], int] = {}
        self._speculative: dict[int, list[int]] = {d: [] for d in policy.devices}
        self.inflight: set[tuple[int, int]] = set()
        self.protected: set[tuple[int, int]] = set()  # prefetched, prediction not yet resolved
        self.stats: Counter = Counter()
        for device in policy.devices:
            for copy in residency.replicas(device):
                self.track(copy.tensor, device)

    # -- bookkeeping --------------------------------------------------------------
    def track(self, tensor: int, device: int, *, speculative: bool = False) -> None:
        if device not in self._cached or (device, tensor) in self._pos:
            return
        self._pos[(device, tensor)] = len(self._cached[device])
        self._cached[device].append(tensor)
        if speculative:
            self._speculative[device].append(tensor)

    def untrack(self, tensor: int, device: int) -> None:
        pos = self._pos.pop((device, tensor), None)
        if pos is None:
            return
        items = self._cached[device]
        last = items.pop()
        if pos < len(items):
            items[pos] = last
            self._pos[(device, last)] = pos

    def cached(self, device: int) -> int:
        return len(self._cached.get(device, ()))

    # -- policy -------------------------------------------------------------------
    def value(self, tensor: int) -> float:
        """Estimated probability that the tensor is needed on one visit to its layer."""
        return self.freq_fn(tensor)

    def _score(self, copy: Copy) -> float:
        if copy.speculative:
            return -1.0
        if self.policy.kind == "lru":
            return copy.last_use
        return self.freq_fn(copy.tensor)

    def _evictable(self, tensor: int, device: int) -> Copy | None:
        if (tensor, device) in self.protected:
            return None
        copy = self.residency.copy(tensor, device)
        if (copy is None or copy.state is not CopyState.VALID or copy.pinned or copy.owner
                or copy.in_use):  # fmt: skip
            return None
        return copy

    def victims(self, device: int) -> list[Copy]:
        """Eviction candidates in eviction order (unused prefetches first)."""
        if self.policy.kind == "none" or device not in self._cached:
            return []
        seen: set[int] = set()
        found: list[Copy] = []
        spec = self._speculative[device]
        spec[:] = [t for t in spec
                   if (c := self.residency.copy(t, device)) is not None and c.speculative]  # fmt: skip
        for tensor in spec:
            copy = self._evictable(tensor, device)
            if copy is not None:
                seen.add(tensor)
                found.append(copy)
        items = self._cached[device]
        if len(items) <= self.policy.sample:
            pool = list(items)  # small cache: examine everything
        else:
            pool = [items[self._rng.randrange(len(items))] for _ in range(2 * self.policy.sample)]
        for tensor in pool:
            if tensor in seen:
                continue
            seen.add(tensor)
            copy = self._evictable(tensor, device)
            if copy is not None:
                found.append(copy)
            if len(found) >= self.policy.sample:
                break
        found.sort(key=lambda c: (self._score(c), c.tensor))
        return found

    def admits(self, value: float, victim: Copy) -> bool:
        """May a replica worth ``value`` displace ``victim``?"""
        if self.policy.kind == "none":
            return False
        if victim.speculative or self.policy.kind == "lru":
            return True
        return value > self.policy.hysteresis * self.freq_fn(victim.tensor)

    def budget_rate(self, src: int, dst: int) -> float:
        """Bytes/s available to background traffic on the path (shared by all of it)."""
        return self.policy.budget_fraction * self.topology.transfer_rate(src, dst)

    def propose(self, tensor: int, dst: int, st, cost, *, value: float, saving_s: float,
                speculative: bool, tag: str) -> ReplicaAction | None:  # fmt: skip
        """Decide whether to replicate ``tensor`` onto ``dst`` in the background.

        ``value`` is the probability of use per visit (long-run frequency for
        reactive caching, predicted next-visit probability for prefetching) and
        ``saving_s`` the time saved per use once the replica exists.
        """
        policy = self.policy
        if policy.kind == "none" or dst not in policy.devices or saving_s <= 0:
            return None
        if (tensor, dst) in self.inflight or len(self.inflight) >= policy.max_inflight:
            self.stats["throttled"] += (tensor, dst) not in self.inflight
            return None
        if st.avail(tensor, dst) is not None:
            return None
        if not self.registry.spec(tensor).policy.replicable:
            return None
        best = None
        for src, t_src in st.sources(tensor):
            fmt = st.format_at(tensor, src) or self.registry.spec(tensor).fmt
            landed = cost.landing_format(tensor, fmt, dst)
            if landed is None:
                continue
            nbytes = self.registry.nbytes(tensor, fmt)
            stored = self.registry.nbytes(tensor, landed)
            latency, bw = cost.transfer_estimate(
                src, dst, nbytes, decode_fmt=fmt if landed != fmt else None, dst_bytes=stored
            )
            t = max(t_src, st.now) + latency + bw
            if best is None or t < best[0]:
                best = (t, src, fmt, nbytes, latency + bw, landed, stored)
        if best is None:
            return None
        _, src, fmt, nbytes, transfer_s, landed, stored = best
        expected_uses = value if speculative else value * policy.horizon_visits
        if expected_uses * saving_s <= (0.0 if speculative else transfer_s):
            self.stats["rejected_cost"] += 1
            return None
        evict: list[tuple[int, int]] = []
        if not cost.reserve(dst, stored, st, evict, value, retain=True):
            self.stats["rejected_room"] += 1
            return None
        self.stats["proposed"] += 1
        self.inflight.add((tensor, dst))
        st.add_arrival(tensor, dst, st.now + transfer_s / policy.budget_fraction)
        st.formats[(tensor, dst)] = landed
        return ReplicaAction(tensor, src, dst, nbytes, fmt, evict, speculative,
                             self.budget_rate(src, dst), value, tag, landed,
                             stored if landed != fmt else None)  # fmt: skip
