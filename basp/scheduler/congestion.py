"""Planning state: estimated resource availability and tentative residency.

A :class:`PlanState` is a cheap, forkable snapshot of everything a cost estimate
depends on: when each device's kernel queue drains, when each link direction
is occupied by bytes already committed to it, which tensor copies exist or
are on their way, and how much replica memory is free.  The scheduler forks it
to evaluate alternatives and commits the fork of the winning assignment.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable

from basp.memory.residency import Copy, CopyState, ResidencyMap


def earliest_slot(busy: tuple[tuple[float, float], ...], ready: float, duration: float) -> tuple[float, float]:
    """(start, end) of ``duration`` seconds of link time beginning at or after ``ready``.

    The work starts at the first idle instant and is suspended across later
    reservations it runs into (on the real link the two share bandwidth; the
    last byte arrives at the same time either way), so a tiny transfer booked
    in the future never blocks a long one from using the idle time before it.
    """
    start = ready
    for s, e in busy:
        if s <= start < e:
            start = e
    t, remaining = start, duration
    for s, e in busy:
        if e <= t:
            continue
        if t + remaining <= s:
            break
        remaining -= max(s - t, 0.0)
        t = e
    return start, t + remaining


def reserve_slot(busy: tuple[tuple[float, float], ...], start: float, end: float):
    """Busy intervals with [start, end) added; touching intervals are merged."""
    merged: list[tuple[float, float]] = []
    placed = False
    for s, e in busy:
        if e < start or placed:
            merged.append((s, e))
        elif s > end:
            merged.append((start, end))
            merged.append((s, e))
            placed = True
        else:
            start, end = min(s, start), max(e, end)
    if not placed:
        merged.append((start, end))
    return tuple(merged)


class PlanState:
    __slots__ = (
        "accum",
        "accum_of",
        "arrivals",
        "device_free",
        "formats",
        "free",
        "gone",
        "link_busy",
        "link_bytes",
        "locked",
        "now",
        "pending_eta",
        "residency",
        "sram_actual",
        "sram_tentative",
        "victim_idx",
        "victims",
        "victims_fn",
        "where",
    )  # fmt: skip

    def __init__(
        self,
        now: float,
        residency: ResidencyMap,
        *,
        device_free: dict[int, float] | None = None,
        link_free: dict[tuple[int, bool], float] | None = None,
        pending_eta: dict[tuple[int, int], float] | None = None,
        sram_actual: dict[int, object] | None = None,
        victims_fn: Callable[[int], list[Copy]] | None = None,
    ) -> None:
        self.now = now
        self.residency = residency
        self.device_free = dict(device_free or {})
        # Per link direction: sorted busy intervals.  Reservations fill idle gaps, so a
        # transfer booked for later does not block an earlier one that fits before it.
        self.link_busy: dict[tuple[int, bool], tuple[tuple[float, float], ...]] = {
            key: ((now, until),) for key, until in (link_free or {}).items() if until > now
        }
        self.arrivals: dict[tuple[Hashable, int], float] = {}
        self.where: dict[Hashable, tuple[int, ...]] = {}
        self.gone: set[tuple[int, int]] = set()
        self.free: dict[int, float] = {}
        self.victims: dict[int, list[Copy]] = {}
        self.victim_idx: dict[int, int] = {}
        self.victims_fn = victims_fn
        self.sram_tentative: dict[int, frozenset] = {}
        self.sram_actual = sram_actual or {}
        self.accum: dict[Hashable, tuple[float, tuple[int, ...], frozenset, float]] = {}
        self.accum_of: dict[int, tuple[Hashable, ...]] = {}  # output -> partial sums holding it
        self.pending_eta = pending_eta or {}
        self.locked: frozenset[tuple[int, int]] = frozenset()
        self.link_bytes = 0.0
        self.formats: dict[tuple[int, int], object] = {}  # formats of tentative copies

    def fork(self) -> PlanState:
        other = PlanState.__new__(PlanState)
        other.now = self.now
        other.residency = self.residency
        other.device_free = self.device_free.copy()
        other.link_busy = self.link_busy.copy()
        other.arrivals = self.arrivals.copy()
        other.where = self.where.copy()
        other.gone = self.gone.copy()
        other.free = self.free.copy()
        other.victims = self.victims  # shared, filled lazily, never mutated in place
        other.victim_idx = self.victim_idx.copy()
        other.victims_fn = self.victims_fn
        other.sram_tentative = self.sram_tentative.copy()
        other.sram_actual = self.sram_actual
        other.accum = self.accum.copy()
        other.accum_of = self.accum_of.copy()
        other.pending_eta = self.pending_eta
        other.locked = self.locked
        other.link_bytes = self.link_bytes
        other.formats = self.formats.copy()
        return other

    # -- links --------------------------------------------------------------------
    def book(self, hops, ready: float, duration: float) -> tuple[float, float]:
        """Reserve ``duration`` seconds of link time on every hop; returns (start, end)."""
        if duration <= 0 or not hops:
            return ready, ready + max(duration, 0.0)
        keys = [(hop.link, hop.forward) for hop in hops]
        start = ready
        for key in keys:
            start = max(start, earliest_slot(self.link_busy.get(key, ()), start, duration)[0])
        end = max(earliest_slot(self.link_busy.get(key, ()), start, duration)[1] for key in keys)
        for key in keys:
            self.link_busy[key] = reserve_slot(self.link_busy.get(key, ()), start, end)
        return start, end

    # -- residency view -----------------------------------------------------------
    def avail(self, key: Hashable, device: int) -> float | None:
        """Time from which ``key`` is usable on ``device``; ``None`` if it will not be there."""
        t = self.arrivals.get((key, device))
        if t is not None:
            return t
        if not isinstance(key, int) or (key, device) in self.gone:
            return None
        copy = self.residency.copy(key, device)
        if copy is None:
            return None
        if copy.state is CopyState.VALID:
            return self.now if self.residency.valid(key, device) else None
        if copy.state is CopyState.PENDING:
            return max(self.now, self.pending_eta.get((key, device), self.now))
        return None

    def sources(self, key: Hashable) -> list[tuple[int, float]]:
        """Devices that hold (or will hold) ``key`` with their availability times."""
        found: dict[int, float] = {}
        if isinstance(key, int):
            for copy in self.residency.holders(key):
                if (key, copy.device) not in self.gone:
                    found[copy.device] = self.now
        for device in self.where.get(key, ()):
            found[device] = self.arrivals[(key, device)]
        return sorted(found.items())

    def add_arrival(self, key: Hashable, device: int, time: float) -> None:
        if (key, device) not in self.arrivals:
            self.where[key] = self.where.get(key, ()) + (device,)
            self.arrivals[(key, device)] = time
        else:
            self.arrivals[(key, device)] = max(self.arrivals[(key, device)], time)

    def remove_arrival(self, key: Hashable, device: int) -> None:
        """Forget a tentative copy (a transient load that is freed after its kernel)."""
        if self.arrivals.pop((key, device), None) is not None:
            self.where[key] = tuple(d for d in self.where.get(key, ()) if d != device)

    def format_at(self, tensor: int, device: int):
        """Stored format of the (existing or tentative) copy of ``tensor`` on ``device``."""
        fmt = self.formats.get((tensor, device))
        if fmt is not None:
            return fmt
        copy = self.residency.copy(tensor, device)
        return copy.fmt if copy is not None else None

    # -- capacity -----------------------------------------------------------------
    def free_bytes(self, device: int) -> float:
        if device not in self.free:
            self.free[device] = self.residency.free_bytes(device)
        return self.free[device]

    def victim_list(self, device: int) -> list[Copy]:
        if device not in self.victims:
            self.victims[device] = self.victims_fn(device) if self.victims_fn else []
        return self.victims[device]

    # -- SRAM tile cache view -----------------------------------------------------
    def sram_has(self, device: int, tensors) -> bool:
        actual = self.sram_actual.get(device)
        tentative = self.sram_tentative.get(device, frozenset())
        return all(t in tentative or (actual is not None and t in actual) for t in tensors)

    def sram_add(self, device: int, tensors) -> None:
        self.sram_tentative[device] = self.sram_tentative.get(device, frozenset()) | set(tensors)
