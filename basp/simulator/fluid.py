"""Fluid-flow model of shared bandwidth and throughput.

Every contended quantity in the system (a memory channel group, one direction
of a link, a decoder, a MAC array) is a :class:`Resource` with a capacity in
work units per second.  A :class:`Flow` is a piece of work that needs a fixed
amount of work on one or more resources *simultaneously* and progresses at the
rate allowed by its slowest stage, exactly like a ready/valid pipeline:

    time alone = max over resources (work / capacity)          (roofline)

When several flows share resources, rates follow max-min fairness on each
flow's normalised progress rate (dominant-resource fairness, by progressive
filling).  Priority classes are strict: a lower class only receives capacity
left over by all higher classes, and a flow may carry an absolute rate cap,
which is how prefetch traffic is kept within its bandwidth budget.

Accounting is exact by construction: the work recorded on a resource is the
sum over flows of ``work * progress`` and can never exceed capacity x time.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping

from basp.simulator.engine import Simulator
from basp.simulator.event_queue import Event


class Resource:
    __slots__ = ("busy_time", "by_tag", "capacity", "load_time", "name", "served", "unit")

    def __init__(self, name: str, capacity: float, unit: str = "bytes") -> None:
        if not capacity > 0:
            raise ValueError(f"resource {name!r} needs positive capacity")
        self.name = name
        self.capacity = float(capacity)
        self.unit = unit
        self.served = 0.0  # total work done
        self.by_tag: dict[str, float] = {}
        self.busy_time = 0.0  # seconds with at least one progressing flow
        self.load_time = 0.0  # integral of utilisation (served / capacity)

    def utilisation(self, elapsed: float) -> float:
        return self.load_time / elapsed if elapsed > 0 else 0.0

    def __repr__(self) -> str:
        return f"Resource({self.name!r}, {self.capacity:g} {self.unit}/s)"


class Flow:
    __slots__ = (
        "active",
        "demands",
        "fid",
        "finished_at",
        "label",
        "max_rate",
        "on_done",
        "priority",
        "progress",
        "rate",
        "started_at",
        "tag",
    )  # fmt: skip

    def __init__(self, fid, demands, priority, max_rate, tag, label, on_done, now) -> None:
        self.fid = fid
        self.demands: tuple[tuple[Resource, float], ...] = demands
        self.priority = priority
        self.max_rate = max_rate  # cap on progress per second, or None
        self.tag = tag
        self.label = label
        self.progress = 0.0  # fraction complete in [0, 1]
        self.rate = 0.0  # progress per second
        self.on_done = on_done
        self.started_at = now
        self.finished_at: float | None = None
        self.active = True

    def work_done(self, resource: Resource) -> float:
        return sum(w for r, w in self.demands if r is resource) * self.progress


class FluidNetwork:
    def __init__(self, sim: Simulator) -> None:
        self.sim = sim
        self.resources: dict[str, Resource] = {}
        self._flows: dict[int, Flow] = {}
        self._usage: dict[Resource, float] = {}
        self._last = sim.now
        self._timer: Event | None = None
        self._flush_event: Event | None = None
        self._finishers: list[Flow] = []
        self._next_id = 0
        self.reallocations = 0

    # -- public API ---------------------------------------------------------------
    def resource(self, name: str, capacity: float, unit: str = "bytes") -> Resource:
        if name in self.resources:
            raise ValueError(f"duplicate resource {name!r}")
        res = Resource(name, capacity, unit)
        self.resources[name] = res
        return res

    def start(
        self,
        demands: Mapping[Resource, float] | Iterable[tuple[Resource, float]],
        on_done: Callable[[Flow], None] | None = None,
        *,
        priority: int = 0,
        tag: str = "",
        max_rate: float | None = None,
        label: str = "",
    ) -> Flow:
        """Start a flow; ``on_done(flow)`` fires when all of its work is complete."""
        pairs = demands.items() if isinstance(demands, Mapping) else demands
        merged: dict[Resource, float] = {}
        for res, work in pairs:
            if work < 0:
                raise ValueError("negative work")
            if work > 0:
                merged[res] = merged.get(res, 0.0) + float(work)
        if max_rate is not None and not max_rate > 0:
            raise ValueError("max_rate must be positive")
        flow = Flow(self._next_id, tuple(merged.items()), priority, max_rate, tag, label, on_done,
                    self.sim.now)  # fmt: skip
        self._next_id += 1
        unconstrained = max_rate is None and all(math.isinf(r.capacity) for r in merged)
        self._advance()
        if unconstrained:
            flow.active = False
            self._account(flow, 1.0)
            flow.progress = 1.0
            self.sim.after(0.0, self._finish, flow)
        else:
            self._flows[flow.fid] = flow
            self._mark_dirty()
        return flow

    def cancel(self, flow: Flow) -> float:
        """Stop a flow; work already done stays accounted.  Returns its progress."""
        if flow.active and flow.fid in self._flows:
            self._advance()
            del self._flows[flow.fid]
            flow.active = False
            flow.rate = 0.0
            self._mark_dirty()
        return flow.progress

    def reprioritise(self, flow: Flow, priority: int, max_rate: float | None = None) -> None:
        if flow.active and flow.fid in self._flows:
            self._advance()
            flow.priority = priority
            flow.max_rate = max_rate
            self._mark_dirty()

    def release(self, flow: Flow, resources) -> None:
        """Drop ``flow``'s remaining demand on ``resources`` (work done so far stays accounted)."""
        if flow.active and flow.fid in self._flows:
            self._advance()
            dropped = set(resources)
            flow.demands = tuple((r, w) for r, w in flow.demands if r not in dropped)
            self._mark_dirty()

    @property
    def active_flows(self) -> int:
        return len(self._flows)

    def eta(self, flow: Flow) -> float:
        """Seconds until ``flow`` completes at its current rate (inf when starved)."""
        if not flow.active:
            return 0.0
        self._advance()
        return (1.0 - flow.progress) / flow.rate if flow.rate > 0 else math.inf

    def settle(self) -> None:
        """Bring accounting up to the current simulation time."""
        self._advance()

    # -- internals ----------------------------------------------------------------
    def _account(self, flow: Flow, dprogress: float) -> None:
        tag = flow.tag
        for res, work in flow.demands:
            amount = work * dprogress
            res.served += amount
            res.by_tag[tag] = res.by_tag.get(tag, 0.0) + amount

    def _advance(self) -> None:
        now = self.sim.now
        dt = now - self._last
        if dt <= 0:
            return
        self._last = now
        for flow in self._flows.values():
            if flow.rate > 0:
                step = min(flow.rate * dt, 1.0 - flow.progress)
                if step > 0:
                    flow.progress += step
                    self._account(flow, step)
        for res, used in self._usage.items():
            if used > 0:
                res.busy_time += dt
                if not math.isinf(res.capacity):
                    res.load_time += dt * min(used / res.capacity, 1.0)

    def _mark_dirty(self) -> None:
        if self._flush_event is None:
            self._flush_event = self.sim.after(0.0, self._flush)

    def _flush(self) -> None:
        self._flush_event = None
        self._advance()
        self._reallocate()
        self.sim.cancel(self._timer)
        self._timer = None
        eta = math.inf
        for flow in self._flows.values():
            if flow.rate > 0:
                eta = min(eta, (1.0 - flow.progress) / flow.rate)
        if math.isinf(eta):
            self._finishers = []
            return
        limit = eta * (1.0 + 1e-9)
        self._finishers = [
            f for f in self._flows.values() if f.rate > 0 and (1.0 - f.progress) / f.rate <= limit
        ]
        self._timer = self.sim.after(max(eta, 0.0), self._on_timer)

    def _on_timer(self) -> None:
        self._timer = None
        self._advance()
        done = [f for f in self._finishers if f.active]
        done_ids = {f.fid for f in done}
        done.extend(f for f in self._flows.values() if f.progress >= 1.0 and f.fid not in done_ids)
        self._finishers = []
        for flow in done:
            remainder = 1.0 - flow.progress
            if remainder > 0:
                self._account(flow, remainder)
            flow.progress = 1.0
            flow.rate = 0.0
            flow.active = False
            del self._flows[flow.fid]
        self._mark_dirty()
        for flow in done:
            self._finish(flow)

    def _finish(self, flow: Flow) -> None:
        flow.finished_at = self.sim.now
        if flow.on_done is not None:
            flow.on_done(flow)

    def _reallocate(self) -> None:
        self.reallocations += 1
        usage: dict[Resource, float] = {}
        self._usage = usage
        if not self._flows:
            return
        residual: dict[Resource, float] = {}
        classes: dict[int, list[Flow]] = {}
        for flow in self._flows.values():
            classes.setdefault(flow.priority, []).append(flow)
            for res, _ in flow.demands:
                residual[res] = res.capacity
        inf = math.inf
        for priority in sorted(classes):
            solo: dict[int, float] = {}
            active: list[Flow] = []
            for flow in classes[priority]:
                s = inf if flow.max_rate is None else flow.max_rate
                for res, work in flow.demands:
                    cap = residual[res]
                    if cap != inf:
                        s = min(s, cap / work)
                if s <= 0.0 or s == inf:
                    flow.rate = 0.0
                else:
                    solo[flow.fid] = s
                    active.append(flow)
            fixed: dict[Resource, float] = {}
            while active:
                denom: dict[Resource, float] = {}
                for flow in active:
                    s = solo[flow.fid]
                    for res, work in flow.demands:
                        if residual[res] != inf:
                            denom[res] = denom.get(res, 0.0) + s * work
                level = 1.0
                for res, d in denom.items():
                    avail = residual[res] - fixed.get(res, 0.0)
                    lvl = avail / d if avail > 0.0 else 0.0
                    level = min(level, lvl)
                if level >= 1.0:
                    frozen, active = active, []
                else:
                    edge = level * (1.0 + 1e-12)
                    saturated = {
                        res for res, d in denom.items()
                        if residual[res] - fixed.get(res, 0.0) <= d * edge
                    }  # fmt: skip
                    frozen = [f for f in active if any(r in saturated for r, _ in f.demands)]
                    frozen_ids = {f.fid for f in frozen}
                    active = [f for f in active if f.fid not in frozen_ids]
                for flow in frozen:
                    rate = level * solo[flow.fid]
                    flow.rate = rate
                    for res, work in flow.demands:
                        used = rate * work
                        usage[res] = usage.get(res, 0.0) + used
                        if residual[res] != inf:
                            fixed[res] = fixed.get(res, 0.0) + used
            for res, used in fixed.items():
                residual[res] = max(residual[res] - used, 0.0)
