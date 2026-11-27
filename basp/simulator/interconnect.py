"""Interconnect model: bidirectional links, DMA queues and multi-hop transfers.

Each link direction is a fluid resource with its own bandwidth, so traffic in
opposite directions does not interfere.  A transfer occupies one DMA slot per
hop for its whole lifetime; when a hop has no free slot the transfer waits in a
FIFO (DMA queue congestion).  After the route latency the payload moves as one
flow constrained by the source memory, every hop and the destination memory.
"""

from __future__ import annotations

from collections.abc import Callable

from basp.core.topology import Hop, Topology
from basp.simulator.engine import Simulator
from basp.simulator.fluid import Flow, FluidNetwork, Resource


class TransferHandle:
    __slots__ = (
        "dst",
        "dst_bytes",
        "event",
        "extra",
        "finished",
        "flow",
        "max_rate",
        "nbytes",
        "on_done",
        "priority",
        "requested",
        "route",
        "src",
        "started",
        "state",
        "tag",
    )  # fmt: skip

    def __init__(self, src, dst, nbytes, tag, priority, max_rate, on_done, route, now):
        self.dst_bytes: float | None = None  # bytes written at the destination, if not nbytes
        self.extra: tuple[tuple[Resource, float], ...] = ()  # e.g. the destination decoder
        self.src = src
        self.dst = dst
        self.nbytes = nbytes
        self.tag = tag
        self.priority = priority
        self.max_rate = max_rate  # bytes per second cap, or None
        self.on_done = on_done
        self.route: tuple[Hop, ...] = route
        self.requested = now
        self.started: float | None = None
        self.finished: float | None = None
        self.flow: Flow | None = None
        self.event = None
        self.state = "waiting"  # waiting -> latency -> flowing -> done | cancelled

    @property
    def bytes_moved(self) -> float:
        if self.state == "done":
            return self.nbytes
        return self.nbytes * self.flow.progress if self.flow is not None else 0.0


class LinkDirectionStats:
    __slots__ = ("active", "max_active", "max_waiting", "transfers", "wait_time")

    def __init__(self) -> None:
        self.active = 0
        self.transfers = 0
        self.wait_time = 0.0
        self.max_waiting = 0
        self.max_active = 0


class Interconnect:
    def __init__(
        self,
        sim: Simulator,
        net: FluidNetwork,
        topology: Topology,
        mem: dict[int, Resource],
    ) -> None:
        self.sim = sim
        self.net = net
        self.topology = topology
        self.mem = mem
        self.link_res: dict[tuple[int, bool], Resource] = {}
        self.stats: dict[tuple[int, bool], LinkDirectionStats] = {}
        for i, link in enumerate(topology.links):
            for forward in (True, False):
                a, b = (link.a, link.b) if forward else (link.b, link.a)
                name = f"link:{topology.devices[a].name}->{topology.devices[b].name}"
                self.link_res[(i, forward)] = net.resource(name, link.bandwidth(forward))
                self.stats[(i, forward)] = LinkDirectionStats()
        self._waiting: list[TransferHandle] = []
        self._active: list[TransferHandle] = []

    def transfer(
        self,
        src: int,
        dst: int,
        nbytes: float,
        on_done: Callable[[TransferHandle], None],
        *,
        tag: str = "data",
        priority: int = 0,
        max_rate: float | None = None,
        dst_bytes: float | None = None,
        extra: tuple[tuple[Resource, float], ...] = (),
    ) -> TransferHandle:
        """Move ``nbytes`` from ``src`` to ``dst``.

        ``dst_bytes`` and ``extra`` describe transcoding on arrival: the bytes
        written into destination memory and additional simultaneous demands
        (the destination decoder), all part of the same pipelined flow.
        """
        if src == dst:
            raise ValueError("transfer needs distinct endpoints")
        route = self.topology.route(src, dst)
        handle = TransferHandle(src, dst, float(nbytes), tag, priority, max_rate, on_done, route,
                                self.sim.now)  # fmt: skip
        handle.dst_bytes = dst_bytes
        handle.extra = tuple(extra)
        self._waiting.append(handle)
        self._admit()
        return handle

    def cancel(self, handle: TransferHandle) -> float:
        """Abort a transfer; returns the bytes that had already moved."""
        if handle.state in ("done", "cancelled"):
            return handle.bytes_moved
        moved = 0.0
        if handle.state == "waiting":
            self._waiting.remove(handle)
        else:
            if handle.flow is not None:
                moved = handle.nbytes * self.net.cancel(handle.flow)
            self.sim.cancel(handle.event)
            self._release(handle)
        handle.state = "cancelled"
        handle.finished = self.sim.now
        return moved

    def promote(self, handle: TransferHandle, priority: int = 0) -> None:
        """Raise a background transfer to demand priority and lift its bandwidth budget."""
        handle.priority = priority
        handle.max_rate = None
        budget = [res for res, _ in handle.extra if res.unit == "budget"]
        handle.extra = tuple((res, w) for res, w in handle.extra if res.unit != "budget")
        if handle.flow is not None:
            self.net.release(handle.flow, budget)
            self.net.reprioritise(handle.flow, priority, None)

    def queued_by_direction(self) -> dict[tuple[int, bool], float]:
        """Demand-priority bytes still to cross each link direction (for cost estimates)."""
        totals: dict[tuple[int, bool], float] = {}
        for handle in self._inflight:
            if handle.priority != 0:
                continue
            remaining = handle.nbytes - handle.bytes_moved
            for hop in handle.route:
                key = (hop.link, hop.forward)
                totals[key] = totals.get(key, 0.0) + remaining
        return totals

    def remaining_s(self, handle: TransferHandle) -> float:
        """Uncontended seconds until ``handle`` completes at full path bandwidth."""
        if handle.state in ("done", "cancelled"):
            return 0.0
        latency = 0.0
        if handle.state == "waiting":
            latency = self.topology.path_latency(handle.src, handle.dst)
        elif handle.state == "latency" and handle.started is not None:
            total = self.topology.path_latency(handle.src, handle.dst)
            latency = max(handle.started + total - self.sim.now, 0.0)
        remaining = handle.nbytes - handle.bytes_moved
        return latency + remaining / self.topology.transfer_rate(handle.src, handle.dst)

    # -- internals ----------------------------------------------------------------
    @property
    def _inflight(self) -> list[TransferHandle]:
        return self._active + self._waiting

    def _admit(self) -> None:
        if not self._waiting:
            return
        still: list[TransferHandle] = []
        for handle in self._waiting:
            hops = [(h.link, h.forward) for h in handle.route]
            if all(
                self.stats[k].active < self.topology.links[k[0]].dma_queue_depth for k in hops
            ) and handle.state == "waiting":
                for k in hops:
                    st = self.stats[k]
                    st.active += 1
                    st.transfers += 1
                    st.max_active = max(st.max_active, st.active)
                    st.wait_time += self.sim.now - handle.requested
                handle.state = "latency"
                handle.started = self.sim.now
                self._active.append(handle)
                latency = sum(self.topology.links[k[0]].latency_s for k in hops)
                handle.event = self.sim.after(latency, self._flow, handle)
            else:
                still.append(handle)
        self._waiting = still
        waiting_per_dir: dict[tuple[int, bool], int] = {}
        for handle in still:
            for h in handle.route:
                key = (h.link, h.forward)
                waiting_per_dir[key] = waiting_per_dir.get(key, 0) + 1
        for key, n in waiting_per_dir.items():
            self.stats[key].max_waiting = max(self.stats[key].max_waiting, n)

    def _flow(self, handle: TransferHandle) -> None:
        handle.event = None
        handle.state = "flowing"
        written = handle.nbytes if handle.dst_bytes is None else handle.dst_bytes
        demands = [(self.mem[handle.src], handle.nbytes), (self.mem[handle.dst], written)]
        demands.extend((self.link_res[(h.link, h.forward)], handle.nbytes) for h in handle.route)
        demands.extend(handle.extra)
        cap = None
        if handle.max_rate is not None and handle.nbytes > 0:
            cap = handle.max_rate / handle.nbytes
        handle.flow = self.net.start(
            demands,
            lambda _f: self._done(handle),
            priority=handle.priority,
            tag=handle.tag,
            max_rate=cap,
        )

    def _done(self, handle: TransferHandle) -> None:
        handle.state = "done"
        handle.finished = self.sim.now
        self._release(handle)
        handle.on_done(handle)

    def _release(self, handle: TransferHandle) -> None:
        for h in handle.route:
            self.stats[(h.link, h.forward)].active -= 1
        if handle in self._active:
            self._active.remove(handle)
        self.sim.after(0.0, self._admit)
