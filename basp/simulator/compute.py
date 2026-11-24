"""Kernel descriptions, roofline timing and per-device kernel execution.

A :class:`Kernel` is a sequence of :class:`Phase` objects.  Each phase is a set
of simultaneous resource demands, so its uncontended duration is the roofline
``max(work / capacity)`` -- for a fused streaming kernel that is

    T_kernel = max(W / B, C / P, W / D)

A staged kernel (decode into a buffer, then compute) is simply two phases.  The
same objects are evaluated analytically by the scheduler's cost model and
executed with contention by the simulator, so both views agree by construction
whenever resources are uncontended.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass

from basp.core.topology import DeviceSpec, Topology
from basp.simulator.dram import TileCache
from basp.simulator.engine import Simulator
from basp.simulator.fluid import FluidNetwork, Resource

#: ("mem", device) | ("compute", device) | ("decode", device, format) | ("link", index, forward)
ResKey = tuple


def mem_key(device: int) -> ResKey:
    return ("mem", device)


def compute_key(device: int) -> ResKey:
    return ("compute", device)


def decode_key(device: int, fmt) -> ResKey:
    return ("decode", device, int(fmt))


def link_key(link: int, forward: bool) -> ResKey:
    return ("link", link, forward)


@dataclass(frozen=True)
class Phase:
    demands: tuple[tuple[ResKey, float], ...]
    delay_s: float = 0.0  # pure latency before the phase starts (e.g. link latency)


@dataclass(frozen=True)
class Kernel:
    """Executable kernel descriptor referenced by an EXECUTE/REDUCE command."""

    device: int
    phases: tuple[Phase, ...]
    macs: float = 0.0
    startup_s: float = 0.0
    #: (tensor, bytes) pairs kept in the device SRAM tile cache (row-stationary mode)
    sram_tensors: tuple[tuple[Hashable, int], ...] = ()
    #: phases used instead of ``phases`` when every SRAM tensor is already cached
    sram_hit_phases: tuple[Phase, ...] | None = None


def capacity_of(topology: Topology, key: ResKey) -> float:
    kind = key[0]
    if kind == "mem":
        return topology.devices[key[1]].mem_bandwidth
    if kind == "compute":
        return topology.devices[key[1]].compute_rate
    if kind == "decode":
        return topology.devices[key[1]].caps.decode_rate.get(key[2], math.inf)
    if kind == "link":
        return topology.links[key[1]].bandwidth(key[2])
    raise KeyError(key)


def phase_time(topology: Topology, phase: Phase) -> float:
    longest = 0.0
    for key, work in phase.demands:
        if work > 0:
            longest = max(longest, work / capacity_of(topology, key))
    return phase.delay_s + longest


def kernel_time(topology: Topology, kernel: Kernel, *, sram_hit: bool = False) -> float:
    """Uncontended duration of ``kernel`` (the roofline estimate)."""
    phases = kernel.sram_hit_phases if sram_hit and kernel.sram_hit_phases is not None else kernel.phases
    return kernel.startup_s + sum(phase_time(topology, p) for p in phases)


class KernelJob:
    __slots__ = (
        "cancelled",
        "event",
        "finished",
        "flow",
        "ideal_s",
        "kernel",
        "on_done",
        "on_start",
        "sram_hit",
        "started",
        "submitted",
        "tag",
    )  # fmt: skip

    def __init__(self, kernel, on_done, on_start, tag, now, ideal_s):
        self.kernel = kernel
        self.on_done = on_done
        self.on_start = on_start
        self.tag = tag
        self.submitted = now
        self.started: float | None = None
        self.finished: float | None = None
        self.ideal_s = ideal_s
        self.flow = None
        self.event = None
        self.cancelled = False
        self.sram_hit = False


class DeviceExecutor:
    """FIFO kernel queue of one device; one kernel runs at a time."""

    def __init__(
        self,
        sim: Simulator,
        net: FluidNetwork,
        topology: Topology,
        device: DeviceSpec,
        resolve: Callable[[ResKey], Resource | None],
    ) -> None:
        self.sim = sim
        self.net = net
        self.topology = topology
        self.device = device
        self._resolve = resolve
        self.tile_cache = TileCache(device.sram_bytes)
        self._queue: deque[KernelJob] = deque()
        self._running: KernelJob | None = None
        self.kernels_run = 0
        self.busy_time = 0.0
        self.wait_time = 0.0
        self.max_queue = 0

    def submit(self, kernel: Kernel, on_done, *, on_start=None, tag: str = "kernel") -> KernelJob:
        """Queue a kernel.  ``on_start()`` may return ``False`` to abort the job."""
        if kernel.device != self.device.device_id:
            raise ValueError("kernel submitted to the wrong device")
        job = KernelJob(kernel, on_done, on_start, tag, self.sim.now,
                        kernel_time(self.topology, kernel))  # fmt: skip
        self._queue.append(job)
        self.max_queue = max(self.max_queue, len(self._queue))
        if self._running is None:
            self._next()
        return job

    def cancel(self, job: KernelJob) -> None:
        if job.finished is not None or job.cancelled:
            return
        job.cancelled = True
        if job is self._running:
            if job.flow is not None:
                self.net.cancel(job.flow)
            self.sim.cancel(job.event)
            self._end(job)
        else:
            self._queue.remove(job)

    def backlog_s(self) -> float:
        """Estimated seconds until every queued kernel has finished."""
        total = sum(job.ideal_s for job in self._queue)
        job = self._running
        if job is not None and job.started is not None:
            total += max(job.started + job.ideal_s - self.sim.now, 0.0)
        return total

    @property
    def queue_length(self) -> int:
        return len(self._queue) + (self._running is not None)

    # -- internals ----------------------------------------------------------------
    def _next(self) -> None:
        while self._queue and self._running is None:
            job = self._queue.popleft()
            job.started = self.sim.now
            self.wait_time += job.started - job.submitted
            self._running = job
            if job.on_start is not None and job.on_start() is False:
                job.cancelled = True
                self._end(job)
                continue
            kernel = job.kernel
            ids = [t for t, _ in kernel.sram_tensors]
            job.sram_hit = bool(ids) and self.tile_cache.contains_all(ids)
            if ids:
                if job.sram_hit:
                    self.tile_cache.hits += 1
                else:
                    self.tile_cache.misses += 1
            phases = kernel.phases
            if job.sram_hit and kernel.sram_hit_phases is not None:
                phases = kernel.sram_hit_phases
            job.event = self.sim.after(kernel.startup_s, self._run_phase, job, phases, 0)

    def _run_phase(self, job: KernelJob, phases: tuple[Phase, ...], index: int) -> None:
        job.event = None
        if index == len(phases):
            self._finish(job)
            return
        phase = phases[index]
        if phase.delay_s > 0:
            job.event = self.sim.after(phase.delay_s, self._start_flow, job, phases, index)
        else:
            self._start_flow(job, phases, index)

    def _start_flow(self, job: KernelJob, phases: tuple[Phase, ...], index: int) -> None:
        job.event = None
        demands = []
        for key, work in phases[index].demands:
            res = self._resolve(key)
            if res is not None and work > 0:
                demands.append((res, work))
        job.flow = self.net.start(
            demands, lambda _f: self._run_phase(job, phases, index + 1), tag=job.tag
        )

    def _finish(self, job: KernelJob) -> None:
        for tensor, nbytes in job.kernel.sram_tensors:
            if tensor in self.tile_cache:
                self.tile_cache.touch(tensor)
            else:
                self.tile_cache.insert(tensor, nbytes)
        self.kernels_run += 1
        self._end(job)
        job.on_done(job)

    def _end(self, job: KernelJob) -> None:
        job.finished = self.sim.now
        job.flow = None
        if job.started is not None:
            self.busy_time += job.finished - job.started
        if self._running is job:
            self._running = None
            self.sim.after(0.0, self._next)
