"""Simulated system executing BASP Stream Protocol commands.

The host submits small command descriptors; devices resolve dependencies from
explicit completion events, move data by DMA and run kernels.  The model
includes bounded per-device command queues with backpressure, DMA windows
(a device only accepts weight data into memory that was ALLOCATEd for it),
tensor-version checks, timeouts, cancellation and a complete execution log.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass

from basp.core.tensor import CompressionFormat
from basp.core.topology import Topology
from basp.memory.allocator import OutOfMemory
from basp.memory.replication import ReplicaManager
from basp.memory.residency import CopyState, ResidencyError, ResidencyMap
from basp.memory.tensor_registry import TensorRegistry
from basp.scheduler.congestion import PlanState
from basp.simulator.compute import (
    DeviceExecutor,
    Kernel,
    compute_key,
    decode_key,
    link_key,
    mem_key,
)
from basp.simulator.engine import Simulator
from basp.simulator.fluid import FluidNetwork, Resource
from basp.simulator.interconnect import Interconnect
from basp.streams.commands import Command, validate
from basp.streams.completion import CompletionEvent, CompletionQueue
from basp.streams.protocol import NO_DEVICE, BoundedQueue, Flags, Opcode, ProtocolError, Status

_MOVES = (Opcode.TRANSFER, Opcode.LOAD, Opcode.PREFETCH)
_RUNS = (Opcode.EXECUTE, Opcode.REDUCE)


@dataclass(frozen=True)
class ExecRecord:
    cmd_id: int
    opcode: Opcode
    device: int
    src_device: int
    submitted: float
    started: float | None
    finished: float
    status: Status
    nbytes: float
    tag: str
    meta: Hashable = None


class CmdState:
    __slots__ = (
        "cmd",
        "dependents",
        "dst_bytes",
        "finished",
        "handle",
        "holds",
        "kernel",
        "max_rate",
        "meta",
        "moved",
        "nbytes",
        "on_complete",
        "piece",
        "promoted",
        "queued",
        "reads",
        "started",
        "status",
        "submitted",
        "tag",
        "timer",
        "waiting",
        "writes",
    )  # fmt: skip

    def __init__(self, cmd: Command, now: float) -> None:
        self.cmd = cmd
        self.kernel: Kernel | None = None
        self.nbytes: float | None = None
        self.tag = ""
        self.meta: Hashable = None
        self.reads: tuple[tuple[int, int], ...] = ()
        self.writes: tuple[int, ...] = ()
        self.holds: list[tuple[int, int]] = []
        self.max_rate: float | None = None
        self.piece = False
        self.dst_bytes: float | None = None
        self.promoted = False
        self.status = Status.PENDING
        self.waiting: set[int] = set()
        self.dependents: list[CmdState] = []
        self.on_complete: Callable[[CmdState], None] | None = None
        self.submitted = now
        self.started: float | None = None
        self.finished: float | None = None
        self.handle = None
        self.timer = None
        self.queued = False
        self.moved = 0.0


class VirtualSystem:
    def __init__(
        self,
        topology: Topology,
        registry: TensorRegistry,
        residency: ResidencyMap,
        *,
        replicas: ReplicaManager | None = None,
        background_budget: float | None = None,
        completion_capacity: int = 4096,
    ) -> None:
        """``background_budget`` is the share of each link direction's bandwidth that all
        background traffic together (replication and prefetch) may use."""
        self.topology = topology
        self.registry = registry
        self.residency = residency
        self.replicas = replicas
        self.sim = Simulator()
        self.net = FluidNetwork(self.sim)
        self.res: dict[tuple, Resource] = {}
        for d, dev in topology.devices.items():
            owner = topology.memory_owner(d)
            if owner != d and mem_key(owner) in self.res:
                self.res[mem_key(d)] = self.res[mem_key(owner)]  # unified memory: one shared bus
            else:
                name = dev.memory_domain or dev.name
                self.res[mem_key(d)] = self.net.resource(f"mem:{name}", dev.mem_bandwidth)
            if dev.caps.can_compute:
                self.res[compute_key(d)] = self.net.resource(f"mac:{dev.name}", dev.compute_rate, "mac")
            for fmt in dev.caps.decode_rate:
                if dev.caps.decode_limited(fmt):
                    self.res[decode_key(d, fmt)] = self.net.resource(
                        f"decode:{dev.name}:{CompressionFormat(fmt).name}", dev.caps.decode_rate[fmt]
                    )
        mem = {d: self.res[mem_key(d)] for d in topology.devices}
        self.interconnect = Interconnect(self.sim, self.net, topology, mem)
        for (link, forward), res in self.interconnect.link_res.items():
            self.res[link_key(link, forward)] = res
        self.budget: dict[tuple[int, bool], Resource] = {}
        if background_budget is not None:
            if not 0 < background_budget <= 1:
                raise ValueError("background_budget must be in (0, 1]")
            for key, res in self.interconnect.link_res.items():
                self.budget[key] = self.net.resource(
                    f"budget:{res.name}", background_budget * res.capacity, "budget"
                )
        self.executors = {
            d: DeviceExecutor(self.sim, self.net, topology, dev, self.res.get)
            for d, dev in topology.devices.items() if dev.caps.can_compute
        }  # fmt: skip
        self.endpoints = {d: BoundedQueue(dev.queue_depth) for d, dev in topology.devices.items()}
        self.backlog: dict[int, deque[CmdState]] = {d: deque() for d in topology.devices}
        self.cmds: dict[int, CmdState] = {}
        self.completions = CompletionQueue(completion_capacity)
        self.loading: dict[tuple[int, int], int] = {}
        self.log: list[ExecRecord] = []
        self.violations: list[str] = []
        self.stats: Counter = Counter()
        self._next_cmd = 1

    # -- host interface -----------------------------------------------------------
    def new_id(self) -> int:
        cid = self._next_cmd
        self._next_cmd += 1
        return cid

    def submit(
        self,
        cmd: Command,
        *,
        kernel: Kernel | None = None,
        nbytes: float | None = None,
        tag: str = "",
        reads: tuple[tuple[int, int], ...] = (),
        writes: tuple[int, ...] = (),
        max_rate: float | None = None,
        piece: bool = False,
        dst_bytes: float | None = None,
        meta: Hashable = None,
        on_complete: Callable[[CmdState], None] | None = None,
    ) -> CmdState:
        """Submit a command.  ``kernel`` is the compiled kernel an EXECUTE/REDUCE refers to;
        ``reads``/``writes`` are its full operand lists as (tensor, device) / tensor ids."""
        validate(cmd)
        if cmd.op_id in self.cmds:
            raise ProtocolError(f"duplicate command id {cmd.op_id}")
        if cmd.opcode in _RUNS and kernel is None:
            raise ProtocolError(f"{cmd.opcode.name} command {cmd.op_id} has no kernel descriptor")
        state = CmdState(cmd, self.sim.now)
        state.kernel, state.nbytes, state.tag, state.meta = kernel, nbytes, tag, meta
        state.reads, state.writes, state.max_rate, state.piece = reads, writes, max_rate, piece
        state.dst_bytes = dst_bytes
        state.on_complete = on_complete
        self.cmds[cmd.op_id] = state
        if cmd.opcode in _MOVES and not piece:
            # known from submission on, so later plans can wait for (and promote) this load
            self.loading[(cmd.inputs[0], cmd.device_id)] = cmd.op_id
        self.stats["submitted"] += 1
        self._next_cmd = max(self._next_cmd, cmd.op_id + 1)
        for tensor, device in reads:
            copy = self.residency.copy(tensor, device)
            if copy is not None:
                copy.in_use += 1
                state.holds.append((tensor, device))
        broken = False
        for dep in cmd.deps:
            other = self.cmds.get(dep)
            if other is None:
                raise ProtocolError(f"command {cmd.op_id} depends on unknown command {dep}")
            if not other.status.is_terminal:
                state.waiting.add(dep)
                other.dependents.append(state)
            elif other.status is not Status.SUCCESS:
                broken = True
        if cmd.timeout_ns:
            state.timer = self.sim.after(cmd.timeout_ns * 1e-9, self._timeout, state)
        if broken:
            self.sim.after(0.0, self._complete, state, Status.FAILED)
        elif not state.waiting:
            self._dispatch(state)
        return state

    def status(self, cmd_id: int) -> Status:
        return self.cmds[cmd_id].status

    def cancel(self, cmd_id: int) -> bool:
        state = self.cmds.get(cmd_id)
        if state is None or state.status.is_terminal:
            return False
        self._abort(state, Status.CANCELLED)
        return True

    def promote(self, cmd_id: int) -> None:
        """Raise an in-flight background load to demand priority."""
        state = self.cmds.get(cmd_id)
        if state is None or state.status.is_terminal:
            return
        state.max_rate = None
        state.promoted = True
        if state.handle is not None and state.cmd.opcode in _MOVES:
            self.interconnect.promote(state.handle)
        copy = self.residency.copy(state.cmd.inputs[0], state.cmd.device_id)
        if copy is not None:
            copy.speculative = False
        self.stats["promoted"] += 1

    def run(self, until: float | None = None, max_events: int | None = None) -> float:
        now = self.sim.run(until, max_events)
        self.net.settle()
        return now

    def plan_state(self) -> PlanState:
        """Snapshot of availability estimates for the scheduler."""
        now = self.sim.now
        device_free = {
            d: now + ex.backlog_s() for d, ex in self.executors.items() if ex.queue_length
        }
        for d, ex in self.executors.items():
            if d not in device_free and ex.last_end == now:
                device_free[d] = now  # just finished: a kernel issued now joins its launch
        link_free = {
            key: now + queued / self.interconnect.link_res[key].capacity
            for key, queued in self.interconnect.queued_by_direction().items() if queued > 0
        }  # fmt: skip
        pending_eta = {}
        for key, cmd_id in self.loading.items():
            handle = self.cmds[cmd_id].handle
            if handle is not None:
                pending_eta[key] = now + self.interconnect.remaining_s(handle)
        for device in self.topology.devices:
            for copy in self.residency.on_device(device):
                key = (copy.tensor, device)
                if copy.state is CopyState.PENDING and key not in pending_eta and copy.eta:
                    pending_eta[key] = max(copy.eta, now)
        sram = {d: ex.tile_cache for d, ex in self.executors.items() if ex.device.sram_bytes}
        victims = self.replicas.victims if self.replicas is not None else None
        return PlanState(now, self.residency, device_free=device_free, link_free=link_free,
                         pending_eta=pending_eta, sram_actual=sram, victims_fn=victims)  # fmt: skip

    # -- dispatch -----------------------------------------------------------------
    def _dispatch(self, state: CmdState) -> None:
        device = state.cmd.device_id
        if device == NO_DEVICE:
            self._begin(state)
            return
        if device not in self.endpoints:
            self._complete(state, Status.FAILED)
            return
        if self.backlog[device] or not self.endpoints[device].try_put(state):
            self.backlog[device].append(state)
            self.stats["backpressure"] += 1
            return
        state.queued = True
        state.status = Status.QUEUED
        self._begin(state)

    def _begin(self, state: CmdState) -> None:
        cmd = state.cmd
        op = cmd.opcode
        state.started = self.sim.now
        if op in _MOVES:
            self._begin_move(state)
        elif op in _RUNS:
            self._begin_run(state)
        elif op is Opcode.ALLOCATE:
            fmt = CompressionFormat(cmd.precision) if cmd.precision else None
            try:
                self.residency.begin_load(
                    cmd.inputs[0], cmd.device_id, fmt=fmt,
                    speculative=bool(cmd.flags & Flags.SPECULATIVE),
                    eta=cmd.arg1 * 1e-9 if cmd.arg1 else None,
                )  # fmt: skip
            except (OutOfMemory, ResidencyError, KeyError) as exc:
                self._violation(f"ALLOCATE {cmd.op_id}: {exc}")
                self._complete(state, Status.FAILED)
                return
            self._complete(state, Status.SUCCESS)
        elif op is Opcode.FREE:
            tensor, device = cmd.inputs[0], cmd.device_id
            try:
                self.residency.evict(tensor, device)
            except ResidencyError as exc:
                self._violation(f"FREE {cmd.op_id}: {exc}")
                self._complete(state, Status.FAILED)
                return
            if self.replicas is not None:
                self.replicas.untrack(tensor, device)
            if device in self.executors:
                self.executors[device].tile_cache.discard(tensor)
            self._complete(state, Status.SUCCESS)
        elif op is Opcode.CANCEL:
            self._complete(state, Status.SUCCESS, detail=int(self.cancel(cmd.arg0)))
        elif op is Opcode.QUERY_STATUS:
            target = self.cmds.get(cmd.arg0)
            detail = int(target.status) if target is not None else int(Status.FAILED)
            self._complete(state, Status.SUCCESS, detail=detail)
        else:  # BARRIER, COMPLETE: pure synchronisation points
            self._complete(state, Status.SUCCESS)

    def _begin_move(self, state: CmdState) -> None:
        cmd = state.cmd
        tensor, src, dst = cmd.inputs[0], cmd.src_device, cmd.device_id
        if tensor not in self.registry or src not in self.topology.devices:
            self._complete(state, Status.PROTECTION_FAULT)
            return
        source = self.residency.copy(tensor, src)
        if source is None or source.state is CopyState.PENDING:
            self._violation(f"{cmd.opcode.name} {cmd.op_id}: tensor {tensor} is not on device {src}")
            self._complete(state, Status.PROTECTION_FAULT)
            return
        if cmd.version != self.registry.version(tensor) or not self.residency.valid(tensor, src):
            self._complete(state, Status.VERSION_MISMATCH)
            return
        target = self.residency.copy(tensor, dst)
        if target is None:
            if not self.registry.spec(tensor).policy.temporary:
                # DMA isolation: weight data only lands in memory ALLOCATEd for it.
                self._violation(f"{cmd.opcode.name} {cmd.op_id}: no DMA window for tensor {tensor} on {dst}")
                self._complete(state, Status.PROTECTION_FAULT)
                return
            try:
                self.residency.begin_load(tensor, dst)
            except OutOfMemory as exc:
                self._violation(f"{cmd.opcode.name} {cmd.op_id}: {exc}")
                self._complete(state, Status.FAILED)
                return
        elif target.state is CopyState.VALID and not state.piece:
            self._complete(state, Status.SUCCESS)  # already there: nothing to move
            return
        nbytes = state.nbytes if state.nbytes is not None else float(source.nbytes)
        default_tag = "prefetch" if cmd.opcode is Opcode.PREFETCH else "data"
        extra: tuple = ()
        landed = self.residency.copy(tensor, dst)
        if state.dst_bytes is not None and landed is not None and landed.fmt != source.fmt:
            # Transcode on arrival: the payload passes the destination decoder as it lands.
            decoder = self.res.get(decode_key(dst, source.fmt))
            if decoder is not None:
                extra = ((decoder, nbytes),)
        priority = 0 if state.promoted else cmd.priority
        if priority > 0 and self.budget:
            # Background traffic shares one bandwidth budget per link direction.
            extra += tuple(
                (self.budget[(hop.link, hop.forward)], nbytes)
                for hop in self.topology.route(src, dst)
            )
        self.loading[(tensor, dst)] = cmd.op_id
        state.status = Status.RUNNING
        state.handle = self.interconnect.transfer(
            src, dst, nbytes, lambda _h: self._moved(state), tag=state.tag or default_tag,
            priority=priority, max_rate=state.max_rate, dst_bytes=state.dst_bytes, extra=extra,
        )  # fmt: skip

    def _moved(self, state: CmdState) -> None:
        cmd = state.cmd
        tensor, dst = cmd.inputs[0], cmd.device_id
        if self.loading.get((tensor, dst)) == cmd.op_id:
            del self.loading[(tensor, dst)]
        copy = self.residency.copy(tensor, dst)
        if copy is not None and copy.state is CopyState.PENDING:
            self.residency.commit(tensor, dst, self.sim.now)
            if cmd.flags & Flags.RETAIN and self.replicas is not None:
                self.replicas.track(tensor, dst, speculative=copy.speculative)
                self.replicas.inflight.discard((tensor, dst))
        state.moved = state.handle.nbytes
        self._complete(state, Status.SUCCESS)

    def _begin_run(self, state: CmdState) -> None:
        cmd = state.cmd
        executor = self.executors.get(cmd.device_id)
        if executor is None:
            self._violation(f"{cmd.opcode.name} {cmd.op_id}: device {cmd.device_id} cannot compute")
            self._complete(state, Status.FAILED)
            return
        state.handle = executor.submit(
            state.kernel, lambda _job: self._ran(state), on_start=lambda: self._check_inputs(state),
            tag="kernel",
        )  # fmt: skip

    def _check_inputs(self, state: CmdState) -> bool:
        state.status = Status.RUNNING
        state.started = self.sim.now
        for tensor, device in state.reads:
            if not self.residency.valid(tensor, device):
                self._violation(
                    f"{state.cmd.opcode.name} {state.cmd.op_id} started on device "
                    f"{state.cmd.device_id} without a valid copy of tensor {tensor} on {device}"
                )
                self.sim.after(0.0, self._complete, state, Status.FAILED)
                return False
            self.residency.touch(tensor, device, self.sim.now)
        return True

    def _ran(self, state: CmdState) -> None:
        device = state.cmd.device_id
        for tensor in state.writes:
            if self.residency.copy(tensor, device) is None:
                try:
                    self.residency.place(tensor, device, now=self.sim.now)
                except OutOfMemory as exc:
                    self._violation(f"EXECUTE {state.cmd.op_id}: {exc}")
                    self._complete(state, Status.FAILED)
                    return
        self._complete(state, Status.SUCCESS)

    # -- completion ---------------------------------------------------------------
    def _timeout(self, state: CmdState) -> None:
        state.timer = None
        if not state.status.is_terminal:
            self._abort(state, Status.TIMEOUT)

    def _abort(self, state: CmdState, status: Status) -> None:
        cmd = state.cmd
        if state.handle is not None:
            if cmd.opcode in _MOVES:
                state.moved = self.interconnect.cancel(state.handle)
            else:
                self.executors[cmd.device_id].cancel(state.handle)
        if cmd.opcode in _MOVES:
            tensor, dst = cmd.inputs[0], cmd.device_id
            if self.loading.get((tensor, dst)) == cmd.op_id:
                del self.loading[(tensor, dst)]
            copy = self.residency.copy(tensor, dst)
            if copy is not None and copy.state is CopyState.PENDING:
                self.residency.abort(tensor, dst)
            if self.replicas is not None:
                self.replicas.inflight.discard((tensor, dst))
        device = cmd.device_id
        if device in self.backlog and state in self.backlog[device]:
            self.backlog[device].remove(state)
        self._complete(state, status)

    def _complete(self, state: CmdState, status: Status, *, detail: int = 0) -> None:
        if state.status.is_terminal:
            return
        cmd = state.cmd
        if cmd.opcode in _MOVES and self.loading.get((cmd.inputs[0], cmd.device_id)) == cmd.op_id:
            del self.loading[(cmd.inputs[0], cmd.device_id)]
        state.status = status
        state.finished = self.sim.now
        self.sim.cancel(state.timer)
        state.timer = None
        for tensor, device in state.holds:
            copy = self.residency.copy(tensor, device)
            if copy is not None and copy.in_use > 0:
                copy.in_use -= 1
        state.holds = []
        self.stats[status.name.lower()] += 1
        self.log.append(ExecRecord(cmd.op_id, cmd.opcode, cmd.device_id, cmd.src_device,
                                   state.submitted, state.started, state.finished, status,
                                   state.moved, state.tag, state.meta))  # fmt: skip
        if not self.completions.post(CompletionEvent(cmd.completion_id, cmd.op_id, cmd.device_id,
                                                     status, int(self.sim.now * 1e9),
                                                     int(state.moved), detail)):  # fmt: skip
            self.stats["completion_overflow"] += 1
        if state.queued:
            state.queued = False
            queue = self.endpoints[cmd.device_id]
            queue.remove(lambda item: item is state)
            backlog = self.backlog[cmd.device_id]
            while backlog and queue.credits > 0:
                nxt = backlog.popleft()
                queue.put(nxt)
                nxt.queued = True
                nxt.status = Status.QUEUED
                self.sim.after(0.0, self._begin_if_live, nxt)
        if state.on_complete is not None:
            state.on_complete(state)  # the host sees a completion before anything it unblocks
        dependents, state.dependents = state.dependents, []
        for dep in dependents:
            if dep.status.is_terminal:
                continue
            dep.waiting.discard(cmd.op_id)
            if status is not Status.SUCCESS:
                self._abort(dep, Status.FAILED)
            elif not dep.waiting:
                self._dispatch(dep)

    def _begin_if_live(self, state: CmdState) -> None:
        if not state.status.is_terminal:
            self._begin(state)

    def _violation(self, message: str) -> None:
        self.violations.append(f"t={self.sim.now:.9f}: {message}")
