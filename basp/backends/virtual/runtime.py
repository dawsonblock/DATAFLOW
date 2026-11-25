"""Event-driven execution of a MoE decoding workload on the virtual system.

Per decoding step a stream walks the layers: the trunk block of a layer produces
the routing decision, the selected experts, their reduction and the next trunk
block are then scheduled as one batch, emitted as command streams and awaited
through completion events.  Routing comes from a :class:`RoutingSource` -- a
recorded/synthetic trace, or real values when a numeric payload is attached --
and is never altered by the scheduler.

Several independent streams (concurrent requests) can share one system; a
:class:`Meter` collects the system-wide measurements.
"""

from __future__ import annotations

import itertools
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Protocol

from basp.backends.virtual.system import CmdState, VirtualSystem
from basp.compiler.emit import Emitter
from basp.compiler.lower import ModelProgram
from basp.core.capabilities import DeviceKind
from basp.core.graph import Graph, OpCode, OpNode
from basp.dataflow.base import DataflowMode, PathEstimate, ResolvedPlan
from basp.dataflow.predictive import Prefetcher
from basp.scheduler.scheduler import Scheduler
from basp.simulator.energy import EnergyReport, energy_report
from basp.streams.commands import Command
from basp.streams.protocol import NO_DEVICE, Opcode, Status
from basp.workloads.trace import RoutingStep, RoutingTrace

_TRUNK_OPS = (OpCode.TRUNK, OpCode.RMS_NORM, OpCode.ATTENTION, OpCode.ROUTE, OpCode.ROPE,
              OpCode.KV_APPEND)  # fmt: skip


class RoutingSource(Protocol):
    def select(self, step: int, layer: int) -> RoutingStep: ...


class TraceRouting:
    """Routing decisions replayed from a trace (wrapping around at its end).

    With ``row`` only that batch row of the trace is replayed, which turns a
    batched trace into independent single-sequence streams.
    """

    def __init__(self, trace: RoutingTrace, row: int | None = None) -> None:
        self.trace = trace
        self.row = row

    def select(self, step: int, layer: int) -> RoutingStep:
        routing = self.trace.at(step % self.trace.n_steps, layer)
        if self.row is None:
            return routing
        return RoutingStep((routing.experts[self.row],), (routing.coeffs[self.row],))


def op_kind(op: OpNode) -> str:
    """Coarse role of an operation in a decoding step: trunk, expert, reduce or head."""
    if op.attrs.get("head"):
        return "head"
    if op.opcode is OpCode.REDUCE:
        return "reduce"
    if op.opcode in _TRUNK_OPS:
        return "trunk"
    if op.opcode is OpCode.EXPERT or "expert" in op.attrs:
        return "expert"
    return "other"


@dataclass
class OpRecord:
    op_id: int
    name: str
    opcode: OpCode
    devices: tuple[int, ...]
    modes: tuple[str, ...]
    issued: float
    est_finish: float
    started: float | None = None
    finished: float | None = None
    deps: tuple[int, ...] = ()
    kind: str = "other"
    step: int = 0
    stream: int = 0
    inputs_ready: float | None = None  # when the last input transfer of the op arrived
    delivered: float | None = None  # when its results were available to their consumer
    estimate: PathEstimate | None = None


@dataclass
class RunResult:
    policy: str
    tokens: int
    batch: int
    elapsed_s: float
    token_times: list[float]
    link_bytes: dict[str, float]
    energy: EnergyReport
    devices: dict[str, dict[str, float]]
    mode_counts: dict[str, int]
    scheduler: dict[str, float]
    cache: dict[str, float]
    prefetch: dict[str, float]
    stall_s: float
    weight_transfer_s: float
    estimate_error: dict[str, float]
    system: dict[str, float]
    violations: list[str]
    failures: list[str]
    op_log: list[OpRecord] = field(default_factory=list)
    streams: int = 1

    @property
    def tokens_per_s(self) -> float:
        """Aggregate throughput: tokens of all sequences per second."""
        return self.tokens * self.batch / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def token_latency_s(self) -> float:
        """Mean time between successive tokens of one sequence."""
        return sum(self.token_times) / len(self.token_times) if self.token_times else 0.0

    @property
    def link_bytes_total(self) -> float:
        return sum(self.link_bytes.values())

    @property
    def link_bytes_per_token(self) -> float:
        n = self.tokens * self.batch
        return self.link_bytes_total / n if n else 0.0

    @property
    def tokens_per_joule(self) -> float:
        total = self.energy.total_j
        return self.tokens * self.batch / total if total > 0 else 0.0

    @property
    def average_power_w(self) -> float:
        return self.energy.total_j / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def ok(self) -> bool:
        return not self.violations and not self.failures


class _Batch:
    __slots__ = ("est", "issued", "label", "pending", "then")

    def __init__(self, issued: float, est: float, then: Callable[[], None], label: str) -> None:
        self.issued = issued
        self.est = est
        self.pending: set[int] = set()
        self.then = then
        self.label = label


class Meter:
    """System-wide measurements shared by every stream running on one system."""

    def __init__(self, system: VirtualSystem, program: ModelProgram, scheduler: Scheduler,
                 prefetcher: Prefetcher | None, *, warmup_tokens: int = 0) -> None:  # fmt: skip
        self.system = system
        self.sim = system.sim
        self.program = program
        self.scheduler = scheduler
        self.prefetcher = prefetcher
        self.warmup_tokens = warmup_tokens
        self.counters: Counter = Counter()
        self.failures: list[str] = []
        self.op_log: list[OpRecord] = []
        self.token_log: list[tuple[int, int, float, float]] = []  # stream, step, start, end
        self.stall = 0.0
        self.err_abs = 0.0
        self.err_signed = 0.0
        self.err_n = 0
        self.wall = 0.0
        self._mark: dict | None = None
        self.take_mark()

    def token_done(self, stream: int, step: int, start: float) -> None:
        self.token_log.append((stream, step, start, self.sim.now))
        if len(self.token_log) == self.warmup_tokens and self.warmup_tokens > 0:
            self.take_mark()

    def batch_done(self, estimate: float, actual: float) -> None:
        if actual > 0:
            self.err_abs += abs(estimate - actual) / actual
            self.err_signed += (estimate - actual) / actual
            self.err_n += 1

    def fail(self, state: CmdState) -> None:
        cmd = state.cmd
        self.failures.append(
            f"command {cmd.op_id} ({cmd.opcode.name} on device {cmd.device_id}) ended {state.status.name}"
        )

    # -- results ------------------------------------------------------------------
    def take_mark(self) -> None:
        self.system.net.settle()
        self._mark = {
            "time": self.sim.now,
            "served": {k: r.served for k, r in self.system.res.items()},
            "tags": {k: dict(r.by_tag) for k, r in self.system.res.items()},
            "busy": {d: ex.busy_time for d, ex in self.system.executors.items()},
            "gated": {d: (ex.gated_seconds(self.sim.now), ex.wakes)
                      for d, ex in self.system.executors.items()},
            "kernels": {d: ex.kernels_run for d, ex in self.system.executors.items()},
            "wait": {d: ex.wait_time for d, ex in self.system.executors.items()},
            "counters": Counter(self.counters),
            "stall": self.stall,
            "tokens": len(self.token_log),
            "sched": Counter(self.scheduler.stats),
            "sched_wall": self.scheduler.wall_s,
            "modes": Counter(self.scheduler.mode_counts),
            "log": len(self.system.log),
            "prefetch": Counter(self.prefetcher.stats) if self.prefetcher else Counter(),
        }

    def result(self) -> RunResult:
        system, topo, mark = self.system, self.system.topology, self._mark
        system.net.settle()
        elapsed = self.sim.now - mark["time"]
        delta = {k: SimpleNamespace(served=r.served - mark["served"][k]) for k, r in system.res.items()}
        link_bytes: dict[str, float] = {}
        for key, res in system.res.items():
            if key[0] != "link":
                continue
            for tag, amount in res.by_tag.items():
                moved = amount - mark["tags"][key].get(tag, 0.0)
                if moved > 0:
                    link_bytes[tag] = link_bytes.get(tag, 0.0) + moved
        devices: dict[str, dict[str, float]] = {}
        for d, dev in topo.devices.items():
            row = {"mem_bytes": delta[("mem", d)].served}
            row["mem_utilisation"] = (
                row["mem_bytes"] / (dev.mem_bandwidth * elapsed) if elapsed > 0 else 0.0
            )
            ex = system.executors.get(d)
            if ex is not None:
                busy = ex.busy_time - mark["busy"][d]
                row.update(
                    macs=delta[("compute", d)].served,
                    kernels=ex.kernels_run - mark["kernels"][d],
                    busy_s=busy,
                    busy_fraction=busy / elapsed if elapsed > 0 else 0.0,
                    queue_wait_s=ex.wait_time - mark["wait"][d],
                    sram_hits=ex.tile_cache.hits,
                    sram_misses=ex.tile_cache.misses,
                )
                # Memory bandwidth actually used while the device was executing kernels.
                row["mem_utilisation_busy"] = (
                    row["mem_bytes"] / (dev.mem_bandwidth * busy) if busy > 0 else 0.0
                )
            devices[dev.name] = row
        gating = {
            d: (ex.gated_seconds(self.sim.now) - mark["gated"][d][0], ex.wakes - mark["gated"][d][1])
            for d, ex in system.executors.items()
        }
        counters = self.counters - mark["counters"]
        sched = self.scheduler.stats - mark["sched"]
        sched_wall = self.scheduler.wall_s - mark["sched_wall"]
        modes = self.scheduler.mode_counts - mark["modes"]
        log = system.log[mark["log"]:]
        weight_s = sum(
            r.finished - r.submitted for r in log
            if r.tag == "weights" and r.status is Status.SUCCESS
        )  # fmt: skip
        wasted = sum(r.nbytes for r in log if r.tag == "prefetch" and r.status is Status.CANCELLED)
        pstats: Counter = Counter()
        if self.prefetcher is not None:
            pstats = self.prefetcher.stats - mark["prefetch"]
            wasted += pstats.get("wasted_bytes", 0)
        dma_wait = sum(s.wait_time for s in system.interconnect.stats.values())
        dma_max = max((s.max_waiting for s in system.interconnect.stats.values()), default=0)
        measured = self.token_log[mark["tokens"]:]
        tokens = len(measured)
        expert_ops = counters.get("expert_ops", 0)
        decisions = sched.get("batches", 0) * self.scheduler.policy.decision_latency_s
        overhead = decisions / elapsed if elapsed > 0 else 0.0
        return RunResult(
            policy=self.scheduler.policy.name,
            tokens=tokens,
            batch=self.program.batch,
            elapsed_s=elapsed,
            token_times=[end - start for _, _, start, end in measured],
            link_bytes=link_bytes,
            energy=energy_report(topo, delta, elapsed, gating),
            devices=devices,
            mode_counts={
                m.name.lower(): n for m, n in sorted(modes.items()) if isinstance(m, DataflowMode)
            },
            scheduler={
                "batches": sched.get("batches", 0),
                "ops": sched.get("ops", 0),
                "plans_evaluated": sched.get("evaluated", 0),
                "joint_improvements": sched.get("joint_improvements", 0),
                "relaxed": sched.get("relaxed", 0),
                "wall_s": sched_wall,
                "wall_us_per_op": 1e6 * sched_wall / max(sched.get("ops", 0), 1),
                "modelled_overhead_fraction": overhead,
            },
            cache={
                "expert_ops": expert_ops,
                "cache_hits": counters.get("cache_hits", 0),
                "hit_rate": counters.get("cache_hits", 0) / expert_ops if expert_ops else 0.0,
                "fills": counters.get("fills", 0),
                "fills_completed": counters.get("fills_completed", 0),
                "fills_failed": counters.get("fills_failed", 0),
                "fill_evictions": counters.get("fill_evictions", 0),
            },
            prefetch={
                "issued": pstats.get("issued", 0),
                "hits": pstats.get("hits", 0),
                "mispredicted": pstats.get("mispredicted", 0),
                "cancelled": counters.get("prefetch_cancelled", 0),
                "issued_bytes": pstats.get("issued_bytes", 0),
                "wasted_bytes": wasted,
            },
            stall_s=self.stall - mark["stall"],
            weight_transfer_s=weight_s,
            estimate_error={
                "mean_abs_rel": self.err_abs / self.err_n if self.err_n else 0.0,
                "mean_signed_rel": self.err_signed / self.err_n if self.err_n else 0.0,
                "batches": self.err_n,
            },
            system={
                "events": self.sim.events_processed,
                "commands": system.stats.get("submitted", 0),
                "backpressure": system.stats.get("backpressure", 0),
                "dma_wait_s": dma_wait,
                "dma_max_waiting": dma_max,
                "completion_overflow": system.stats.get("completion_overflow", 0),
                "wall_s": self.wall,
            },
            violations=list(system.violations),
            failures=list(self.failures),
            op_log=self.op_log,
        )


class Runtime:
    """One stream: a sequence (or a lock-step batch of sequences) being decoded."""

    def __init__(
        self,
        system: VirtualSystem,
        program: ModelProgram,
        scheduler: Scheduler,
        routing: RoutingSource,
        *,
        tokens: int,
        warmup_tokens: int = 0,
        predictor=None,
        prefetcher: Prefetcher | None = None,
        payload=None,
        keep_log: bool = False,
        meter: Meter | None = None,
        stream: int = 0,
    ) -> None:
        if meter is None and tokens <= warmup_tokens:
            raise ValueError("tokens must exceed warmup_tokens")
        self.system = system
        self.sim = system.sim
        self.program = program
        self.scheduler = scheduler
        self.policy = scheduler.policy
        self.routing = routing
        self.tokens = tokens
        self.predictor = predictor
        self.prefetcher = prefetcher
        self.payload = payload
        self.keep_log = keep_log
        self.stream = stream
        self.meter = meter or Meter(system, program, scheduler, prefetcher, warmup_tokens=warmup_tokens)
        self.emitter = Emitter(system, program.spec.hidden, program.batch)
        self.graph = Graph()
        self.done = False
        self._step = 0
        self._step_start = 0.0
        self._plans: dict[int, ResolvedPlan] = {}
        self._op_pending: dict[int, int] = {}
        self._records: dict[int, OpRecord] = {}
        host = self.policy.host_in_loop
        self._sync_s = system.topology.sync_latency_s if host else 0.0
        self._decision_s = self.policy.decision_latency_s if host else 0.0
        self._expert_devices: dict[int, tuple[int, ...]] = {}
        self._released: set[int] = set()
        self._live: dict[int, None] = {}  # the step's temporaries not yet released, in creation order
        self._scanned = 0  # graph tensors already examined for temporaries

    # backwards-compatible views of the shared measurements
    @property
    def token_times(self) -> list[float]:
        return [end - start for s, _, start, end in self.meter.token_log if s == self.stream]

    @property
    def op_log(self) -> list[OpRecord]:
        return self.meter.op_log

    @property
    def failures(self) -> list[str]:
        return self.meter.failures

    # -- driver -------------------------------------------------------------------
    def start(self, at: float = 0.0) -> None:
        self.sim.at(max(at, self.sim.now), self._begin_step, 0)

    def run(self, max_events: int | None = None) -> RunResult:
        t0 = time.perf_counter()
        self.start()
        self.system.run(max_events=max_events)
        self.meter.wall = time.perf_counter() - t0
        if not self.done and not self.meter.failures:
            self.meter.failures.append("simulation stalled before the workload completed")
        return self.meter.result()

    def _begin_step(self, step: int) -> None:
        self._step = step
        self._step_start = self.sim.now
        self.graph = Graph()
        self.emitter.new_step()
        self._plans.clear()
        self._op_pending.clear()
        self._records.clear()
        self._released = set()
        self._live = {}
        self._scanned = 0
        program = self.program
        x = program.input_tensor(self.graph, step)
        self.system.residency.place(x, program.trunk_device[0], now=self.sim.now)
        if self.payload is not None:
            self.payload.begin_step(step, x)
        trunk = program.trunk(self.graph, step, 0, x)
        self._submit(trunk.ops, lambda: self._after_trunk(step, 0, trunk.outputs), f"s{step}.trunk0",
                     broadcast=(trunk.outputs["h"], 0))  # fmt: skip

    def _after_trunk(self, step: int, layer: int, tensors: dict[str, int]) -> None:
        """The router output of ``layer`` is known: issue the selected experts."""
        program = self.program
        if self.payload is not None:
            routing = self.payload.route(step, layer, tensors["logits"])
        else:
            routing = self.routing.select(step, layer)
        selected = set(routing.unique_experts())
        # Everything produced before this trunk block has been consumed: release it now, so
        # scratch memory holds one layer's temporaries rather than a whole step's.
        self._release(keep=set(tensors.values()))
        if self.prefetcher is not None:
            for tensor, device in self.prefetcher.reveal(layer, selected):
                loading = self.system.loading.get((tensor, device))
                if loading is not None:
                    cancel = Command(self.system.new_id(), Opcode.CANCEL, NO_DEVICE, arg0=loading)
                    self.system.submit(cancel, tag="cancel")
                    self.meter.counters["prefetch_cancelled"] += 1
        if self.predictor is not None:
            if len(routing.experts) == 1:
                self.predictor.observe(layer, routing.experts[0], stream=self.stream)
            else:
                self.predictor.observe_step(layer, routing)
        experts = program.experts(self.graph, step, layer, tensors["h"], routing)
        reduce = program.reduce(self.graph, step, layer, experts.outputs, tensors["resid"])
        ops = experts.ops + reduce.ops
        x = reduce.outputs["x"]
        broadcast = None
        if layer + 1 < program.spec.n_layers:
            trunk = program.trunk(self.graph, step, layer + 1, x)
            ops += trunk.ops
            broadcast = (trunk.outputs["h"], layer + 1)
            then = lambda: self._after_trunk(step, layer + 1, trunk.outputs)
        else:
            head = program.lm_head(self.graph, step, x)
            ops += head.ops
            then = lambda: self._end_step(step)
        self.meter.counters["expert_ops"] += len(experts.outputs)
        self._submit(ops, then, f"s{step}.l{layer}", prefetch_after=layer, broadcast=broadcast)

    def _end_step(self, step: int) -> None:
        residency = self.system.residency
        if self.payload is not None:
            self.payload.end_step(step)
        self._release(keep=set())
        for layer, kv in enumerate(self.program.kv):
            residency.write(kv, self.program.trunk_device[layer])  # KV append = versioned write
        self.meter.token_done(self.stream, step, self._step_start)
        if step + 1 < self.tokens:
            self._begin_step(step + 1)
        else:
            self.done = True

    def _release(self, keep: set[int]) -> None:
        """Free the step's temporaries (dependency-managed lifetime) except ``keep``."""
        residency, registry = self.system.residency, self.system.registry
        tensors = self.graph.tensors
        if len(tensors) > self._scanned:  # only tensors created since the last release are new
            for tensor, spec in itertools.islice(tensors.items(), self._scanned, None):
                if spec.policy.temporary:
                    self._live[tensor] = None
            self._scanned = len(tensors)
        for tensor in [*self._live, *self.emitter.partials]:
            if tensor in keep or tensor in self._released or tensor not in registry:
                continue
            if residency.copies(tensor):
                residency.drop(tensor)
            registry.unregister(tensor)
            self._released.add(tensor)
            self._live.pop(tensor, None)

    # -- batches ------------------------------------------------------------------
    def _targets(self, layer: int, source: int) -> tuple[int, ...]:
        """Devices other than ``source`` that hold experts of ``layer`` and may run them."""
        if layer not in self._expert_devices:
            residency, allowed = self.system.residency, self.policy.compute_devices
            found: set[int] = set()
            for tensor in self.program.layer_expert_tensors(layer):
                found.update(c.device for c in residency.holders(tensor))
            self._expert_devices[layer] = tuple(sorted(
                d for d in found
                if self.system.topology.devices[d].caps.can_compute and (allowed is None or d in allowed)
            ))  # fmt: skip
        return tuple(d for d in self._expert_devices[layer] if d != source)

    def _submit(self, ops: list[OpNode], then: Callable[[], None], label: str,
                prefetch_after: int | None = None, broadcast: tuple[int, int] | None = None) -> None:  # fmt: skip
        system, meter = self.system, self.meter
        st = system.plan_state()
        plans, final = self.scheduler.schedule(ops, self.graph, st)
        now = self.sim.now
        batch = _Batch(now, max(p.finish for p in plans), then, label)
        emitted = []
        extra = []
        for plan in plans:
            self._account_plan(plan)
            ep = self.emitter.emit_plan(plan)
            emitted.append(ep)
            batch.pending.update(ep.final)
            op = plan.op
            self._plans[op.op_id] = plan
            self._op_pending[op.op_id] = len(ep.executes)
            if self.keep_log:
                record = OpRecord(
                    op.op_id, op.name, op.opcode, plan.devices,
                    tuple(sorted(m.name.lower() for m in plan.choice.modes)), now, plan.finish,
                    deps=self.graph.deps(op.op_id), kind=op_kind(op), step=self._step,
                    stream=self.stream, estimate=plan.estimate,
                )  # fmt: skip
                self._records[op.op_id] = record
                meter.op_log.append(record)
            if (broadcast is not None and self.policy.eager_activations
                    and broadcast[0] in op.outputs and len(plan.shards) == 1):  # fmt: skip
                # Stream the hidden state to every device that may run this layer's experts
                # as soon as it exists, instead of after the routing decision.
                source = plan.shards[0].device
                extra.extend(self.emitter.emit_broadcast(
                    broadcast[0], source, self._targets(broadcast[1], source), ep.executes))  # fmt: skip
        fills = self.scheduler.propose_fills(plans, final)
        if self.prefetcher is not None and prefetch_after is not None:
            fills += self.prefetcher.propose(prefetch_after, final)
        fill_cmds = []
        for action in fills:
            eta = now + action.nbytes / action.max_rate
            fill_cmds.extend(self.emitter.emit_fill(action, eta))
            meter.counters["fills"] += 1
            meter.counters["fill_evictions"] += len(action.evict)

        def issue() -> None:
            finals = batch.pending.copy()
            for ep in emitted:
                executes = set(ep.executes)
                for e in ep.commands:
                    cid = e.cmd.op_id
                    hook = lambda s, b=batch, op=ep.plan.op, x=cid in executes, f=cid in finals: (
                        self._on_command(s, b, op, x, f)
                    )
                    system.submit(e.cmd, kernel=e.kernel, nbytes=e.nbytes, tag=e.tag, reads=e.reads,
                                  writes=e.writes, max_rate=e.max_rate, piece=e.piece,
                                  dst_bytes=e.dst_bytes, meta=e.meta, on_complete=hook)  # fmt: skip
            for e in extra:
                system.submit(e.cmd, nbytes=e.nbytes, tag=e.tag, reads=e.reads, meta=e.meta,
                              on_complete=self._on_other)  # fmt: skip
            for e in fill_cmds:
                system.submit(e.cmd, nbytes=e.nbytes, tag=e.tag, reads=e.reads,
                              dst_bytes=e.dst_bytes, meta=e.meta, on_complete=self._on_fill)  # fmt: skip
            system.completions.drain()

        if self._decision_s > 0:
            self.sim.after(self._decision_s, issue)
        else:
            issue()

    def _account_plan(self, plan: ResolvedPlan) -> None:
        op = plan.op
        counters = self.meter.counters
        self.meter.stall += plan.weight_wait
        if plan.weight_wait > 0:
            counters["stalled_ops"] += 1
        if op.opcode is OpCode.EXPERT or op.attrs.get("part") == "down":
            # An expert run that found its weights already resident on a GPU.
            shard = plan.shards[0]
            on_gpu = self.system.topology.devices[shard.device].kind is DeviceKind.GPU
            moved = plan.choice.stream or any(t.tag == "weights" for t in shard.inputs)
            if on_gpu and not moved and len(plan.shards) == 1:
                counters["cache_hits"] += 1

    def _on_other(self, state: CmdState) -> None:
        if state.status is not Status.SUCCESS:
            self.meter.fail(state)

    def _on_fill(self, state: CmdState) -> None:
        """Background replication and prefetch are best effort: a failure costs nothing but
        the opportunity, so it is counted rather than treated as a failed run."""
        counters = self.meter.counters
        moves = state.cmd.opcode in (Opcode.TRANSFER, Opcode.PREFETCH)
        if state.status is Status.SUCCESS:
            counters["fills_completed"] += moves
            return
        if state.status is Status.CANCELLED:
            counters["fills_cancelled"] += moves
        else:
            counters["fills_failed"] += 1
        if state.cmd.inputs:
            self.scheduler.replicas.inflight.discard((state.cmd.inputs[0], state.cmd.device_id))

    def _on_command(self, state: CmdState, batch: _Batch, op: OpNode, is_execute: bool,
                    is_final: bool) -> None:  # fmt: skip
        if state.status is not Status.SUCCESS:
            self.meter.fail(state)
            return
        record = self._records.get(op.op_id)
        now = self.sim.now
        if is_execute:
            if record is not None:
                record.started = state.started if record.started is None else min(record.started, state.started)
            self._op_pending[op.op_id] -= 1
            if self._op_pending[op.op_id] == 0:
                if record is not None:
                    record.finished = state.finished
                if self.payload is not None:
                    self.payload.execute(op, self._plans[op.op_id])
        elif record is not None and isinstance(state.meta, tuple) and state.meta[0] == "in":
            record.inputs_ready = now if record.inputs_ready is None else max(record.inputs_ready, now)
        if is_final:
            if record is not None:
                record.delivered = now if record.delivered is None else max(record.delivered, now)
            batch.pending.discard(state.cmd.op_id)
            if not batch.pending:
                self.meter.batch_done(batch.est - batch.issued, now - batch.issued)
                self.system.completions.drain()
                self.sim.after(self._sync_s, batch.then)


class StreamGroup:
    """Independent request streams sharing one system (concurrent requests).

    Each stream decodes its own sequence at its own pace; they contend for
    devices and links instead of advancing in lock step as a batch does.
    ``warmup_tokens`` counts tokens per stream; measurement starts once that
    many tokens have completed in total across the group.
    """

    def __init__(
        self,
        system: VirtualSystem,
        program: ModelProgram,
        scheduler: Scheduler,
        routings: list[RoutingSource],
        *,
        tokens: int,
        warmup_tokens: int = 0,
        stagger_s: float = 0.0,
        predictor=None,
        keep_log: bool = False,
    ) -> None:
        if tokens <= warmup_tokens:
            raise ValueError("tokens must exceed warmup_tokens")
        if program.batch != 1:
            raise ValueError("concurrent streams decode one sequence each (program batch must be 1)")
        self.system = system
        self.stagger_s = stagger_s
        # Predictions are settled per layer visit, which is ambiguous across interleaved
        # streams, so speculative prefetching is not used for concurrent streams.
        self.meter = Meter(system, program, scheduler, None,
                           warmup_tokens=warmup_tokens * len(routings))  # fmt: skip
        self.streams = [
            Runtime(system, program, scheduler, routing, tokens=tokens, predictor=predictor,
                    keep_log=keep_log, meter=self.meter, stream=i)
            for i, routing in enumerate(routings)
        ]  # fmt: skip

    def run(self, max_events: int | None = None) -> RunResult:
        t0 = time.perf_counter()
        for i, stream in enumerate(self.streams):
            stream.start(at=i * self.stagger_s)
        self.system.run(max_events=max_events)
        self.meter.wall = time.perf_counter() - t0
        if not all(s.done for s in self.streams) and not self.meter.failures:
            self.meter.failures.append("simulation stalled before every stream completed")
        result = self.meter.result()
        result.streams = len(self.streams)
        return result
