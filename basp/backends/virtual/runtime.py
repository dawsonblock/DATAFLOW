"""Event-driven execution of a MoE decoding workload on the virtual system.

Per decoding step the runtime walks the layers: the trunk block of a layer
produces the routing decision, the host then schedules the selected experts,
their reduction and the next trunk block as one batch, emits the command
streams and waits for the batch's completion events.  Routing comes from a
:class:`RoutingSource` -- a recorded/synthetic trace, or real values when a
numeric payload is attached -- and is never altered by the scheduler.
"""

from __future__ import annotations

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
from basp.dataflow.base import DataflowMode, ResolvedPlan
from basp.dataflow.predictive import Prefetcher
from basp.scheduler.scheduler import Scheduler
from basp.simulator.energy import EnergyReport, energy_report
from basp.streams.commands import Command
from basp.streams.protocol import NO_DEVICE, Opcode, Status
from basp.workloads.trace import RoutingStep, RoutingTrace


class RoutingSource(Protocol):
    def select(self, step: int, layer: int) -> RoutingStep: ...


class TraceRouting:
    """Routing decisions replayed from a trace (wrapping around at its end)."""

    def __init__(self, trace: RoutingTrace) -> None:
        self.trace = trace

    def select(self, step: int, layer: int) -> RoutingStep:
        return self.trace.at(step % self.trace.n_steps, layer)


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


class Runtime:
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
    ) -> None:
        if tokens <= warmup_tokens:
            raise ValueError("tokens must exceed warmup_tokens")
        self.system = system
        self.sim = system.sim
        self.program = program
        self.scheduler = scheduler
        self.routing = routing
        self.tokens = tokens
        self.warmup = warmup_tokens
        self.predictor = predictor
        self.prefetcher = prefetcher
        self.payload = payload
        self.keep_log = keep_log
        self.emitter = Emitter(system, program.spec.hidden, program.batch)
        self.graph = Graph()
        self.token_times: list[float] = []
        self.failures: list[str] = []
        self.op_log: list[OpRecord] = []
        self.counters: Counter = Counter()
        self._step_start = 0.0
        self._done = False
        self._plans: dict[int, ResolvedPlan] = {}
        self._op_pending: dict[int, int] = {}
        self._records: dict[int, OpRecord] = {}
        self._err_abs = 0.0
        self._err_signed = 0.0
        self._err_n = 0
        self._stall = 0.0
        self._mark: dict | None = None
        self._wall = 0.0

    # -- driver -------------------------------------------------------------------
    def run(self, max_events: int | None = None) -> RunResult:
        t0 = time.perf_counter()
        self._take_mark()
        self.sim.after(0.0, self._begin_step, 0)
        self.system.run(max_events=max_events)
        self._wall = time.perf_counter() - t0
        if not self._done and not self.failures:
            self.failures.append("simulation stalled before the workload completed")
        return self._result()

    def _begin_step(self, step: int) -> None:
        if step == self.warmup and step > 0:
            self._take_mark()
        self._step_start = self.sim.now
        self.graph = Graph()
        self.emitter.new_step()
        self._plans.clear()
        self._op_pending.clear()
        program = self.program
        x = program.input_tensor(self.graph, step)
        self.system.residency.place(x, program.trunk_device[0], now=self.sim.now)
        if self.payload is not None:
            self.payload.begin_step(step, x)
        trunk = program.trunk(self.graph, step, 0, x)
        self._submit(trunk.ops, lambda: self._after_trunk(step, 0, trunk.outputs), f"s{step}.trunk0")

    def _after_trunk(self, step: int, layer: int, tensors: dict[str, int]) -> None:
        """The router output of ``layer`` reached the host: issue the experts."""
        program = self.program
        if self.payload is not None:
            routing = self.payload.route(step, layer, tensors["logits"])
        else:
            routing = self.routing.select(step, layer)
        selected = set(routing.unique_experts())
        if self.prefetcher is not None:
            for tensor, device in self.prefetcher.reveal(layer, selected):
                loading = self.system.loading.get((tensor, device))
                if loading is not None:
                    cancel = Command(self.system.new_id(), Opcode.CANCEL, NO_DEVICE, arg0=loading)
                    self.system.submit(cancel, tag="cancel")
                    self.counters["prefetch_cancelled"] += 1
        if self.predictor is not None:
            self.predictor.observe_step(layer, routing)
        experts = program.experts(self.graph, step, layer, tensors["h"], routing)
        reduce = program.reduce(self.graph, step, layer, experts.outputs, tensors["resid"])
        ops = experts.ops + reduce.ops
        x = reduce.outputs["x"]
        if layer + 1 < program.spec.n_layers:
            trunk = program.trunk(self.graph, step, layer + 1, x)
            ops += trunk.ops
            then = lambda: self._after_trunk(step, layer + 1, trunk.outputs)
        else:
            head = program.lm_head(self.graph, step, x)
            ops += head.ops
            then = lambda: self._end_step(step)
        self.counters["expert_ops"] += len(experts.ops) if program.fused else len(experts.outputs)
        self._submit(ops, then, f"s{step}.l{layer}", prefetch_after=layer)

    def _end_step(self, step: int) -> None:
        now = self.sim.now
        self.token_times.append(now - self._step_start)
        residency, registry = self.system.residency, self.system.registry
        if self.payload is not None:
            self.payload.end_step(step)
        for tensor in self.program.temporaries(self.graph) + self.emitter.partials:
            if tensor in registry:
                if residency.copies(tensor):
                    residency.drop(tensor)
                registry.unregister(tensor)
        for layer, kv in enumerate(self.program.kv):
            residency.write(kv, self.program.trunk_device[layer])  # KV append = versioned write
        if step + 1 < self.tokens:
            self._begin_step(step + 1)
        else:
            self._done = True

    # -- batches ------------------------------------------------------------------
    def _submit(self, ops: list[OpNode], then: Callable[[], None], label: str,
                prefetch_after: int | None = None) -> None:  # fmt: skip
        system = self.system
        st = system.plan_state()
        plans, final = self.scheduler.schedule(ops, self.graph, st)
        now = self.sim.now
        batch = _Batch(now, max(p.finish for p in plans), then, label)
        emitted = []
        for plan in plans:
            self._account_plan(plan)
            ep = self.emitter.emit_plan(plan)
            emitted.append(ep)
            batch.pending.update(ep.final)
            self._plans[plan.op.op_id] = plan
            self._op_pending[plan.op.op_id] = len(ep.executes)
            if self.keep_log:
                record = OpRecord(
                    plan.op.op_id, plan.op.name, plan.op.opcode, plan.devices,
                    tuple(sorted(m.name.lower() for m in plan.choice.modes)), now, plan.finish,
                    deps=self.graph.deps(plan.op.op_id),
                )  # fmt: skip
                self._records[plan.op.op_id] = record
                self.op_log.append(record)
        fills = self.scheduler.propose_fills(plans, final)
        if self.prefetcher is not None and prefetch_after is not None:
            fills += self.prefetcher.propose(prefetch_after, final)
        fill_cmds = []
        for action in fills:
            eta = now + action.nbytes / action.max_rate
            fill_cmds.extend(self.emitter.emit_fill(action, eta))
            self.counters["fills"] += 1
            self.counters["fill_evictions"] += len(action.evict)

        def issue() -> None:
            finals = batch.pending.copy()
            for ep in emitted:
                executes = set(ep.executes)
                for e in ep.commands:
                    cid = e.cmd.op_id
                    hook = None
                    if cid in finals or cid in executes:
                        hook = lambda s, b=batch, op=ep.plan.op, x=cid in executes: (
                            self._on_command(s, b, op, x)
                        )
                    else:
                        hook = self._on_other
                    system.submit(e.cmd, kernel=e.kernel, nbytes=e.nbytes, tag=e.tag, reads=e.reads,
                                  writes=e.writes, max_rate=e.max_rate, piece=e.piece,
                                  dst_bytes=e.dst_bytes, meta=e.meta, on_complete=hook)  # fmt: skip
            for e in fill_cmds:
                system.submit(e.cmd, nbytes=e.nbytes, tag=e.tag, reads=e.reads,
                              dst_bytes=e.dst_bytes, meta=e.meta, on_complete=self._on_fill)  # fmt: skip
            system.completions.drain()

        delay = self.scheduler.policy.decision_latency_s
        if delay > 0:
            self.sim.after(delay, issue)
        else:
            issue()

    def _account_plan(self, plan: ResolvedPlan) -> None:
        op = plan.op
        self._stall += plan.weight_wait
        if plan.weight_wait > 0:
            self.counters["stalled_ops"] += 1
        if op.opcode is OpCode.EXPERT or op.attrs.get("part") == "down":
            # An expert run that found its weights already resident on a GPU.
            shard = plan.shards[0]
            on_gpu = self.system.topology.devices[shard.device].kind is DeviceKind.GPU
            moved = plan.choice.stream or any(t.tag == "weights" for t in shard.inputs)
            if on_gpu and not moved:
                self.counters["cache_hits"] += 1

    def _on_other(self, state: CmdState) -> None:
        if state.status is not Status.SUCCESS:
            self._fail(state)

    def _on_fill(self, state: CmdState) -> None:
        """Background replication and prefetch are best effort: a failure costs nothing but
        the opportunity, so it is counted rather than treated as a failed run."""
        moves = state.cmd.opcode in (Opcode.TRANSFER, Opcode.PREFETCH)
        if state.status is Status.SUCCESS:
            self.counters["fills_completed"] += moves
            return
        if state.status is Status.CANCELLED:
            self.counters["fills_cancelled"] += moves
        else:
            self.counters["fills_failed"] += 1
        if state.cmd.inputs:
            self.scheduler.replicas.inflight.discard((state.cmd.inputs[0], state.cmd.device_id))

    def _on_command(self, state: CmdState, batch: _Batch, op: OpNode, is_execute: bool) -> None:
        if state.status is not Status.SUCCESS:
            self._fail(state)
            return
        if is_execute:
            record = self._records.get(op.op_id)
            if record is not None:
                record.started = state.started if record.started is None else min(record.started, state.started)
            self._op_pending[op.op_id] -= 1
            if self._op_pending[op.op_id] == 0:
                if record is not None:
                    record.finished = state.finished
                if self.payload is not None:
                    self.payload.execute(op, self._plans[op.op_id])
        cid = state.cmd.op_id
        if cid in batch.pending:
            batch.pending.discard(cid)
            if not batch.pending:
                actual = self.sim.now - batch.issued
                est = batch.est - batch.issued
                if actual > 0:
                    self._err_abs += abs(est - actual) / actual
                    self._err_signed += (est - actual) / actual
                    self._err_n += 1
                self.system.completions.drain()
                self.sim.after(self.system.topology.sync_latency_s, batch.then)

    def _fail(self, state: CmdState) -> None:
        cmd = state.cmd
        self.failures.append(
            f"command {cmd.op_id} ({cmd.opcode.name} on device {cmd.device_id}) ended {state.status.name}"
        )

    # -- results ------------------------------------------------------------------
    def _take_mark(self) -> None:
        self.system.net.settle()
        self._mark = {
            "time": self.sim.now,
            "served": {k: r.served for k, r in self.system.res.items()},
            "tags": {k: dict(r.by_tag) for k, r in self.system.res.items()},
            "busy": {d: ex.busy_time for d, ex in self.system.executors.items()},
            "kernels": {d: ex.kernels_run for d, ex in self.system.executors.items()},
            "wait": {d: ex.wait_time for d, ex in self.system.executors.items()},
            "counters": Counter(self.counters),
            "stall": self._stall,
            "tokens": len(self.token_times),
            "sched": Counter(self.scheduler.stats),
            "sched_wall": self.scheduler.wall_s,
            "modes": Counter(self.scheduler.mode_counts),
            "log": len(self.system.log),
            "prefetch": Counter(self.prefetcher.stats) if self.prefetcher else Counter(),
        }

    def _result(self) -> RunResult:
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
        tokens = len(self.token_times) - mark["tokens"]
        expert_ops = counters.get("expert_ops", 0)
        decisions = sched.get("batches", 0) * self.scheduler.policy.decision_latency_s
        overhead = decisions / elapsed if elapsed > 0 else 0.0
        return RunResult(
            policy=self.scheduler.policy.name,
            tokens=tokens,
            batch=self.program.batch,
            elapsed_s=elapsed,
            token_times=self.token_times[mark["tokens"]:],
            link_bytes=link_bytes,
            energy=energy_report(topo, delta, elapsed),
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
            stall_s=self._stall - mark["stall"],
            weight_transfer_s=weight_s,
            estimate_error={
                "mean_abs_rel": self._err_abs / self._err_n if self._err_n else 0.0,
                "mean_signed_rel": self._err_signed / self._err_n if self._err_n else 0.0,
                "batches": self._err_n,
            },
            system={
                "events": self.sim.events_processed,
                "commands": system.stats.get("submitted", 0),
                "backpressure": system.stats.get("backpressure", 0),
                "dma_wait_s": dma_wait,
                "dma_max_waiting": dma_max,
                "completion_overflow": system.stats.get("completion_overflow", 0),
                "wall_s": self._wall,
            },
            violations=list(system.violations),
            failures=list(self.failures),
            op_log=self.op_log,
        )
