"""Small hand-built systems for studying single scheduling decisions.

:class:`MicroSystem` wires registry, residency, scheduler, virtual system and
emitter around an arbitrary topology so individual operations can be planned,
executed and timed without a model.  It backs the dataflow-mode study and the
tests that compare analytical estimates with simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

from basp.backends.virtual.system import VirtualSystem
from basp.compiler.emit import Emitter
from basp.core.graph import Graph, OpCode, OpNode
from basp.core.tensor import CompressionFormat, DType, FormatTable, TensorKind
from basp.core.topology import Topology
from basp.dataflow.base import ExecutionPolicy, ResolvedPlan
from basp.memory.replication import ReplicaManager
from basp.memory.residency import ResidencyMap
from basp.memory.tensor_registry import TensorRegistry
from basp.scheduler.scheduler import Scheduler
from basp.streams.protocol import Status


@dataclass
class MicroResult:
    plans: list[ResolvedPlan]
    issued: float
    estimate_s: float  # scheduler's estimate of the batch completion time
    actual_s: float  # simulated completion time
    link_bytes: float  # bytes carried by links during the batch (each hop counts)
    failures: list[str]

    @property
    def plan(self) -> ResolvedPlan:
        return self.plans[0]


class MicroSystem:
    def __init__(self, topology: Topology, policy: ExecutionPolicy, *,
                 formats: FormatTable | None = None, hidden: int = 1, batch: int = 1) -> None:  # fmt: skip
        self.topology = topology
        self.policy = policy
        self.registry = TensorRegistry(formats)
        self.residency = ResidencyMap(self.registry, topology)
        self.replicas = None
        self.frequency: dict[int, float] = {}
        if policy.cache.kind != "none":
            self.replicas = ReplicaManager(policy.cache, self.residency, self.registry, topology,
                                           freq_fn=lambda t: self.frequency.get(t, 0.0))  # fmt: skip
        self.scheduler = Scheduler(topology, self.registry, self.residency, policy,
                                   replicas=self.replicas)  # fmt: skip
        budget = policy.cache.budget_fraction if self.replicas is not None else None
        self.system = VirtualSystem(topology, self.registry, self.residency, replicas=self.replicas,
                                    background_budget=budget)  # fmt: skip
        self.emitter = Emitter(self.system, hidden, batch)
        self.graph = Graph()
        self._next_op = 1

    # -- construction -------------------------------------------------------------
    def weight(self, name: str, n_weights: int, device: int, *, fmt: CompressionFormat | None = None,
               stored_as: CompressionFormat | None = None, pinned: bool = False,
               kind: TensorKind = TensorKind.EXPERT_WEIGHT) -> int:  # fmt: skip
        """Register a weight tensor (canonical format ``fmt``) and place it on ``device``."""
        fmt = fmt or CompressionFormat.INT4
        spec = self.registry.add(name, kind, (n_weights,), DType.BYTES,
                                 self.registry.formats.weight_bytes(n_weights, fmt), fmt=fmt,
                                 n_weights=n_weights)  # fmt: skip
        self.graph.add_tensor(spec)
        self.residency.place(spec.tensor_id, device, pinned=pinned, fmt=stored_as)
        return spec.tensor_id

    def replicate(self, tensor: int, device: int, *, fmt: CompressionFormat | None = None) -> None:
        self.residency.place(tensor, device, owner=False, fmt=fmt)
        if self.replicas is not None:
            self.replicas.track(tensor, device)

    def activation(self, name: str, nbytes: int, device: int | None = None) -> int:
        spec = self.registry.add(name, TensorKind.ACTIVATION, (max(nbytes // 2, 1),), DType.FP16, nbytes)
        self.graph.add_tensor(spec)
        if device is not None:
            self.residency.place(spec.tensor_id, device, now=self.system.sim.now)
        return spec.tensor_id

    def op(self, opcode: OpCode, name: str, *, inputs=(), weights=(), outputs=(), macs: float = 0,
           **kwargs) -> OpNode:  # fmt: skip
        node = OpNode(self._next_op, opcode, name, tuple(inputs), tuple(weights), tuple(outputs),
                      macs=int(macs), **kwargs)  # fmt: skip
        self._next_op += 1
        return self.graph.add_op(node)

    # -- execution ----------------------------------------------------------------
    def run(self, ops: list[OpNode]) -> MicroResult:
        """Schedule ``ops`` as one batch, execute it and report estimate versus simulation."""
        system = self.system
        system.net.settle()
        before = sum(r.served for r in system.interconnect.link_res.values())
        st = system.plan_state()
        plans, _final = self.scheduler.schedule(ops, self.graph, st)
        issued = system.sim.now
        finals: set[int] = set()
        done_at = [issued]
        failures: list[str] = []

        def on_complete(state) -> None:
            if state.status is not Status.SUCCESS:
                failures.append(f"command {state.cmd.op_id} {state.cmd.opcode.name}: {state.status.name}")
            if state.cmd.op_id in finals:
                done_at[0] = max(done_at[0], state.finished)

        emitted = [self.emitter.emit_plan(plan) for plan in plans]
        for ep in emitted:
            finals.update(ep.final)
        for ep in emitted:
            for e in ep.commands:
                system.submit(e.cmd, kernel=e.kernel, nbytes=e.nbytes, tag=e.tag, reads=e.reads,
                              writes=e.writes, piece=e.piece, dst_bytes=e.dst_bytes, meta=e.meta,
                              on_complete=on_complete)  # fmt: skip
        system.run()
        system.completions.drain()
        failures.extend(system.violations)
        after = sum(r.served for r in system.interconnect.link_res.values())
        return MicroResult(plans, issued, max(p.finish for p in plans) - issued,
                           done_at[0] - issued, after - before, failures)  # fmt: skip
