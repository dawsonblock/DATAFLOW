"""Deterministic cost-based scheduler with joint placement refinement.

Scheduling works on a *batch*: the operations that are ready (or about to be)
together with their dependents, e.g. the selected experts of a layer, their
weighted reduction and the next trunk block.

1. Operations are ordered by upward rank (critical path first).
2. A greedy earliest-finish-time pass picks the best choice per operation
   against the evolving planning state.
3. Coordinate descent then re-evaluates the *whole batch* with one operation's
   choice swapped at a time and keeps any change that lowers the batch
   completion time (ties: less interconnect traffic).  This is what stops two
   operations from each "optimally" choosing the same link or device.

Model behaviour is never altered: the scheduler chooses where and how selected
operations run, not which experts are selected.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence

from basp.core.graph import Graph, OpNode
from basp.core.topology import Topology
from basp.dataflow import adaptive
from basp.dataflow.base import Choice, DataflowMode, ExecutionPolicy, PathEstimate, ResolvedPlan
from basp.dataflow.context import PlanContext
from basp.memory.replication import ReplicaAction, ReplicaManager
from basp.memory.residency import ResidencyMap
from basp.memory.tensor_registry import TensorRegistry
from basp.scheduler.congestion import PlanState
from basp.scheduler.cost_model import CostModel
from basp.scheduler.critical_path import priority_order

_EPS = 1e-12


class SchedulingError(RuntimeError):
    pass


class Scheduler:
    def __init__(
        self,
        topology: Topology,
        registry: TensorRegistry,
        residency: ResidencyMap,
        policy: ExecutionPolicy,
        *,
        replicas: ReplicaManager | None = None,
    ) -> None:
        self.topology = topology
        self.registry = registry
        self.residency = residency
        self.policy = policy
        self.replicas = replicas
        self.cost = CostModel(topology, registry, residency, replicas=replicas,
                              compute_devices=policy.compute_devices)  # fmt: skip
        self._fast_compute = max(d.compute_rate for d in topology.devices.values())
        self._fast_mem = max(d.mem_bandwidth for d in topology.devices.values())
        self._best_cache: dict[tuple, float] = {}
        self._energy_weight = policy.energy_weight_s_per_j
        self.stats: Counter = Counter()
        self.mode_counts: Counter = Counter()
        self.wall_s = 0.0

    # -- public API ---------------------------------------------------------------
    def schedule(self, ops: Sequence[OpNode], graph: Graph, st: PlanState) -> tuple[list[ResolvedPlan], PlanState]:
        """Plan a batch.  Returns the resolved plans in issue order and the final state."""
        t0 = time.perf_counter()
        by_id = {op.op_id: op for op in ops}
        order = [by_id[i] for i in priority_order(graph, by_id, lambda i: self._nominal(by_id[i]))]
        st.locked = frozenset(
            (w, copy.device) for op in ops for w in op.weights for copy in self.residency.copies(w)
        )
        ctx = PlanContext(self.topology, self.registry, self.residency, self.policy, self.cost, st)
        options: dict[int, list[Choice]] = {}
        for op in order:
            choices, relaxed = adaptive.candidates(op, ctx)
            if not choices:
                raise SchedulingError(f"no execution path for {op.name!r} under policy {self.policy.name!r}")
            self.stats["relaxed"] += relaxed
            options[op.op_id] = choices

        assign = self._greedy(order, options, st)
        if self.policy.joint_rounds > 0 and len(order) > 1:
            assign = self._refine(order, options, assign, st)

        final = st.fork()
        plans: list[ResolvedPlan] = []
        sinks = set(graph.sinks(by_id))
        for op in order:
            plan = self.cost.resolve(op, assign[op.op_id], final)
            if plan is None:  # cannot happen: the assignment was evaluated on the same state
                raise SchedulingError(f"plan for {op.name!r} became infeasible")
            if op.op_id in sinks:
                est = plan.estimate
                plan.estimate = PathEstimate(est.queue, est.input, est.kernel, est.output, self.cost.sync_s)
            plans.append(plan)
            for mode in plan.choice.modes:
                self.mode_counts[mode] += 1
        self.stats["batches"] += 1
        self.stats["ops"] += len(order)
        self.wall_s += time.perf_counter() - t0
        return plans, final

    def propose_fills(self, plans: Sequence[ResolvedPlan], st: PlanState) -> list[ReplicaAction]:
        """Reactive caching: replicate weights whose reuse would repay the transfer."""
        cache = self.policy.cache
        if self.replicas is None or cache.kind != "benefit" or not cache.background:
            return []
        actions: list[ReplicaAction] = []
        for plan in plans:
            op = plan.op
            if len(op.weights) != 1:
                continue
            tensor = op.weights[0]
            est = plan.estimate
            current = est.input + est.kernel + est.output
            for dst in cache.devices:
                if st.avail(tensor, dst) is not None or not self.cost.allowed(dst, op):
                    continue
                local = self.cost.local_time(op, dst)
                if local is None:
                    continue
                action = self.replicas.propose(
                    tensor, dst, st, self.cost, value=self.replicas.value(tensor),
                    saving_s=current - local, speculative=False, tag="replicate",
                )  # fmt: skip
                if action is not None:
                    actions.append(action)
        return actions

    def best_time(self, op: OpNode, now: float) -> float:
        """Idle-system time of the best currently available path for ``op`` (memoised)."""
        key = (
            op.opcode, op.macs, op.output_device, op.pinned_device,
            tuple((w, tuple((c.device, c.fmt) for c in self.residency.holders(w))) for w in op.weights),
        )  # fmt: skip
        if key not in self._best_cache:
            st = PlanState(now, self.residency)
            home = op.output_device if op.output_device is not None else self.topology.host
            for tid in op.inputs:
                if not st.sources(tid):
                    st.add_arrival(tid, home, now)  # a what-if input waiting on its home device
            ctx = PlanContext(self.topology, self.registry, self.residency, self.policy, self.cost, st)
            best = float("inf")
            choices, _ = adaptive.candidates(op, ctx)
            for choice in choices:
                if choice.retain:
                    continue
                plan = self.cost.resolve(op, choice, st.fork())
                if plan is not None:
                    best = min(best, plan.finish - now)
            if len(self._best_cache) > 4096:
                self._best_cache.clear()
            self._best_cache[key] = best
        return self._best_cache[key]

    # -- internals ----------------------------------------------------------------
    def _nominal(self, op: OpNode) -> float:
        weight_bytes = sum(self.registry.spec(w).nbytes for w in op.weights) * op.weight_fraction
        return op.macs / self._fast_compute + weight_bytes / self._fast_mem + 1e-9

    def _greedy(self, order, options, st: PlanState) -> dict[int, Choice]:
        assign: dict[int, Choice] = {}
        state = st
        for op in order:
            best = None
            for index, choice in enumerate(options[op.op_id]):
                trial = state.fork()
                plan = self.cost.resolve(op, choice, trial)
                self.stats["evaluated"] += 1
                if plan is None:
                    continue
                key = (plan.finish + self._energy_weight * plan.energy_j, plan.link_bytes, index)
                if best is None or key < best[0]:
                    best = (key, choice, trial)
            if best is None:
                raise SchedulingError(f"no feasible execution path for {op.name!r}")
            assign[op.op_id] = best[1]
            state = best[2]
        return assign

    def _objective(self, order, assign, st: PlanState) -> tuple[float, float, float] | None:
        state = st.fork()
        makespan = total = energy = 0.0
        for op in order:
            plan = self.cost.resolve(op, assign[op.op_id], state)
            self.stats["evaluated"] += 1
            if plan is None:
                return None
            makespan = max(makespan, plan.finish)
            total += plan.finish
            energy += plan.energy_j
        return makespan + self._energy_weight * energy, state.link_bytes, total

    @staticmethod
    def _better(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
        if a[0] < b[0] - _EPS:
            return True
        if a[0] > b[0] + _EPS:
            return False
        if a[1] < b[1] - 0.5:
            return True
        if a[1] > b[1] + 0.5:
            return False
        return a[2] < b[2] - _EPS

    def _refine(self, order, options, assign, st: PlanState) -> dict[int, Choice]:
        best = self._objective(order, assign, st)
        if best is None:
            return assign
        spent_at_start = self.stats["evaluated"]
        budget = self.policy.joint_budget
        for _ in range(self.policy.joint_rounds):
            improved = False
            for op in order:
                current = assign[op.op_id]
                for choice in options[op.op_id]:
                    if choice is current:
                        continue
                    if self.stats["evaluated"] - spent_at_start + len(order) > budget:
                        self.stats["joint_budget_exhausted"] += 1
                        return assign
                    trial = dict(assign)
                    trial[op.op_id] = choice
                    score = self._objective(order, trial, st)
                    if score is not None and self._better(score, best):
                        best, assign, current, improved = score, trial, choice, True
                        self.stats["joint_improvements"] += 1
            if not improved:
                break
        return assign


def mode_name(mode: DataflowMode) -> str:
    return mode.name.lower()
