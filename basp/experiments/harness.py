"""Assemble a simulated system from a topology, a model and an execution policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from basp.backends.virtual.runtime import (
    RoutingSource,
    RunResult,
    Runtime,
    StreamGroup,
    TraceRouting,
)
from basp.backends.virtual.system import VirtualSystem
from basp.compiler.lower import ModelProgram
from basp.core.tensor import FormatTable
from basp.core.topology import Topology
from basp.dataflow.base import ExecutionPolicy
from basp.dataflow.predictive import Prefetcher
from basp.memory.replication import ReplicaManager
from basp.memory.residency import ResidencyMap
from basp.memory.tensor_registry import TensorRegistry
from basp.scheduler.placement import PlacementReport, place_auxiliary
from basp.scheduler.predictor import ExpertPredictor
from basp.scheduler.scheduler import Scheduler
from basp.workloads.model import MoEModelSpec
from basp.workloads.trace import RoutingTrace

Placer = Callable[[ModelProgram, ResidencyMap], PlacementReport]


@dataclass
class Setup:
    topology: Topology
    registry: TensorRegistry
    residency: ResidencyMap
    program: ModelProgram
    policy: ExecutionPolicy
    placement: PlacementReport
    predictor: ExpertPredictor
    replicas: ReplicaManager | None
    scheduler: Scheduler
    system: VirtualSystem
    prefetcher: Prefetcher | None

    def run(self, routing: RoutingSource | RoutingTrace, *, tokens: int, warmup_tokens: int = 0,
            payload=None, keep_log: bool = False, max_events: int | None = None) -> RunResult:  # fmt: skip
        if isinstance(routing, RoutingTrace):
            routing = TraceRouting(routing)
        runtime = Runtime(
            self.system, self.program, self.scheduler, routing, tokens=tokens,
            warmup_tokens=warmup_tokens, predictor=self.predictor, prefetcher=self.prefetcher,
            payload=payload, keep_log=keep_log,
        )  # fmt: skip
        return runtime.run(max_events=max_events)

    def run_concurrent(self, trace: RoutingTrace, *, streams: int, tokens: int,
                       warmup_tokens: int = 0, stagger_s: float = 0.0, keep_log: bool = False,
                       max_events: int | None = None) -> RunResult:  # fmt: skip
        """Decode ``streams`` independent sequences at once (concurrent requests).

        Stream ``i`` replays batch row ``i`` of ``trace``; the setup must have been
        built with ``batch=1``.  Speculative prefetching is not used.
        """
        if streams > trace.batch:
            raise ValueError(f"trace has {trace.batch} rows, cannot drive {streams} streams")
        group = StreamGroup(
            self.system, self.program, self.scheduler,
            [TraceRouting(trace, row=i) for i in range(streams)], tokens=tokens,
            warmup_tokens=warmup_tokens, stagger_s=stagger_s, predictor=self.predictor,
            keep_log=keep_log,
        )  # fmt: skip
        return group.run(max_events=max_events)


def build(
    topology: Topology,
    spec: MoEModelSpec,
    policy: ExecutionPolicy,
    place: Placer,
    *,
    batch: int = 1,
    context_len: int = 0,
    max_new_tokens: int = 0,
    head_shards: int = 1,
    expert_shards: int = 1,
    fused: bool = True,
    formats: FormatTable | None = None,
    prior: RoutingTrace | None = None,
    seed: int = 0,
    prompt: bool = False,
) -> Setup:
    """Create registry, program, placement, scheduler and virtual system.

    ``prior`` is a routing trace whose statistics are known before execution
    (profiling data): it initialises the expert predictor that drives caching
    and prefetching.  Evaluate on a different, held-out trace.
    """
    registry = TensorRegistry(formats)
    program = ModelProgram(spec, registry, batch=batch, context_len=context_len,
                           max_new_tokens=max_new_tokens, head_shards=head_shards,
                           expert_shards=expert_shards, fused=fused, prompt=prompt)  # fmt: skip
    residency = ResidencyMap(registry, topology)
    placement = place(program, residency)
    place_auxiliary(program, residency, placement)  # embedding table and final norm
    predictor = ExpertPredictor(spec.n_layers, spec.n_experts)
    if prior is not None:
        predictor.fit(prior)
        predictor.reset_streams()

    def frequency(tensor: int) -> float:
        where = program.expert_of.get(tensor)
        return predictor.frequency(*where) if where is not None else 0.0

    replicas = None
    if policy.cache.kind != "none":
        replicas = ReplicaManager(policy.cache, residency, registry, topology, freq_fn=frequency,
                                  seed=seed)  # fmt: skip
    scheduler = Scheduler(topology, registry, residency, policy, replicas=replicas)
    budget = policy.cache.budget_fraction if replicas is not None else None
    system = VirtualSystem(topology, registry, residency, replicas=replicas,
                           background_budget=budget)  # fmt: skip
    prefetcher = None
    if policy.prefetch is not None and replicas is not None:
        prefetcher = Prefetcher(
            policy.prefetch, predictor, replicas, scheduler, n_layers=spec.n_layers,
            expert_tensor=lambda layer, expert: program.expert_w[layer][expert],
            probe_op=program.probe_expert,
        )  # fmt: skip
    return Setup(topology, registry, residency, program, policy, placement, predictor, replicas,
                 scheduler, system, prefetcher)  # fmt: skip
