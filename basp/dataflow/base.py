"""Shared vocabulary of the dataflow layer: modes, choices, policies and plans.

A *choice* is a state-independent description of how an operation could run
(where, what moves, which local mapping).  The cost model *resolves* a choice
against the current planning state into a concrete plan with transfers, a
kernel descriptor and timing estimates; the same resolved plan is later turned
into a BSP command stream.
"""

from __future__ import annotations

import enum
from collections.abc import Hashable
from dataclasses import dataclass, field

from basp.core.graph import OpNode
from basp.core.tensor import CompressionFormat
from basp.simulator.compute import Kernel


class DataflowMode(enum.IntEnum):
    WEIGHT_STATIONARY = 1  # run where the weights live; activations move
    ACTIVATION_STATIONARY = 2  # run where the activations live; weights move
    OUTPUT_STATIONARY = 3  # accumulate partial results in place; only reduced results move
    ROW_STATIONARY = 4  # keep weight tiles in local SRAM and reuse them across invocations
    STREAMING = 5  # weights flow source -> link -> decoder -> MAC as one overlapped pipeline
    DISTRIBUTED = 6  # partition one operation across several devices
    PREDICTIVE = 7  # prefetch probable future requirements


ALL_MODES = frozenset(DataflowMode)


@dataclass(frozen=True)
class ShardChoice:
    device: int
    weights: tuple[int, ...] = ()
    macs: float = 0.0
    fraction: float = 1.0  # share of the operation's outputs and non-weight memory traffic
    remote: tuple[tuple[int, int], ...] = ()  # (weight tensor, source device) not resident here

    def source(self, tensor: int) -> int | None:
        for tid, src in self.remote:
            if tid == tensor:
                return src
        return None


@dataclass(frozen=True)
class Choice:
    mode: DataflowMode
    shards: tuple[ShardChoice, ...] = ()
    stream: bool = False  # remote weights are consumed as an overlapped stream, never stored
    retain: bool = False  # remote weights are stored and kept as a replica
    use_sram: bool = False  # serve weights from the SRAM tile cache when present
    accumulate: bool = False  # fold the output into a device-local partial sum
    follow_inputs: bool = False  # weightless op: run wherever its inputs end up

    @property
    def modes(self) -> frozenset[DataflowMode]:
        found = {self.mode}
        if self.accumulate:
            found.add(DataflowMode.OUTPUT_STATIONARY)
        if self.use_sram:
            found.add(DataflowMode.ROW_STATIONARY)
        return frozenset(found)

    def signature(self) -> tuple:
        """Identity of the physical plan, ignoring the mode label."""
        return (self.shards, self.stream, self.retain, self.use_sram, self.accumulate,
                self.follow_inputs)  # fmt: skip


@dataclass(frozen=True)
class CachePolicy:
    """Replica management on cache devices ("adaptive expert caching").

    ``lru`` keeps whatever was fetched last; ``benefit`` only replicates a tensor
    when its expected reuse over ``horizon_visits`` pays for the transfer and it
    is more valuable than the entry it would evict.
    """

    kind: str = "none"  # none | lru | benefit
    devices: tuple[int, ...] = ()
    horizon_visits: float = 64.0
    hysteresis: float = 1.25
    background: bool = True  # replicate off the demand path with leftover bandwidth
    budget_fraction: float = 0.5  # share of each link all background traffic may use together
    max_inflight: int = 2  # concurrent background loads (they split the budget)
    sample: int = 32  # eviction candidates examined per decision (sampled LFU/LRU)

    def __post_init__(self) -> None:
        if self.kind not in ("none", "lru", "benefit"):
            raise ValueError(f"unknown cache policy {self.kind!r}")
        if not 0 < self.budget_fraction <= 1:
            raise ValueError("budget_fraction must be in (0, 1]")


@dataclass(frozen=True)
class PrefetchPolicy:
    """Speculative loading of predicted tensors (predictive dataflow)."""

    lookahead_layers: int = 4
    candidates: int = 12  # predictions examined per layer
    min_probability: float = 0.5
    cancel_mispredicted: bool = True


@dataclass(frozen=True)
class ExecutionPolicy:
    name: str
    modes: frozenset[DataflowMode] = ALL_MODES
    compute_devices: frozenset[int] | None = None  # None: any capable device
    joint_rounds: int = 2  # coordinate-descent passes after the greedy assignment
    #: plan evaluations the refinement may spend on one batch (bounds scheduler work on
    #: wide batches; the greedy assignment is kept once the budget is used up)
    joint_budget: int = 4000
    cache: CachePolicy = field(default_factory=CachePolicy)
    prefetch: PrefetchPolicy | None = None
    decision_latency_s: float = 0.0  # modelled host time per scheduling decision
    #: False: devices chain batches through dependency events alone -- no host round trip
    #: and no scheduling decision between layers.  Only valid for static dataflows, where
    #: every operation's location follows from placement.
    host_in_loop: bool = True
    #: stream each layer's hidden state to every device that may run its experts as soon
    #: as it exists, so the transfer overlaps the routing decision
    eager_activations: bool = False
    #: seconds of completion time the scheduler will trade for one joule (0: time only)
    energy_weight_s_per_j: float = 0.0

    def __post_init__(self) -> None:
        if not self.host_in_loop:
            dynamic = {DataflowMode.ACTIVATION_STATIONARY, DataflowMode.STREAMING,
                       DataflowMode.PREDICTIVE}  # fmt: skip
            if self.cache.kind != "none" or self.prefetch is not None or self.modes & dynamic:
                raise ValueError(
                    "host_in_loop=False needs a static dataflow: no replica cache, no prefetch "
                    "and no mode that moves weights"
                )
        if self.energy_weight_s_per_j < 0:
            raise ValueError("energy_weight_s_per_j must be non-negative")

    def enabled(self, mode: DataflowMode) -> bool:
        return mode in self.modes


@dataclass(frozen=True)
class PathEstimate:
    """T_path = T_queue + T_input + T_kernel + T_output + T_sync (critical shard)."""

    queue: float = 0.0
    input: float = 0.0
    kernel: float = 0.0
    output: float = 0.0
    sync: float = 0.0

    @property
    def total(self) -> float:
        return self.queue + self.input + self.kernel + self.output + self.sync


@dataclass
class RTransfer:
    tensor: Hashable  # tensor id, or ("partial", group, device) for a fused partial sum
    src: int
    dst: int
    nbytes: float
    tag: str  # activation | weights | output
    retain: bool = False
    start: float = 0.0
    arrive: float = 0.0
    after_ops: tuple[int, ...] = ()  # ops of this batch that must finish on ``src`` first
    fmt: CompressionFormat | None = None  # format on the wire (the source copy's format)
    store_fmt: CompressionFormat | None = None  # format of the copy created at ``dst``
    dst_bytes: float | None = None  # bytes written at ``dst`` when transcoding on arrival


@dataclass
class RShard:
    device: int
    inputs: list[RTransfer]
    kernel: Kernel
    start: float
    done: float
    outputs: list[RTransfer]
    evict: list[tuple[int, int]]  # (tensor, device) replicas freed to make room
    local_weights: tuple[int, ...]  # weights read from this device's memory
    transient: tuple[int, ...]  # weights loaded for this kernel only
    partial: Hashable | None = None  # partial-sum key this shard accumulates into
    sram_hit: bool = False
    waits_for: tuple[tuple[Hashable, int], ...] = ()  # (tensor, device) already in flight


@dataclass
class ResolvedPlan:
    op: OpNode
    choice: Choice
    shards: list[RShard]
    issued: float
    finish: float
    estimate: PathEstimate
    link_bytes: float = 0.0
    weight_wait: float = 0.0  # time spent waiting for synchronous weight movement
    energy_j: float = 0.0  # dynamic energy of the plan (MACs and bytes moved)

    @property
    def devices(self) -> tuple[int, ...]:
        return tuple(s.device for s in self.shards)
