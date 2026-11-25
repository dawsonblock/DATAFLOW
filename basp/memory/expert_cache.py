"""Expert caching policies for a fast device with too little memory for every expert, and a
harness that compares them on a routing trace.

Setting: a GPU holds a cache of experts; the rest live in host memory.  When the router
selects an expert that is not cached, the runtime either executes it on the CPU (``T_cpu``)
or migrates it to the GPU (``T_migration``) and executes it there (``T_gpu``), inserting it
into the cache.  Policies differ in which experts they admit and which they evict:

* **LRU** -- admit every miss, evict the least recently used expert;
* **LFU** -- admit every miss, evict the least frequently used (decayed counts);
* **cost-aware** -- value of caching expert *e*::

      V_e = (N_e * (T_cpu - T_gpu) - T_migration) / M_e

  with ``N_e`` the expected uses over a horizon (decayed frequency x horizon) and ``M_e`` its
  size; admit a miss only if its value is positive and exceeds the value of what it would
  displace, otherwise execute it on the CPU;
* **predictive** -- the cost-aware value with ``N_e`` from a predictor that uses the measured
  probability that an expert selected in one step is selected again in the next.

The harness charges the critical path only (no overlap of migration with computation, for
every policy alike) and reports hits, misses, migrations, bytes moved, CPU executions and
total expert time.  It is a model (evidence E1): the costs come from a profile, not from a
measurement of the runtime.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

Key = tuple[int, int]  # (layer, expert)


@dataclass(frozen=True)
class ExpertCost:
    """Per-expert execution and migration costs (seconds) and size (bytes)."""

    bytes: int
    t_gpu: float
    t_cpu: float
    t_migration: float

    def __post_init__(self) -> None:
        if self.bytes <= 0 or min(self.t_gpu, self.t_cpu, self.t_migration) < 0:
            raise ValueError("expert size must be positive and times non-negative")

    @classmethod
    def from_rates(cls, bytes_: int, *, gpu_bandwidth: float, cpu_bandwidth: float, link_bandwidth: float,
                   link_latency_s: float = 0.0, launch_s: float = 0.0) -> ExpertCost:  # fmt: skip
        """Memory-bound decoding: each use reads the expert's weights once."""
        return cls(bytes_, launch_s + bytes_ / gpu_bandwidth, bytes_ / cpu_bandwidth,
                   link_latency_s + bytes_ / link_bandwidth)  # fmt: skip


@dataclass
class Stats:
    """Online statistics every policy may read (updated after each step)."""

    decay: float = 0.98
    freq: dict[Key, float] = field(default_factory=dict)  # decayed fraction of steps used
    last_use: dict[Key, int] = field(default_factory=dict)
    last_step: set[Key] = field(default_factory=set)
    repeats: float = 0.0  # decayed count of (used at t and at t+1)
    opportunities: float = 0.0  # decayed count of (used at t)
    step: int = 0

    def observe(self, used: set[Key]) -> None:
        d = self.decay
        for key in list(self.freq):
            self.freq[key] *= d
        for key in used:
            self.freq[key] = self.freq.get(key, 0.0) + (1 - d)
            self.last_use[key] = self.step
        self.repeats = self.repeats * d + len(used & self.last_step)
        self.opportunities = self.opportunities * d + len(self.last_step)
        self.last_step = set(used)
        self.step += 1

    @property
    def repeat_rate(self) -> float:
        return self.repeats / self.opportunities if self.opportunities else 0.0


class Policy:
    name = "policy"

    def admit(self, key: Key, cost: ExpertCost, victims: list[Key], stats: Stats, costs) -> bool:
        return True

    def eviction_order(self, cached: Iterable[Key], stats: Stats, costs) -> list[Key]:
        raise NotImplementedError


class LRU(Policy):
    name = "LRU"

    def eviction_order(self, cached, stats, costs):
        return sorted(cached, key=lambda k: (stats.last_use.get(k, -1), k))


class LFU(Policy):
    name = "LFU"

    def eviction_order(self, cached, stats, costs):
        return sorted(cached, key=lambda k: (stats.freq.get(k, 0.0), stats.last_use.get(k, -1), k))


class CostAware(Policy):
    name = "cost-aware"

    def __init__(self, horizon: int = 32) -> None:
        self.horizon = horizon

    def expected_uses(self, key: Key, stats: Stats) -> float:
        return stats.freq.get(key, 0.0) * self.horizon

    def value(self, key: Key, cost: ExpertCost, stats: Stats) -> float:
        return (self.expected_uses(key, stats) * (cost.t_cpu - cost.t_gpu) - cost.t_migration) / cost.bytes

    def admit(self, key, cost, victims, stats, costs):
        mine = self.value(key, cost, stats)
        if mine <= 0:
            return False
        return all(self.value(v, costs(v), stats) < mine for v in victims)

    def eviction_order(self, cached, stats, costs):
        return sorted(cached, key=lambda k: (self.value(k, costs(k), stats), k))


class Predictive(CostAware):
    """Cost-aware value with ``N_e`` predicted from the next-step repeat probability."""

    name = "predictive"

    def expected_uses(self, key: Key, stats: Stats) -> float:
        f = min(stats.freq.get(key, 0.0), 0.999)
        r = stats.repeat_rate
        # P(used next | used now) = r; P(used next | not used now) from total probability
        p_next = r if key in stats.last_step else max(0.0, f * (1 - r) / (1 - f))
        # the next step is predicted; later steps revert to the long-run frequency
        return p_next + f * (self.horizon - 1)


POLICIES = {"LRU": LRU, "LFU": LFU, "cost-aware": CostAware, "predictive": Predictive}


@dataclass
class CacheResult:
    policy: str
    capacity_bytes: int
    activations: int = 0
    hits: int = 0
    migrations: int = 0
    cpu_executions: int = 0
    bytes_migrated: int = 0
    time_s: float = 0.0
    peak_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.activations if self.activations else 0.0

    def to_json(self) -> dict:
        return {"policy": self.policy, "capacity_bytes": self.capacity_bytes, "activations": self.activations,
                "hits": self.hits, "hit_rate": self.hit_rate, "migrations": self.migrations,
                "cpu_executions": self.cpu_executions, "bytes_migrated": self.bytes_migrated,
                "time_s": self.time_s, "peak_bytes": self.peak_bytes}  # fmt: skip


def simulate(steps: Iterable[Iterable[Key]], policy: Policy, capacity_bytes: int, costs,
             *, warm: Iterable[Key] = ()) -> CacheResult:  # fmt: skip
    """Run ``policy`` over a sequence of steps (each: the (layer, expert) pairs selected).

    ``costs(key) -> ExpertCost``.  ``warm`` experts start cached (if they fit).  Within a step
    the experts in use are protected from eviction (they are referenced), so a step never
    evicts what it is about to execute.
    """
    stats = Stats()
    cached: dict[Key, int] = {}
    result = CacheResult(policy.name, capacity_bytes)
    used_bytes = 0
    for key in warm:
        size = costs(key).bytes
        if used_bytes + size <= capacity_bytes:
            cached[key] = size
            used_bytes += size
    for step in steps:
        used = set(step)
        # statistics change only between steps, so one eviction order serves the whole step
        order: list[Key] | None = None
        cursor = 0
        for key in sorted(used):
            cost = costs(key)
            result.activations += 1
            if key in cached:
                result.hits += 1
                result.time_s += cost.t_gpu
                continue
            if cost.bytes > capacity_bytes:
                result.cpu_executions += 1
                result.time_s += cost.t_cpu
                continue
            # what would have to go to make room (never an expert of the current step)
            free = capacity_bytes - used_bytes
            victims: list[Key] = []
            j = cursor
            if free < cost.bytes:
                if order is None:
                    order = [k for k in policy.eviction_order(list(cached), stats, costs) if k not in used]
                while free < cost.bytes and j < len(order):
                    v = order[j]
                    j += 1
                    if v in cached:
                        victims.append(v)
                        free += cached[v]
            if free < cost.bytes or not policy.admit(key, cost, victims, stats, costs):
                result.cpu_executions += 1
                result.time_s += cost.t_cpu
                continue
            cursor = j
            for v in victims:
                used_bytes -= cached.pop(v)
            cached[key] = cost.bytes
            used_bytes += cost.bytes
            result.migrations += 1
            result.bytes_migrated += cost.bytes
            result.time_s += cost.t_migration + cost.t_gpu
            result.peak_bytes = max(result.peak_bytes, used_bytes)
            assert used_bytes <= capacity_bytes
        stats.observe(used)
    return result


def trace_steps(trace) -> list[list[Key]]:
    """(layer, expert) pairs selected in each step of a ``RoutingTrace`` (all batch rows)."""
    out = []
    for step in trace.steps:
        keys = set()
        for layer, routing in enumerate(step):
            for row in routing.experts:
                keys.update((layer, int(e)) for e in row)
        out.append(sorted(keys))
    return out


def compare(steps: list[list[Key]], capacity_bytes: int, costs, *, policies=None) -> list[CacheResult]:
    """Every policy on the same steps; also the two bounds: nothing cached (all on the CPU)
    and an unbounded cache (each expert migrated once)."""
    out = [simulate(steps, POLICIES[name]() if isinstance(name, str) else name, capacity_bytes, costs)
           for name in (policies or POLICIES)]  # fmt: skip
    cpu = CacheResult("none (CPU only)", 0)
    seen: set[Key] = set()
    unbounded = CacheResult("unbounded cache", 0)
    for step in steps:
        for key in step:
            c = costs(key)
            cpu.activations += 1
            cpu.cpu_executions += 1
            cpu.time_s += c.t_cpu
            unbounded.activations += 1
            if key in seen:
                unbounded.hits += 1
                unbounded.time_s += c.t_gpu
            else:
                seen.add(key)
                unbounded.migrations += 1
                unbounded.bytes_migrated += c.bytes
                unbounded.time_s += c.t_migration + c.t_gpu
    unbounded.capacity_bytes = unbounded.peak_bytes = unbounded.bytes_migrated
    return [*out, cpu, unbounded]
