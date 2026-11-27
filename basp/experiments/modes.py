"""Where each dataflow mode wins: six small scenarios, each simulated per mode.

The point of supporting several dataflows is that none is universally best.
Every scenario below runs the same operations under different single-mode
policies on the illustrative ``basp_4module`` profile and reports the simulated
result, so the regime in which a mode pays off is explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from basp.core.capabilities import NO_COMPUTE, DeviceKind
from basp.core.graph import OpCode
from basp.core.topology import DeviceSpec, LinkSpec, Topology
from basp.dataflow.base import CachePolicy, DataflowMode, ExecutionPolicy
from basp.experiments.microbench import MicroSystem
from basp.units import KiB

M = DataflowMode
EXPERT_PARAMS = 3 * 2048 * 768  # one Qwen3-30B-A3B expert
ACT = 2048 * 2  # one FP16 hidden-state vector


@dataclass
class Scenario:
    key: str
    title: str
    metric: str  # what the numbers are
    unit: str
    results: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def winner(self) -> str:
        return min(self.results, key=lambda k: (self.results[k], k))


def _ids(topology: Topology) -> dict[str, int]:
    return {d.name: d.device_id for d in topology.devices.values()}


def _policy(name: str, modes, devices=None, cache: CachePolicy | None = None) -> ExecutionPolicy:
    return ExecutionPolicy(name, frozenset(modes), frozenset(devices) if devices else None,
                           joint_rounds=0, cache=cache or CachePolicy())  # fmt: skip


def _expert_op(ms: MicroSystem, name: str, weight: int, home: int, *, rows: int = 1, group=None,
               params: int = EXPERT_PARAMS, x: int | None = None):  # fmt: skip
    if x is None:
        x = ms.activation(f"{name}.x", ACT * rows, home)
    y = ms.activation(f"{name}.y", ACT * rows)
    attrs = {"reduce_group": group, "rows": tuple(range(rows))} if group is not None else {}
    return ms.op(OpCode.EXPERT, name, inputs=(x,), weights=(weight,), outputs=(y,),
                 macs=rows * params, batch=rows, output_device=home, attrs=attrs)  # fmt: skip


def single_use(topology: Topology) -> Scenario:
    """One expert on a module, used once; activations live on the GPU."""
    ids = _ids(topology)
    sc = Scenario("single_use", "Expert used once, weights on a module", "completion time", "us")
    for mode in (M.WEIGHT_STATIONARY, M.ACTIVATION_STATIONARY, M.STREAMING):
        ms = MicroSystem(topology, _policy(mode.name, {mode}))
        w = ms.weight("expert", EXPERT_PARAMS, ids["mod0"])
        sc.results[mode.name.lower()] = ms.run([_expert_op(ms, "e", w, ids["gpu"])]).actual_s * 1e6
    return sc


def repeated_use(topology: Topology, uses: int = 64) -> Scenario:
    """The same expert used many times: paying the transfer once beats many round trips."""
    ids = _ids(topology)
    sc = Scenario("repeated_use", f"Same expert used {uses} times", "total time", "us")
    setups = {
        "weight_stationary": _policy("ws", {M.WEIGHT_STATIONARY}),
        "activation_stationary": _policy("as", {M.ACTIVATION_STATIONARY},
                                         cache=CachePolicy("lru", (ids["gpu"],), background=False)),
    }  # fmt: skip
    for name, policy in setups.items():
        ms = MicroSystem(topology, policy)
        w = ms.weight("expert", EXPERT_PARAMS, ids["mod0"])
        total = sum(ms.run([_expert_op(ms, f"e{i}", w, ids["gpu"])]).actual_s for i in range(uses))
        sc.results[name] = total * 1e6
    return sc


def fan_in(topology: Topology, experts: int = 8, rows: int = 16) -> Scenario:
    """Several experts per device feeding one reduction: partial sums cut output traffic."""
    ids = _ids(topology)
    sc = Scenario("fan_in", f"{experts} experts on 2 modules, {rows} rows, one reduction",
                  "interconnect traffic", "KiB",
                  note="completion time is unchanged; only the traffic differs")  # fmt: skip
    for label, modes in (("weight_stationary", {M.WEIGHT_STATIONARY}),
                         ("output_stationary", {M.WEIGHT_STATIONARY, M.OUTPUT_STATIONARY})):  # fmt: skip
        ms = MicroSystem(topology, _policy(label, modes), hidden=2048, batch=rows)
        gpu = ids["gpu"]
        hidden_state = ms.activation("h", ACT * rows, gpu)  # every expert reads the same input
        ops = []
        for i in range(experts):
            w = ms.weight(f"expert{i}", EXPERT_PARAMS, ids[f"mod{i % 2}"])
            ops.append(_expert_op(ms, f"e{i}", w, gpu, rows=rows, group="g", x=hidden_state))
        out = ms.activation("sum", ACT * rows)
        reducer = ms.op(OpCode.REDUCE, "reduce", inputs=tuple(o.outputs[0] for o in ops),
                        outputs=(out,), macs=experts * rows * 2048, batch=rows, pinned_device=gpu,
                        attrs={"reduce_group": "g", "reducer": True})  # fmt: skip
        sc.results[label] = ms.run(ops + [reducer]).link_bytes / KiB
    return sc


def sram_reuse(topology: Topology, uses: int = 16) -> Scenario:
    """A hot expert served from the module's SRAM tile cache after the first use."""
    ids = _ids(topology)
    sc = Scenario("sram_reuse", f"Same expert run {uses} times on its module", "total time", "us")
    for label, modes in (("weight_stationary", {M.WEIGHT_STATIONARY}),
                         ("row_stationary", {M.ROW_STATIONARY})):  # fmt: skip
        ms = MicroSystem(topology, _policy(label, modes))
        w = ms.weight("expert", EXPERT_PARAMS, ids["mod0"])
        total = sum(ms.run([_expert_op(ms, f"e{i}", w, ids["gpu"])]).actual_s for i in range(uses))
        sc.results[label] = total * 1e6
    return sc


def cold_stream(topology: Topology) -> Scenario:
    """A large expert held only on cold storage, used once by a compute-memory module.

    Uses the section 6 expert (hidden 4096, intermediate 14336) and adds an NVMe
    device (7 GB/s, assumed) to the profile.
    """
    ids = _ids(topology)
    nvme_id = max(topology.devices) + 1
    nvme = DeviceSpec(nvme_id, "nvme", DeviceKind.STORAGE, 1 << 40, 7e9, 0.0, NO_COMPUTE)
    tiered = Topology(
        [*topology.devices.values(), nvme],
        [*topology.links, LinkSpec(ids["host"], nvme_id, 7e9, latency_s=10e-6, name="nvme")],
        host=topology.host, sync_latency_s=topology.sync_latency_s, name=topology.name + "+nvme",
    )  # fmt: skip
    params = 3 * 4096 * 14336
    sc = Scenario("cold_stream", "Large expert on NVMe only, used once on a module",
                  "completion time", "us")  # fmt: skip
    for mode in (M.ACTIVATION_STATIONARY, M.STREAMING):
        ms = MicroSystem(tiered, _policy(mode.name, {mode}, devices={ids["mod0"]}))
        w = ms.weight("big", params, nvme_id)
        op = _expert_op(ms, "e", w, ids["mod0"], params=params)
        sc.results[mode.name.lower()] = ms.run([op]).actual_s * 1e6
    return sc


def sharded_projection(topology: Topology, vocab: int = 151936, hidden: int = 2048) -> Scenario:
    """The output projection: four row shards on four modules versus all on one."""
    ids = _ids(topology)
    modules = [ids[f"mod{i}"] for i in range(4)]
    sc = Scenario("sharded_projection", "Output projection (151936 x 2048), 4 row shards",
                  "completion time", "us")  # fmt: skip
    for label, modes, spread in (("weight_stationary", {M.WEIGHT_STATIONARY}, False),
                                 ("distributed", {M.DISTRIBUTED}, True)):  # fmt: skip
        ms = MicroSystem(topology, _policy(label, modes))
        rows = vocab // 4
        weights = [ms.weight(f"head{i}", rows * hidden, modules[i] if spread else modules[0])
                   for i in range(4)]  # fmt: skip
        x = ms.activation("x", ACT, ids["gpu"])
        logits = ms.activation("logits", vocab * 2)
        op = ms.op(OpCode.MATVEC, "lm_head", inputs=(x,), weights=weights, outputs=(logits,),
                   macs=vocab * hidden, output_device=ids["gpu"], partitionable=True)  # fmt: skip
        sc.results[label] = ms.run([op]).actual_s * 1e6
    return sc


def mode_study(topology: Topology) -> list[Scenario]:
    return [single_use(topology), repeated_use(topology), fan_in(topology), sram_reuse(topology),
            cold_stream(topology), sharded_projection(topology)]  # fmt: skip


def expected_winners() -> dict[str, str]:
    return {
        "single_use": "weight_stationary",
        "repeated_use": "activation_stationary",
        "fan_in": "output_stationary",
        "sram_reuse": "row_stationary",
        "cold_stream": "streaming",
        "sharded_projection": "distributed",
    }

