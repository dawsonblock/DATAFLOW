"""Command-level behaviour of the virtual backend (BSP semantics)."""

from __future__ import annotations

import pytest

from basp.backends.virtual.system import VirtualSystem
from basp.core.tensor import CompressionFormat, DType, TensorKind
from basp.memory.residency import CopyState, ResidencyMap
from basp.memory.tensor_registry import TensorRegistry
from basp.simulator.compute import Kernel, Phase, compute_key, mem_key
from basp.streams.commands import Command
from basp.streams.protocol import NO_DEVICE, Flags, Opcode, ProtocolError, Status
from basp.verification import check_accounting
from tests.conftest import GPU, HOST, MOD0, make_topology

MB = 1 << 20


class Rig:
    def __init__(self, **topo) -> None:
        self.topology = make_topology(**topo)
        self.registry = TensorRegistry()
        self.residency = ResidencyMap(self.registry, self.topology)
        self.system = VirtualSystem(self.topology, self.registry, self.residency)
        self.done: list[tuple[int, Status, float]] = []

    def weight(self, name: str, device: int, n_weights: int = 16 * MB) -> int:
        nbytes = self.registry.formats.weight_bytes(n_weights, CompressionFormat.INT4)
        tid = self.registry.add(name, TensorKind.EXPERT_WEIGHT, (n_weights,), DType.BYTES, nbytes,
                                fmt=CompressionFormat.INT4, n_weights=n_weights).tensor_id  # fmt: skip
        self.residency.place(tid, device)
        return tid

    def activation(self, name: str, device: int | None = None, nbytes: int = 4096) -> int:
        tid = self.registry.add(name, TensorKind.ACTIVATION, (nbytes // 2,), DType.FP16, nbytes).tensor_id
        if device is not None:
            self.residency.place(tid, device)
        return tid

    def submit(self, opcode, device, **kw):
        extra = {k: kw.pop(k) for k in ("kernel", "reads", "writes", "nbytes", "tag") if k in kw}
        cmd = Command(self.system.new_id(), opcode, device, **kw)
        self.system.submit(cmd, on_complete=self._record, **extra)
        return cmd.op_id

    def _record(self, state) -> None:
        self.done.append((state.cmd.op_id, state.status, self.system.sim.now))

    def status(self, cmd_id: int) -> Status:
        return self.system.status(cmd_id)

    def kernel(self, device: int, macs: float = 256e6) -> Kernel:
        return Kernel(device, (Phase(((compute_key(device), macs),)),), macs)


def test_transfer_then_execute_respects_dependencies():
    rig = Rig()
    x = rig.activation("x", GPU)
    y = rig.activation("y")
    w = rig.weight("w", MOD0)
    move = rig.submit(Opcode.TRANSFER, MOD0, inputs=(x,), src_device=GPU)
    run = rig.submit(Opcode.EXECUTE, MOD0, inputs=(x, w), outputs=(y,), deps=(move,),
                     kernel=rig.kernel(MOD0), reads=((x, MOD0), (w, MOD0)), writes=(y,))  # fmt: skip
    back = rig.submit(Opcode.TRANSFER, GPU, inputs=(y,), src_device=MOD0, deps=(run,))
    barrier = rig.submit(Opcode.BARRIER, NO_DEVICE, deps=(back,))
    rig.system.run()
    assert [cid for cid, _, _ in rig.done] == [move, run, back, barrier]
    assert all(status is Status.SUCCESS for _, status, _ in rig.done)
    times = {cid: t for cid, _, t in rig.done}
    hop = 7e-6 + 4096 / 16e9
    assert times[move] == pytest.approx(hop)
    assert times[run] == pytest.approx(hop + 1e-3)  # 256 M MAC at 256 GMAC/s
    assert times[back] == times[barrier] == pytest.approx(2 * hop + 1e-3)
    assert rig.residency.valid(y, GPU) and rig.residency.valid(y, MOD0)
    assert not rig.system.violations and check_accounting(rig.system) == []
    events = rig.system.completions.drain()
    assert [e.op_id for e in events] == [move, run, back, barrier]
    assert events[0].bytes_moved == 4096 and events[1].timestamp_ns == int((hop + 1e-3) * 1e9)


def test_dma_isolation_rejects_weight_data_without_an_allocated_window():
    rig = Rig()
    w = rig.weight("w", HOST)
    rogue = rig.submit(Opcode.TRANSFER, GPU, inputs=(w,), src_device=HOST)
    rig.system.run()
    assert rig.status(rogue) is Status.PROTECTION_FAULT
    assert rig.residency.copy(w, GPU) is None and "no DMA window" in rig.system.violations[0]
    alloc = rig.submit(Opcode.ALLOCATE, GPU, inputs=(w,), arg0=rig.registry.spec(w).nbytes)
    move = rig.submit(Opcode.TRANSFER, GPU, inputs=(w,), src_device=HOST, deps=(alloc,),
                      flags=Flags.RETAIN)  # fmt: skip
    rig.system.run()
    assert rig.status(move) is Status.SUCCESS and rig.residency.valid(w, GPU)
    missing = rig.submit(Opcode.TRANSFER, MOD0, inputs=(rig.activation("ghost"),), src_device=GPU)
    rig.system.run()
    assert rig.status(missing) is Status.PROTECTION_FAULT  # the source holds no such tensor


def test_version_mismatch_is_detected():
    rig = Rig()
    kv = rig.registry.add("kv", TensorKind.LEARNING_UPDATE, (64,), DType.FP16, 128).tensor_id
    rig.residency.place(kv, HOST)
    rig.residency.place(kv, GPU, owner=False)
    stale_version = rig.registry.version(kv)
    rig.residency.write(kv, GPU)  # GPU now holds version 1; the host copy is stale
    rig.residency.begin_load(kv, MOD0)
    old = rig.submit(Opcode.TRANSFER, MOD0, inputs=(kv,), src_device=GPU, version=stale_version)
    stale_src = rig.submit(Opcode.TRANSFER, MOD0, inputs=(kv,), src_device=HOST, version=1)
    good = rig.submit(Opcode.TRANSFER, MOD0, inputs=(kv,), src_device=GPU, version=1)
    rig.system.run()
    assert rig.status(old) is Status.VERSION_MISMATCH
    assert rig.status(stale_src) is Status.VERSION_MISMATCH
    assert rig.status(good) is Status.SUCCESS and rig.residency.copy(kv, MOD0).version == 1


def test_execute_without_resident_inputs_fails_and_is_reported():
    rig = Rig()
    x, y = rig.activation("x", GPU), rig.activation("y")
    w = rig.weight("w", MOD0)
    run = rig.submit(Opcode.EXECUTE, MOD0, inputs=(x, w), outputs=(y,), kernel=rig.kernel(MOD0),
                     reads=((x, MOD0), (w, MOD0)), writes=(y,))  # fmt: skip
    after = rig.submit(Opcode.BARRIER, NO_DEVICE, deps=(run,))
    rig.system.run()
    assert rig.status(run) is Status.FAILED
    assert rig.status(after) is Status.FAILED  # failure propagates to dependents
    assert "without a valid copy" in rig.system.violations[0]
    assert rig.residency.copy(y, MOD0) is None
    with pytest.raises(ProtocolError, match="kernel descriptor"):
        rig.submit(Opcode.EXECUTE, MOD0, inputs=(w,), outputs=(y,))
    with pytest.raises(ProtocolError, match="unknown command"):
        rig.submit(Opcode.BARRIER, NO_DEVICE, deps=(9999,))


def test_timeout_cancels_work_and_releases_reservations():
    rig = Rig()
    w = rig.weight("w", HOST, n_weights=64 * MB)  # 36 MiB: 1.5 ms over the 24 GB/s link
    alloc = rig.submit(Opcode.ALLOCATE, GPU, inputs=(w,), arg0=rig.registry.spec(w).nbytes)
    slow = rig.submit(Opcode.TRANSFER, GPU, inputs=(w,), src_device=HOST, deps=(alloc,),
                      timeout_ns=500_000)  # fmt: skip
    rig.system.run()
    assert rig.status(slow) is Status.TIMEOUT
    assert rig.done[-1][2] == pytest.approx(500e-6)
    assert rig.residency.copy(w, GPU) is None and rig.residency.used_bytes(GPU) == 0
    moved = next(r for r in rig.system.log if r.cmd_id == slow).nbytes
    assert moved == pytest.approx(24e9 * (500e-6 - 5e-6), rel=1e-6)  # partial bytes stay accounted
    assert check_accounting(rig.system) == []


def test_cancel_and_query_status_commands():
    rig = Rig()
    w = rig.weight("w", HOST, n_weights=64 * MB)
    alloc = rig.submit(Opcode.ALLOCATE, GPU, inputs=(w,), arg0=1, flags=Flags.SPECULATIVE)
    fetch = rig.submit(Opcode.PREFETCH, GPU, inputs=(w,), src_device=HOST, deps=(alloc,), priority=1)
    assert rig.residency.copy(w, GPU).state is CopyState.PENDING
    assert rig.residency.copy(w, GPU).speculative
    rig.system.run(until=100e-6)
    query = rig.submit(Opcode.QUERY_STATUS, NO_DEVICE, arg0=fetch)
    cancel = rig.submit(Opcode.CANCEL, NO_DEVICE, arg0=fetch)
    again = rig.submit(Opcode.CANCEL, NO_DEVICE, arg0=fetch)
    rig.system.run()
    assert rig.status(fetch) is Status.CANCELLED and rig.residency.copy(w, GPU) is None
    details = {e.op_id: e.detail for e in rig.system.completions.drain()}
    assert details[query] == int(Status.RUNNING)
    assert details[cancel] == 1 and details[again] == 0  # second cancel finds nothing to stop
    assert rig.system.net.resources["link:host->gpu"].by_tag.get("prefetch", 0) > 0


def test_bounded_command_queue_applies_backpressure_without_losing_commands():
    rig = Rig(queue_depth=2)
    kernel = rig.kernel(MOD0)
    ids = [rig.submit(Opcode.EXECUTE, MOD0, outputs=(rig.activation(f"y{i}"),), kernel=kernel,
                      writes=()) for i in range(6)]  # fmt: skip
    assert rig.system.stats["backpressure"] == 4  # two in the device queue, four held at the host
    assert [rig.status(i) for i in ids[:3]] == [Status.RUNNING, Status.QUEUED, Status.PENDING]
    rig.system.run()
    assert [cid for cid, _, _ in rig.done] == ids  # FIFO order is preserved
    assert all(status is Status.SUCCESS for _, status, _ in rig.done)
    assert rig.done[-1][2] == pytest.approx(6e-3)
    assert rig.system.endpoints[MOD0].high_water == 2 and len(rig.system.endpoints[MOD0]) == 0


def test_background_budget_caps_aggregate_background_traffic():
    topology = make_topology()
    registry = TensorRegistry()
    residency = ResidencyMap(registry, topology)
    system = VirtualSystem(topology, registry, residency, background_budget=0.25)
    finished = {}
    ids = []
    for i in range(2):
        nbytes = registry.formats.weight_bytes(16 * MB, CompressionFormat.INT4)
        tid = registry.add(f"w{i}", TensorKind.EXPERT_WEIGHT, (16 * MB,), DType.BYTES, nbytes,
                           fmt=CompressionFormat.INT4, n_weights=16 * MB).tensor_id  # fmt: skip
        residency.place(tid, HOST)
        alloc = Command(system.new_id(), Opcode.ALLOCATE, GPU, inputs=(tid,), arg0=nbytes)
        system.submit(alloc)
        move = Command(system.new_id(), Opcode.TRANSFER, GPU, inputs=(tid,), src_device=HOST,
                       deps=(alloc.op_id,), priority=1, flags=Flags.RETAIN)  # fmt: skip
        system.submit(move, tag="replicate",
                      on_complete=lambda s: finished.setdefault(s.cmd.op_id, system.sim.now))  # fmt: skip
        ids.append(move.op_id)
    system.run()
    total = 2 * registry.formats.weight_bytes(16 * MB, CompressionFormat.INT4)
    # both loads share one quarter of the 24 GB/s link
    assert max(finished.values()) == pytest.approx(5e-6 + total / (0.25 * 24e9), rel=1e-6)
    assert check_accounting(system) == []
    with pytest.raises(ValueError):
        VirtualSystem(topology, TensorRegistry(), ResidencyMap(TensorRegistry(), topology),
                      background_budget=1.5)  # fmt: skip


def test_free_releases_memory_and_refuses_unsafe_evictions():
    rig = Rig()
    w = rig.weight("w", HOST)
    rig.residency.place(w, GPU, owner=False)
    free = rig.submit(Opcode.FREE, GPU, inputs=(w,))
    last = rig.submit(Opcode.FREE, HOST, inputs=(w,))
    rig.system.run()
    assert rig.status(free) is Status.SUCCESS and rig.residency.copy(w, GPU) is None
    assert rig.status(last) is Status.FAILED and rig.residency.valid(w, HOST)  # last copy is kept
    assert rig.residency.copy(w, HOST).alloc is not None and mem_key(HOST) in rig.system.res
