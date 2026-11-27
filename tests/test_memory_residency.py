"""Tensor registry, ownership/residency tracking and replica management."""

from __future__ import annotations

import pytest

from basp.core.tensor import CompressionFormat, DType, TensorKind
from basp.dataflow.base import CachePolicy
from basp.memory.allocator import OutOfMemory
from basp.memory.replication import ReplicaManager
from basp.memory.residency import CopyState, ResidencyError, ResidencyMap
from basp.memory.tensor_registry import TensorRegistry
from tests.conftest import GPU, HOST, MOD0, make_topology

MB = 1 << 20


def setup(gpu_mem: int = 64 * MB):
    topology = make_topology(gpu_mem=gpu_mem)
    registry = TensorRegistry()
    return topology, registry, ResidencyMap(registry, topology)


def weight(registry: TensorRegistry, name: str, n_weights: int = 4 * MB) -> int:
    nbytes = registry.formats.weight_bytes(n_weights, CompressionFormat.INT4)
    return registry.add(name, TensorKind.EXPERT_WEIGHT, (n_weights,), DType.BYTES, nbytes,
                        fmt=CompressionFormat.INT4, n_weights=n_weights).tensor_id  # fmt: skip


def test_registry_tracks_versions_owners_and_formats():
    registry = TensorRegistry()
    w = weight(registry, "w", 1024)
    kv = registry.add("kv", TensorKind.KV_CACHE, (8,), DType.FP16, 16).tensor_id
    assert registry.nbytes(w) == 576 and registry.nbytes(w, CompressionFormat.RAW_FP16) == 2048
    assert registry.nbytes(w, CompressionFormat.ENTROPY) < registry.nbytes(w)
    assert registry.version(kv) == 0 and registry.bump(kv) == 1
    with pytest.raises(ValueError, match="immutable"):
        registry.bump(w)
    registry.set_owner(w, 3)
    assert registry.descriptor(w).owner_device == 3 and registry.descriptor(kv).version == 1
    assert len(registry) == 2 and w in registry
    registry.unregister(kv)
    assert kv not in registry
    with pytest.raises(ValueError):
        registry.register(registry.spec(w))


def test_placement_and_replication_lifecycle():
    _, registry, residency = setup()
    w = weight(registry, "w")
    owner = residency.place(w, HOST)
    assert owner.owner and registry.owner(w) == HOST and residency.valid(w, HOST)
    pending = residency.begin_load(w, GPU, speculative=True, eta=1.5)
    assert pending.state is CopyState.PENDING and not residency.valid(w, GPU)
    assert [c.device for c in residency.holders(w)] == [HOST]  # pending copies are not holders
    assert residency.free_bytes(GPU) < residency.main[GPU].capacity
    residency.commit(w, GPU, now=2.0)
    assert [c.device for c in residency.holders(w)] == [HOST, GPU]
    assert [c.tensor for c in residency.replicas(GPU)] == [w] and residency.replicas(HOST) == []
    residency.touch(w, GPU, now=3.0)
    copy = residency.copy(w, GPU)
    assert copy.last_use == 3.0 and copy.uses == 1 and not copy.speculative
    residency.check_invariants()
    assert residency.evict(w, GPU) == copy.alloc.size
    assert residency.copy(w, GPU) is None and residency.used_bytes(GPU) == 0
    residency.check_invariants()


def test_copies_cannot_be_lost_or_pulled_from_under_a_user():
    _, registry, residency = setup()
    w = weight(registry, "w")
    residency.place(w, HOST)
    with pytest.raises(ResidencyError, match="last valid copy"):
        residency.evict(w, HOST)
    replica = residency.place(w, GPU, owner=False)
    replica.in_use = 1
    with pytest.raises(ResidencyError, match="in use"):
        residency.evict(w, GPU)
    replica.in_use = 0
    residency.evict(w, HOST)  # ownership moves to the surviving copy
    assert registry.owner(w) == GPU and residency.copy(w, GPU).owner
    with pytest.raises(ResidencyError):
        residency.place(w, GPU)
    with pytest.raises(ResidencyError):
        residency.commit(w, GPU)
    with pytest.raises(ResidencyError):
        residency.abort(w, GPU)
    residency.check_invariants()


def test_aborted_load_releases_its_reservation():
    _, registry, residency = setup(gpu_mem=8 * MB)
    a, b = weight(registry, "a", 8 * MB), weight(registry, "b", 8 * MB)
    residency.place(a, HOST)
    residency.place(b, HOST)
    residency.begin_load(a, GPU)
    with pytest.raises(OutOfMemory):
        residency.begin_load(b, GPU)  # 4.5 MiB each: only one fits beside the scratch arena
    residency.abort(a, GPU)
    residency.begin_load(b, GPU)
    residency.check_invariants()


def test_versioned_tensors_invalidate_other_copies_on_write():
    _, registry, residency = setup()
    lora = registry.add("lora", TensorKind.LEARNING_UPDATE, (64,), DType.FP16, 128).tensor_id
    residency.place(lora, HOST)
    residency.place(lora, GPU, owner=False)
    assert residency.write(lora, GPU) == 1
    assert residency.valid(lora, GPU) and not residency.valid(lora, HOST)
    assert residency.copy(lora, HOST).state is CopyState.STALE
    assert registry.owner(lora) == GPU and [c.device for c in residency.holders(lora)] == [GPU]
    with pytest.raises(ResidencyError, match="non-current"):
        residency.write(lora, HOST)  # a stale copy must be synchronised explicitly first
    residency.begin_load(lora, HOST)  # explicit synchronisation: reload, then commit
    residency.commit(lora, HOST)
    assert residency.valid(lora, HOST) and residency.copy(lora, HOST).version == 1
    residency.check_invariants()
    weight_id = weight(registry, "w")
    residency.place(weight_id, HOST)
    with pytest.raises(ValueError, match="immutable"):
        residency.write(weight_id, HOST)


def test_temporaries_live_in_the_scratch_arena_and_drop_together():
    _, registry, residency = setup()
    act = registry.add("act", TensorKind.ACTIVATION, (4096,), DType.FP16, 8192).tensor_id
    w = weight(registry, "w")
    residency.place(act, GPU)
    residency.begin_load(act, MOD0)
    assert residency.used_bytes(GPU) == 0  # the main arena is untouched by temporaries
    assert residency.scratch[GPU].used >= 8192
    residency.drop(act)
    assert residency.copies(act) == [] and residency.scratch[GPU].used == 0
    residency.place(w, HOST)
    with pytest.raises(ResidencyError, match="not temporary"):
        residency.drop(w)


def manager(kind: str, n: int = 6, **kwargs):
    topology, registry, residency = setup()
    tensors = [weight(registry, f"e{i}", MB) for i in range(n)]
    for t in tensors:
        residency.place(t, HOST)
    freq = {t: (i + 1) / 10 for i, t in enumerate(tensors)}
    policy = CachePolicy(kind, (GPU,), **kwargs)
    mgr = ReplicaManager(policy, residency, registry, topology, freq_fn=lambda t: freq.get(t, 0.0))
    return mgr, residency, tensors


def test_victims_are_ordered_by_policy_and_respect_use():
    mgr, residency, tensors = manager("benefit")
    for i, t in enumerate(tensors):
        residency.place(t, GPU, owner=False, now=float(10 - i))
        mgr.track(t, GPU)
    assert mgr.cached(GPU) == 6
    assert [c.tensor for c in mgr.victims(GPU)] == tensors  # least frequent first
    residency.copy(tensors[0], GPU).in_use = 1
    residency.copy(tensors[1], GPU).speculative = True
    mgr._speculative[GPU].append(tensors[1])
    mgr.protected.add((tensors[2], GPU))
    order = [c.tensor for c in mgr.victims(GPU)]
    assert order[0] == tensors[1]  # an unused prefetch goes first
    assert tensors[0] not in order and tensors[2] not in order  # in use / protected
    mgr.untrack(tensors[3], GPU)
    assert mgr.cached(GPU) == 5 and tensors[3] not in [c.tensor for c in mgr.victims(GPU)]

    lru, residency, tensors = manager("lru")
    for i, t in enumerate(tensors):
        residency.place(t, GPU, owner=False, now=float(10 - i))
        lru.track(t, GPU)
    assert [c.tensor for c in lru.victims(GPU)] == tensors[::-1]  # least recently used first


def test_admission_requires_more_value_than_the_victim():
    mgr, residency, tensors = manager("benefit", hysteresis=1.5)
    victim = residency.place(tensors[2], GPU, owner=False)  # frequency 0.3
    assert mgr.admits(0.5, victim) and not mgr.admits(0.4, victim)
    victim.speculative = True
    assert mgr.admits(0.0, victim)  # a mispredicted prefetch is always replaceable
    lru, residency, tensors = manager("lru")
    assert lru.admits(0.0, residency.place(tensors[0], GPU, owner=False))
    none, residency, tensors = manager("none")
    assert not none.admits(1.0, residency.place(tensors[0], GPU, owner=False))
    assert none.victims(GPU) == []
    with pytest.raises(ValueError):
        CachePolicy("mystery")
    with pytest.raises(ValueError):
        CachePolicy("lru", budget_fraction=0.0)
