"""Ownership, residency and validity of tensor copies across devices.

The memory manager knows, for every tensor, which device owns it, which devices
hold copies and which copies are valid.  Immutable tensors (weights) never
become stale, which is what keeps distributed memory management simple; mutable
tensors are versioned and a write invalidates every other copy until it is
explicitly synchronised.
"""

from __future__ import annotations

import enum

from basp.core.tensor import CompressionFormat
from basp.core.topology import Topology
from basp.memory.allocator import Allocation, Allocator, OutOfMemory
from basp.memory.tensor_registry import TensorRegistry


class CopyState(enum.Enum):
    VALID = "valid"
    PENDING = "pending"  # memory reserved, data still in flight
    STALE = "stale"  # superseded by a newer version elsewhere


class ResidencyError(RuntimeError):
    pass


class Copy:
    __slots__ = (
        "alloc",
        "device",
        "eta",
        "fmt",
        "in_use",
        "last_use",
        "nbytes",
        "owner",
        "pinned",
        "speculative",
        "state",
        "tensor",
        "uses",
        "version",
    )  # fmt: skip

    def __init__(self, tensor, device, version, state, fmt, nbytes, pinned, owner, alloc):
        self.tensor: int = tensor
        self.device: int = device
        self.version: int = version
        self.state: CopyState = state
        self.fmt: CompressionFormat | None = fmt
        self.nbytes: int = nbytes
        self.pinned: bool = pinned
        self.owner: bool = owner
        self.alloc: Allocation = alloc
        self.last_use = 0.0
        self.uses = 0
        self.in_use = 0  # plans/kernels currently depending on this copy
        self.speculative = False  # prefetched and not used yet
        self.eta: float | None = None

    def __repr__(self) -> str:
        return f"Copy(t{self.tensor}@d{self.device} v{self.version} {self.state.value})"


class ResidencyMap:
    """Tracks every physical copy and the per-device memory it occupies.

    Each device has two arenas: the main arena for weights and replicas, and a
    small scratch arena for temporaries so short-lived activations cannot
    fragment the space that equal-sized expert blobs are recycled through.
    """

    def __init__(self, registry: TensorRegistry, topology: Topology, *, scratch_fraction: float = 1 / 16,
                 scratch_max: int = 256 * 1024 * 1024) -> None:  # fmt: skip
        self.registry = registry
        self.topology = topology
        self.main: dict[int, Allocator] = {}
        self.scratch: dict[int, Allocator] = {}
        for dev in topology.devices.values():
            owner = topology.memory_owner(dev.device_id)
            if owner != dev.device_id and owner in self.main:
                # unified memory: every device of the domain allocates from the same pool
                self.scratch[dev.device_id] = self.scratch[owner]
                self.main[dev.device_id] = self.main[owner]
                continue
            # a shared (unified) memory pool gives every device of the domain its usual scratch share
            sharing = sum(1 for d in topology.devices if topology.memory_owner(d) == dev.device_id)
            scratch = min(int(dev.mem_capacity * scratch_fraction), scratch_max) * sharing
            scratch -= scratch % 64
            self.scratch[dev.device_id] = Allocator(max(scratch, 64))
            self.main[dev.device_id] = Allocator(max(dev.mem_capacity - scratch, 64))
        self._copies: dict[int, dict[int, Copy]] = {}
        self._on_device: dict[int, dict[int, Copy]] = {d: {} for d in topology.devices}

    # -- creation -----------------------------------------------------------------
    def _arena(self, tensor_id: int, device: int) -> Allocator:
        temporary = self.registry.spec(tensor_id).policy.temporary
        return self.scratch[device] if temporary else self.main[device]

    def _new(self, tensor_id, device, state, fmt, pinned, owner) -> Copy:
        if device in self._copies.get(tensor_id, {}):
            raise ResidencyError(f"tensor {tensor_id} already has a copy on device {device}")
        spec = self.registry.spec(tensor_id)
        fmt = fmt if fmt is not None else spec.fmt
        nbytes = self.registry.nbytes(tensor_id, fmt)
        alloc = self._arena(tensor_id, device).allocate(max(nbytes, 1), tag=tensor_id)
        copy = Copy(tensor_id, device, self.registry.version(tensor_id), state, fmt, nbytes,
                    pinned, owner, alloc)  # fmt: skip
        self._copies.setdefault(tensor_id, {})[device] = copy
        self._on_device[device][tensor_id] = copy
        return copy

    def place(self, tensor_id: int, device: int, *, fmt: CompressionFormat | None = None,
              pinned: bool = False, owner: bool = False, now: float = 0.0) -> Copy:  # fmt: skip
        """Create a valid copy immediately (initial placement or a freshly produced tensor)."""
        copy = self._new(tensor_id, device, CopyState.VALID, fmt, pinned, owner)
        copy.last_use = now
        if owner or self.registry.owner(tensor_id) is None:
            copy.owner = True
            self.registry.set_owner(tensor_id, device)
        return copy

    def begin_load(self, tensor_id: int, device: int, *, fmt: CompressionFormat | None = None,
                   speculative: bool = False, eta: float | None = None) -> Copy:  # fmt: skip
        """Reserve memory for a copy that is about to be transferred in."""
        existing = self.copy(tensor_id, device)
        if existing is not None and existing.state is CopyState.STALE:
            existing.state = CopyState.PENDING
            existing.eta = eta
            return existing
        copy = self._new(tensor_id, device, CopyState.PENDING, fmt, False, False)
        copy.speculative = speculative
        copy.eta = eta
        return copy

    def commit(self, tensor_id: int, device: int, now: float = 0.0) -> Copy:
        copy = self._require(tensor_id, device)
        if copy.state is not CopyState.PENDING:
            raise ResidencyError(f"{copy} is not pending")
        copy.state = CopyState.VALID
        copy.version = self.registry.version(tensor_id)
        copy.last_use = now
        copy.eta = None
        return copy

    def abort(self, tensor_id: int, device: int) -> None:
        copy = self._require(tensor_id, device)
        if copy.state is not CopyState.PENDING:
            raise ResidencyError(f"{copy} is not pending")
        self._remove(copy)

    # -- removal ------------------------------------------------------------------
    def evict(self, tensor_id: int, device: int) -> int:
        """Drop a copy and free its memory; returns the bytes released."""
        copy = self._require(tensor_id, device)
        if copy.pinned or copy.in_use:
            raise ResidencyError(f"{copy} is pinned or in use")
        spec = self.registry.spec(tensor_id)
        if copy.state is CopyState.VALID and not spec.policy.temporary:
            others = [c for c in self.holders(tensor_id) if c is not copy]
            if not others:
                raise ResidencyError(f"refusing to evict the last valid copy of tensor {tensor_id}")
            if copy.owner:
                copy.owner = False
                others[0].owner = True
                self.registry.set_owner(tensor_id, others[0].device)
        self._remove(copy)
        return copy.alloc.size

    def drop(self, tensor_id: int) -> None:
        """Free every copy of a temporary tensor."""
        if not self.registry.spec(tensor_id).policy.temporary:
            raise ResidencyError(f"tensor {tensor_id} is not temporary")
        for copy in list(self._copies.get(tensor_id, {}).values()):
            self._remove(copy)

    def _remove(self, copy: Copy) -> None:
        self._arena(copy.tensor, copy.device).free(copy.alloc)
        del self._copies[copy.tensor][copy.device]
        if not self._copies[copy.tensor]:
            del self._copies[copy.tensor]
        del self._on_device[copy.device][copy.tensor]

    # -- queries ------------------------------------------------------------------
    def _require(self, tensor_id: int, device: int) -> Copy:
        copy = self.copy(tensor_id, device)
        if copy is None:
            raise ResidencyError(f"tensor {tensor_id} has no copy on device {device}")
        return copy

    def copy(self, tensor_id: int, device: int) -> Copy | None:
        return self._copies.get(tensor_id, {}).get(device)

    def copies(self, tensor_id: int) -> list[Copy]:
        return sorted(self._copies.get(tensor_id, {}).values(), key=lambda c: c.device)

    def valid(self, tensor_id: int, device: int) -> bool:
        copy = self.copy(tensor_id, device)
        return (
            copy is not None
            and copy.state is CopyState.VALID
            and copy.version == self.registry.version(tensor_id)
        )

    def holders(self, tensor_id: int) -> list[Copy]:
        """Valid copies at the current version, ordered by device id."""
        version = self.registry.version(tensor_id)
        return [
            c for c in self.copies(tensor_id)
            if c.state is CopyState.VALID and c.version == version
        ]  # fmt: skip

    def on_device(self, device: int) -> list[Copy]:
        return list(self._on_device[device].values())

    def replicas(self, device: int) -> list[Copy]:
        """Copies that may be evicted right now: valid, unpinned, idle and not the owner."""
        return [
            c for c in self._on_device[device].values()
            if c.state is CopyState.VALID and not c.pinned and not c.owner and not c.in_use
            and not self.registry.spec(c.tensor).policy.temporary
        ]  # fmt: skip

    def free_bytes(self, device: int) -> int:
        return self.main[device].free_bytes

    def used_bytes(self, device: int) -> int:
        return self.main[device].used

    def touch(self, tensor_id: int, device: int, now: float) -> None:
        copy = self.copy(tensor_id, device)
        if copy is not None:
            copy.last_use = now
            copy.uses += 1
            copy.speculative = False

    # -- mutation of versioned tensors --------------------------------------------
    def write(self, tensor_id: int, device: int) -> int:
        """Record a write on ``device``: bump the version and invalidate other copies."""
        copy = self._require(tensor_id, device)
        if copy.state is not CopyState.VALID or copy.version != self.registry.version(tensor_id):
            raise ResidencyError(f"cannot write through non-current {copy}")
        version = self.registry.bump(tensor_id)
        for other in self._copies[tensor_id].values():
            if other is not copy and other.state is CopyState.VALID:
                other.state = CopyState.STALE
        copy.version = version
        if not copy.owner:
            for other in self._copies[tensor_id].values():
                other.owner = False
            copy.owner = True
            self.registry.set_owner(tensor_id, device)
        return version

    # -- verification -------------------------------------------------------------
    def check_invariants(self) -> None:
        for arena in {id(a): a for a in (*self.main.values(), *self.scratch.values())}.values():
            arena.check_invariants()
        groups: dict[int, list[int]] = {}
        for device in self.topology.devices:
            groups.setdefault(self.topology.memory_owner(device), []).append(device)
        for owner, devices in groups.items():
            tagged = sorted(
                a.tag for arena in (self.main[owner], self.scratch[owner]) for a in arena.allocations()
            )
            expected = sorted(t for d in devices for t in self._on_device[d])
            assert tagged == expected, f"devices {devices}: allocation mismatch"
        for tensor_id, by_dev in self._copies.items():
            spec = self.registry.spec(tensor_id)
            version = self.registry.version(tensor_id)
            owners = [c for c in by_dev.values() if c.owner]
            assert len(owners) <= 1, f"tensor {tensor_id} has {len(owners)} owners"
            for copy in by_dev.values():
                assert copy.version <= version, f"{copy} is from the future"
                assert copy.in_use >= 0
                if spec.policy.immutable:
                    assert copy.state is not CopyState.STALE, f"immutable {copy} is stale"
                if copy.state is CopyState.VALID and copy.version < version:
                    raise AssertionError(f"{copy} is valid but outdated")
            if owners:
                owner = owners[0]
                assert self.registry.owner(tensor_id) == owner.device
                assert owner.state is CopyState.VALID and owner.version == version, (
                    f"owner {owner} does not hold the current version"
                )


__all__ = ["Copy", "CopyState", "OutOfMemory", "ResidencyError", "ResidencyMap"]
