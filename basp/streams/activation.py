"""Activation streaming: split an activation tensor into frames and reassemble it.

Activations, not weight matrices, are what crosses the interconnect in the
weight-stationary dataflows, so they get a dedicated light-weight framing.  The
payload of an ACTIVATION frame is little-endian::

    offset  size  field
    0       8     tensor_id
    8       8     version    tensor version the data belongs to
    16      4     seq        frame index, 0 <= seq < total
    20      4     total      number of frames of this tensor version (>= 1)
    24      n     data       at most ``mtu`` bytes

Each frame costs 24 + 16 = 40 bytes of header and framing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from basp.core.tensor import activation_bytes
from basp.streams.protocol import FrameKind, ProtocolError, frame, unframe_exact

__all__ = ["ActivationFrame", "Reassembler", "activation_bytes", "packetize"]

_HEAD = struct.Struct("<QQII")
_U64 = (1 << 64) - 1
_U32 = (1 << 32) - 1


@dataclass(frozen=True)
class ActivationFrame:
    tensor_id: int
    version: int
    seq: int
    total: int
    payload: bytes

    def _check(self) -> None:
        if not (0 <= self.tensor_id <= _U64 and 0 <= self.version <= _U64):
            raise ProtocolError("activation tensor_id/version do not fit in 64 bits")
        if not 1 <= self.total <= _U32:
            raise ProtocolError(f"activation frame count {self.total} out of range")
        if not 0 <= self.seq < self.total:
            raise ProtocolError(f"activation frame seq {self.seq} outside 0..{self.total - 1}")

    def encode(self) -> bytes:
        self._check()
        head = _HEAD.pack(self.tensor_id, self.version, self.seq, self.total)
        return frame(FrameKind.ACTIVATION, head + bytes(self.payload))

    @classmethod
    def decode(cls, buf: bytes) -> ActivationFrame:
        payload = unframe_exact(buf, FrameKind.ACTIVATION)
        if len(payload) < _HEAD.size:
            raise ProtocolError(f"activation payload too short: {len(payload)} bytes")
        out = cls(*_HEAD.unpack_from(payload), payload[_HEAD.size :])
        out._check()
        return out


def packetize(tensor_id: int, version: int, data: bytes, mtu: int = 4096) -> list[ActivationFrame]:
    """Split ``data`` into frames of at most ``mtu`` data bytes (one frame if empty)."""
    if mtu < 1:
        raise ValueError("mtu must be at least 1 byte")
    data = bytes(data)
    total = max(1, -(-len(data) // mtu))
    return [
        ActivationFrame(tensor_id, version, seq, total, data[seq * mtu : (seq + 1) * mtu])
        for seq in range(total)
    ]


@dataclass
class _Partial:
    version: int
    total: int
    chunks: dict[int, bytes] = field(default_factory=dict)


class Reassembler:
    """Rebuild activation tensors from frames arriving in any order.

    Tensor consistency rule: once a version of a tensor has been seen, frames of
    a lower version are rejected; a higher version supersedes (and discards) an
    incomplete lower one.  Duplicates are ignored and counted.
    """

    def __init__(self) -> None:
        self._partial: dict[int, _Partial] = {}
        self._latest: dict[int, int] = {}  # highest version seen per tensor
        self._done: dict[int, int] = {}  # last completed version per tensor
        self.duplicates = 0
        self.superseded = 0

    def add(self, frame_: ActivationFrame) -> bytes | None:
        """Accept one frame; returns the whole payload when its tensor version completes."""
        frame_._check()
        tid, version = frame_.tensor_id, frame_.version
        latest = self._latest.get(tid)
        if latest is not None and version < latest:
            raise ProtocolError(f"stale activation: tensor {tid} v{version} after v{latest}")
        if self._done.get(tid) == version:
            self.duplicates += 1
            return None
        part = self._partial.get(tid)
        if part is None or part.version != version:
            if part is not None:
                self.superseded += 1
            part = self._partial[tid] = _Partial(version, frame_.total)
            self._latest[tid] = version
        if part.total != frame_.total:
            raise ProtocolError(
                f"tensor {tid} v{version}: frame count {frame_.total} disagrees with {part.total}"
            )
        if frame_.seq in part.chunks:
            self.duplicates += 1
            return None
        part.chunks[frame_.seq] = bytes(frame_.payload)
        if len(part.chunks) < part.total:
            return None
        del self._partial[tid]
        self._done[tid] = version
        return b"".join(part.chunks[seq] for seq in range(part.total))

    def pending(self) -> int:
        """Number of tensors with an incomplete version."""
        return len(self._partial)

    def forget(self, tensor_id: int) -> None:
        """Drop all state for a freed tensor (its id may then be reused from version 0)."""
        for table in (self._partial, self._latest, self._done):
            table.pop(tensor_id, None)
