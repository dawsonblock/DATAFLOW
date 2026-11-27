"""Explicit completion events and the bounded queue that carries them to the host.

The payload of a COMPLETION frame is little-endian (48 bytes, 64 framed)::

    offset  size  field
    0       8     completion_id  tag copied from the command
    8       8     op_id          operation that finished
    16      8     timestamp_ns   device time of completion
    24      8     bytes_moved    bytes transferred or read by the operation
    32      8     detail         status-specific information (error code, fault address)
    40      4     device_id      device that executed the operation
    44      1     status         terminal Status
    45      3     reserved       must be zero
"""

from __future__ import annotations

import operator
import struct
from collections import deque
from dataclasses import dataclass

from basp.streams.protocol import FrameKind, ProtocolError, Status, frame, unframe_exact

_EVENT = struct.Struct("<5QIB3s")
_U64 = (1 << 64) - 1


@dataclass(frozen=True)
class CompletionEvent:
    """Outcome of one operation, posted by the device that ran it."""

    completion_id: int
    op_id: int
    device_id: int
    status: Status
    timestamp_ns: int = 0
    bytes_moved: int = 0
    detail: int = 0

    def __post_init__(self) -> None:
        for name in (
            "completion_id",
            "op_id",
            "device_id",
            "timestamp_ns",
            "bytes_moved",
            "detail",
        ):
            object.__setattr__(self, name, operator.index(getattr(self, name)))
        object.__setattr__(self, "status", Status(self.status))

    def encode(self) -> bytes:
        for name in ("completion_id", "op_id", "timestamp_ns", "bytes_moved", "detail"):
            if not 0 <= getattr(self, name) <= _U64:
                raise ProtocolError(f"completion {name} does not fit in 64 bits")
        if not 0 <= self.device_id <= 0xFFFFFFFF:
            raise ProtocolError("completion device_id does not fit in 32 bits")
        payload = _EVENT.pack(
            self.completion_id, self.op_id, self.timestamp_ns, self.bytes_moved, self.detail,
            self.device_id, self.status, b"\0\0\0",
        )  # fmt: skip
        return frame(FrameKind.COMPLETION, payload)

    @classmethod
    def decode(cls, buf: bytes) -> CompletionEvent:
        payload = unframe_exact(buf, FrameKind.COMPLETION)
        if len(payload) != _EVENT.size:
            raise ProtocolError(f"completion payload is {len(payload)} bytes, not {_EVENT.size}")
        completion_id, op_id, ts, moved, detail, device_id, status, pad = _EVENT.unpack(payload)
        if pad != b"\0\0\0":
            raise ProtocolError("reserved completion bytes must be zero")
        try:
            return cls(completion_id, op_id, device_id, Status(status), ts, moved, detail)
        except ValueError as exc:
            raise ProtocolError(str(exc)) from None


class CompletionQueue:
    """Bounded FIFO of completion events with explicit overflow reporting.

    A full queue refuses the event: ``post`` returns False, ``dropped`` counts
    it and ``overflowed`` stays set until :meth:`acknowledge_overflow`.  The
    consumer must then resynchronise with QUERY_STATUS, so a lost completion is
    never silent.
    """

    def __init__(self, capacity: int = 1024):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._events: deque[CompletionEvent] = deque()
        self.posted = 0
        self.dropped = 0
        self.overflowed = False
        self.high_water = 0

    def post(self, event: CompletionEvent) -> bool:
        if len(self._events) >= self.capacity:
            self.dropped += 1
            self.overflowed = True
            return False
        self._events.append(event)
        self.posted += 1
        self.high_water = max(self.high_water, len(self._events))
        return True

    def poll(self) -> CompletionEvent | None:
        return self._events.popleft() if self._events else None

    def drain(self) -> list[CompletionEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def acknowledge_overflow(self) -> int:
        """Clear the overflow flag after a resync; returns the number of events lost so far."""
        self.overflowed = False
        return self.dropped

    def __len__(self) -> int:
        return len(self._events)
