"""BASP Stream Protocol (BSP): framing, enumerations and flow-control primitives.

Every BSP message travels in one little-endian frame::

    offset  size  field
    0       4     magic       0x42535031 ("BSP1")
    4       1     version     protocol version (1)
    5       1     kind        FrameKind
    6       4     length      payload bytes (n)
    10      2     header_crc  CRC-16/CCITT-FALSE over bytes 0..9
    12      n     payload
    12+n    4     crc         CRC-32 (IEEE 802.3) over bytes 0..12+n-1

Frame overhead is 16 bytes.  The header CRC protects the length field before it
is used to locate the trailer, so a corrupted length is always detected instead
of relying on a chance CRC-32 mismatch at the wrong offset.  Both CRCs detect
every single-bit and double-bit error at BSP frame sizes.

:class:`BoundedQueue` is the credit-based queue used at every stream endpoint:
a full queue refuses the item (backpressure) and counts the refusal; it never
drops or blocks.
"""

from __future__ import annotations

import binascii
import enum
import struct
import zlib
from collections import deque
from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

BSP_MAGIC = 0x42535031
BSP_VERSION = 1
NO_DEVICE = 0xFFFFFFFF

T = TypeVar("T")


class Opcode(enum.IntEnum):
    ALLOCATE = 1
    FREE = 2
    LOAD = 3
    TRANSFER = 4
    PREFETCH = 5
    EXECUTE = 6
    REDUCE = 7
    BARRIER = 8
    COMPLETE = 9
    CANCEL = 10
    QUERY_STATUS = 11


class Status(enum.IntEnum):
    PENDING = 0
    QUEUED = 1
    RUNNING = 2
    SUCCESS = 3
    FAILED = 4
    CANCELLED = 5
    TIMEOUT = 6
    PROTECTION_FAULT = 7
    VERSION_MISMATCH = 8
    CRC_ERROR = 9

    @property
    def is_terminal(self) -> bool:
        """True once the operation can no longer change state."""
        return self >= Status.SUCCESS


class Flags(enum.IntFlag):
    NONE = 0
    RETAIN = 1  # keep the transferred tensor as a replica at the destination
    STREAM = 2  # overlapped pipeline: source read, link, decode and MAC run concurrently
    ACCUMULATE = 4  # add the result into the output tensor (fused weighted reduction)
    SPECULATIVE = 8  # issued on a prediction; may be cancelled without affecting results
    STAGED = 16  # decode into scratch memory first, then compute
    SRAM = 32  # serve / fill the device-local SRAM tile cache


class FrameKind(enum.IntEnum):
    COMMAND = 1
    COMPLETION = 2
    ACTIVATION = 3
    WEIGHT_INDEX = 4


class ProtocolError(Exception):
    """Malformed, truncated or semantically invalid BSP data."""


class CrcError(ProtocolError):
    """A checksum did not match: the data was corrupted in transit or at rest."""


class QueueFull(ProtocolError):
    """A bounded queue had no free credit."""


_HEADER = struct.Struct("<IBBIH")
_TRAILER = struct.Struct("<I")
_HEADER_CRC_SPAN = _HEADER.size - 2
FRAME_OVERHEAD = _HEADER.size + _TRAILER.size  # 16 bytes
MAX_PAYLOAD = 0xFFFFFFFF


def crc32(data: bytes) -> int:
    """CRC-32 (IEEE 802.3) of ``data`` as an unsigned 32-bit integer."""
    return zlib.crc32(data) & 0xFFFFFFFF


def frame(kind: FrameKind | int, payload: bytes) -> bytes:
    """Wrap ``payload`` in a BSP frame of the given kind."""
    kind = FrameKind(kind)
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload of {len(payload)} bytes exceeds the frame limit")
    head = _HEADER.pack(BSP_MAGIC, BSP_VERSION, kind, len(payload), 0)[:_HEADER_CRC_SPAN]
    head += struct.pack("<H", binascii.crc_hqx(head, 0xFFFF))
    body = head + bytes(payload)
    return body + _TRAILER.pack(crc32(body))


def unframe(buf: bytes, offset: int = 0) -> tuple[FrameKind, bytes, int]:
    """Parse one frame at ``buf[offset:]``; returns ``(kind, payload, bytes consumed)``.

    Raises :class:`CrcError` on corruption and :class:`ProtocolError` on a bad
    magic number, unsupported version, unknown kind or truncated input.
    """
    view = memoryview(buf)[offset:]
    if len(view) < _HEADER.size:
        raise ProtocolError(f"truncated frame header: {len(view)} of {_HEADER.size} bytes")
    magic, version, kind, length, header_crc = _HEADER.unpack_from(view)
    if magic != BSP_MAGIC:
        raise ProtocolError(f"bad magic 0x{magic:08x}")
    if binascii.crc_hqx(view[:_HEADER_CRC_SPAN], 0xFFFF) != header_crc:
        raise CrcError("frame header checksum mismatch")
    if version != BSP_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    try:
        kind = FrameKind(kind)
    except ValueError:
        raise ProtocolError(f"unknown frame kind {kind}") from None
    end = _HEADER.size + length
    if len(view) < end + _TRAILER.size:
        raise ProtocolError(f"truncated frame: {len(view)} of {end + _TRAILER.size} bytes")
    if crc32(view[:end]) != _TRAILER.unpack_from(view, end)[0]:
        raise CrcError("frame checksum mismatch")
    return kind, bytes(view[_HEADER.size : end]), end + _TRAILER.size


def iter_frames(buf: bytes) -> Iterator[tuple[FrameKind, bytes]]:
    """Yield ``(kind, payload)`` for each frame of a buffer of concatenated frames."""
    offset = 0
    while offset < len(buf):
        kind, payload, consumed = unframe(buf, offset)
        offset += consumed
        yield kind, payload


def unframe_exact(buf: bytes, kind: FrameKind) -> bytes:
    """Payload of a buffer that must hold exactly one frame of ``kind``."""
    got, payload, consumed = unframe(buf)
    if got != kind:
        raise ProtocolError(f"expected a {kind.name} frame, got {got.name}")
    if consumed != len(buf):
        raise ProtocolError(f"{len(buf) - consumed} trailing bytes after {kind.name} frame")
    return payload


class BoundedQueue(Generic[T]):
    """FIFO queue with a fixed number of credits.

    ``enqueued == dequeued + removed + len(queue)`` always holds; ``rejected``
    counts backpressure events and ``high_water`` the peak occupancy.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._items: deque[T] = deque()
        self.enqueued = 0
        self.dequeued = 0
        self.removed = 0
        self.rejected = 0
        self.high_water = 0

    def try_put(self, item: T) -> bool:
        """Append ``item``; returns False (and counts a rejection) when full."""
        if len(self._items) >= self.capacity:
            self.rejected += 1
            return False
        self._items.append(item)
        self.enqueued += 1
        self.high_water = max(self.high_water, len(self._items))
        return True

    def put(self, item: T) -> None:
        if not self.try_put(item):
            raise QueueFull(f"queue full ({self.capacity} entries)")

    def get(self) -> T:
        item = self._items.popleft()  # IndexError when empty
        self.dequeued += 1
        return item

    def peek(self) -> T:
        return self._items[0]  # IndexError when empty

    def remove(self, pred: Callable[[T], bool]) -> list[T]:
        """Remove and return every queued item matching ``pred`` (order preserved)."""
        taken = [item for item in self._items if pred(item)]
        if taken:
            self._items = deque(item for item in self._items if not pred(item))
            self.removed += len(taken)
        return taken

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    @property
    def full(self) -> bool:
        return len(self._items) >= self.capacity

    @property
    def credits(self) -> int:
        """Free slots a sender may still consume."""
        return self.capacity - len(self._items)
