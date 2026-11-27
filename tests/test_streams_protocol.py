"""BSP framing, enumerations and the bounded queue."""

import binascii
import struct

import pytest

from basp.streams.protocol import (
    BSP_MAGIC,
    BSP_VERSION,
    FRAME_OVERHEAD,
    NO_DEVICE,
    BoundedQueue,
    CrcError,
    Flags,
    FrameKind,
    Opcode,
    ProtocolError,
    QueueFull,
    Status,
    crc32,
    frame,
    iter_frames,
    unframe,
)


def raw_frame(kind: int, payload: bytes, version: int = BSP_VERSION) -> bytes:
    """Build a frame with valid checksums but arbitrary header fields."""
    head = struct.pack("<IBBI", BSP_MAGIC, version, kind, len(payload))
    body = head + struct.pack("<H", binascii.crc_hqx(head, 0xFFFF)) + payload
    return body + struct.pack("<I", crc32(body))


def flip(buf: bytes, bit: int) -> bytes:
    out = bytearray(buf)
    out[bit // 8] ^= 1 << (bit % 8)
    return bytes(out)


def test_spec_constants():
    assert BSP_MAGIC == 0x42535031 and struct.pack(">I", BSP_MAGIC) == b"BSP1"
    assert BSP_VERSION == 1 and NO_DEVICE == 0xFFFFFFFF
    assert {op.name: op.value for op in Opcode} == {
        "ALLOCATE": 1, "FREE": 2, "LOAD": 3, "TRANSFER": 4, "PREFETCH": 5, "EXECUTE": 6,
        "REDUCE": 7, "BARRIER": 8, "COMPLETE": 9, "CANCEL": 10, "QUERY_STATUS": 11,
    }  # fmt: skip
    assert [s.name for s in Status] == [
        "PENDING", "QUEUED", "RUNNING", "SUCCESS", "FAILED", "CANCELLED", "TIMEOUT",
        "PROTECTION_FAULT", "VERSION_MISMATCH", "CRC_ERROR",
    ]  # fmt: skip
    assert [s.value for s in Status] == list(range(10))
    assert {f.name: f.value for f in Flags if f} == {
        "RETAIN": 1, "STREAM": 2, "ACCUMULATE": 4, "SPECULATIVE": 8, "STAGED": 16, "SRAM": 32,
    }  # fmt: skip
    assert [k.value for k in FrameKind] == [1, 2, 3, 4]


def test_status_terminal():
    assert [s.name for s in Status if not s.is_terminal] == ["PENDING", "QUEUED", "RUNNING"]
    assert Status.SUCCESS.is_terminal and Status.CRC_ERROR.is_terminal


def test_crc_reference_vectors():
    assert crc32(b"123456789") == 0xCBF43926
    assert crc32(b"") == 0
    assert binascii.crc_hqx(b"123456789", 0xFFFF) == 0x29B1  # CRC-16/CCITT-FALSE


@pytest.mark.parametrize("kind", list(FrameKind))
@pytest.mark.parametrize("size", [0, 1, 7, 64, 4097])
def test_frame_round_trip(kind, size):
    payload = bytes((i * 31 + size) & 0xFF for i in range(size))
    buf = frame(kind, payload)
    assert len(buf) == size + FRAME_OVERHEAD == size + 16
    assert buf == raw_frame(kind, payload)
    assert unframe(buf) == (kind, payload, len(buf))
    assert frame(kind, payload) == buf  # pure function


def test_frame_accepts_bytes_like():
    buf = frame(FrameKind.ACTIVATION, bytearray(b"abc"))
    for view in (bytearray(buf), memoryview(buf)):
        assert unframe(view) == (FrameKind.ACTIVATION, b"abc", len(buf))


def test_concatenated_frames():
    payloads = [b"", b"x", b"hello world", bytes(300)]
    kinds = [FrameKind.COMMAND, FrameKind.COMPLETION, FrameKind.ACTIVATION, FrameKind.WEIGHT_INDEX]
    buf = b"".join(frame(k, p) for k, p in zip(kinds, payloads))
    assert list(iter_frames(buf)) == list(zip(kinds, payloads))
    offset = 0
    for kind, payload in zip(kinds, payloads):
        got_kind, got_payload, consumed = unframe(buf, offset)
        assert (got_kind, got_payload) == (kind, payload)
        offset += consumed
    assert offset == len(buf)
    assert list(iter_frames(b"")) == []


def test_every_truncation_is_rejected():
    buf = frame(FrameKind.COMMAND, b"payload-bytes")
    for cut in range(len(buf)):
        with pytest.raises(ProtocolError):
            unframe(buf[:cut])
    with pytest.raises(ProtocolError):  # a stream ending inside its last frame
        list(iter_frames(buf + buf[:-1]))


def test_every_single_bit_flip_is_detected():
    buf = frame(FrameKind.COMMAND, bytes(range(40)))
    for bit in range(len(buf) * 8):
        with pytest.raises(ProtocolError):
            unframe(flip(buf, bit))


def test_corrupted_length_is_detected_even_with_trailing_data():
    """A shortened length must not let the parser resynchronise on payload bytes."""
    first = frame(FrameKind.ACTIVATION, bytes(64))
    stream = first + frame(FrameKind.ACTIVATION, bytes(64))
    for bit in range(6 * 8, 10 * 8):  # the length field
        with pytest.raises(CrcError):
            unframe(flip(stream, bit))


def test_payload_and_trailer_corruption_raise_crc_error():
    buf = frame(FrameKind.COMPLETION, b"0123456789")
    for bit in range(12 * 8, len(buf) * 8):
        with pytest.raises(CrcError):
            unframe(flip(buf, bit))


def test_header_field_errors():
    with pytest.raises(ProtocolError, match="magic") as bad_magic:
        unframe(b"XXXX" + frame(FrameKind.COMMAND, b"")[4:])
    assert not isinstance(bad_magic.value, CrcError)
    with pytest.raises(ProtocolError, match="version") as bad_version:
        unframe(raw_frame(FrameKind.COMMAND, b"abc", version=2))
    assert not isinstance(bad_version.value, CrcError)
    with pytest.raises(ProtocolError, match="kind") as bad_kind:
        unframe(raw_frame(9, b"abc"))
    assert not isinstance(bad_kind.value, CrcError)
    with pytest.raises(ValueError):
        frame(9, b"abc")
    assert issubclass(CrcError, ProtocolError) and issubclass(QueueFull, ProtocolError)


def test_bounded_queue_fifo_and_backpressure():
    q = BoundedQueue(3)
    assert (len(q), q.credits, q.full) == (0, 3, False)
    assert all(q.try_put(i) for i in (1, 2, 3))
    assert (len(q), q.credits, q.full, q.high_water) == (3, 0, True, 3)
    assert q.try_put(4) is False
    with pytest.raises(QueueFull):
        q.put(5)
    assert (q.enqueued, q.rejected, len(q)) == (3, 2, 3)  # refused items are not stored
    assert q.peek() == 1 and len(q) == 3
    assert list(q) == [1, 2, 3]
    assert q.get() == 1 and q.credits == 1
    q.put(6)
    assert [q.get() for _ in range(3)] == [2, 3, 6]
    assert (q.enqueued, q.dequeued, q.rejected, q.high_water) == (4, 4, 2, 3)
    with pytest.raises(IndexError):
        q.get()
    with pytest.raises(IndexError):
        q.peek()
    with pytest.raises(ValueError):
        BoundedQueue(0)


def test_bounded_queue_remove_for_cancel():
    q = BoundedQueue(8)
    for i in range(6):
        q.put(i)
    assert q.remove(lambda v: v % 2 == 1) == [1, 3, 5]
    assert list(q) == [0, 2, 4] and q.credits == 5
    assert q.remove(lambda v: v > 100) == []
    assert q.enqueued == q.dequeued + q.removed + len(q) == 6
    assert q.high_water == 6  # peak occupancy is not reduced by removals
    q.put(9)
    assert [q.get() for _ in range(4)] == [0, 2, 4, 9]
    assert q.enqueued == q.dequeued + q.removed + len(q) == 7
