"""Activation frames and independently addressable weight blocks."""

import random

import pytest

from basp.core import tensor
from basp.core.tensor import CompressionFormat, DType
from basp.streams.activation import ActivationFrame, Reassembler, activation_bytes, packetize
from basp.streams.protocol import CrcError, FrameKind, ProtocolError, crc32, frame
from basp.streams.weights import (
    WeightBlock,
    WeightStreamIndex,
    build_stream,
    read_blocks,
    read_range,
)


def pattern(n: int, seed: int = 0) -> bytes:
    return random.Random(seed).randbytes(n)


def flip(buf: bytes, bit: int) -> bytes:
    out = bytearray(buf)
    out[bit // 8] ^= 1 << (bit % 8)
    return bytes(out)


# --------------------------------------------------------------------------- activations


def test_activation_bytes_is_the_core_helper():
    assert activation_bytes is tensor.activation_bytes
    assert activation_bytes(4096) == 8192  # FP16 hidden state of the section-6 example
    assert activation_bytes(4096, DType.FP32) == 16384


@pytest.mark.parametrize(
    "size,mtu,frames",
    [(0, 4096, 1), (1, 4096, 1), (4096, 4096, 1), (4097, 4096, 2), (8192, 4096, 2), (10, 3, 4)],
)
def test_packetize_sizes(size, mtu, frames):
    data = pattern(size)
    out = packetize(7, 3, data, mtu)
    assert len(out) == frames
    assert [f.seq for f in out] == list(range(frames))
    assert all((f.tensor_id, f.version, f.total) == (7, 3, frames) for f in out)
    assert all(len(f.payload) <= mtu for f in out)
    assert all(len(f.payload) == mtu for f in out[:-1])
    assert b"".join(f.payload for f in out) == data


def test_packetize_defaults_and_errors():
    assert len(packetize(1, 0, bytes(8192))) == 2  # default MTU 4096
    with pytest.raises(ValueError):
        packetize(1, 0, b"abc", mtu=0)


def test_activation_frame_round_trip():
    for item in packetize(2**64 - 1, 2**64 - 1, pattern(9000), mtu=4096):
        buf = item.encode()
        assert len(buf) == len(item.payload) + 24 + 16
        assert ActivationFrame.decode(buf) == item and item.encode() == buf
    buf = ActivationFrame(1, 2, 0, 1, b"").encode()
    assert ActivationFrame.decode(buf).payload == b""
    for bit in range(len(buf) * 8):
        with pytest.raises(ProtocolError):
            ActivationFrame.decode(flip(buf, bit))


def test_activation_frame_rejects_bad_fields():
    with pytest.raises(ProtocolError):
        ActivationFrame(1, 0, 2, 2, b"x").encode()  # seq == total
    with pytest.raises(ProtocolError):
        ActivationFrame(1, 0, 0, 0, b"x").encode()  # total == 0
    with pytest.raises(ProtocolError):
        ActivationFrame(-1, 0, 0, 1, b"x").encode()
    with pytest.raises(ProtocolError):
        ActivationFrame.decode(frame(FrameKind.ACTIVATION, bytes(23)))
    with pytest.raises(ProtocolError):
        ActivationFrame.decode(frame(FrameKind.COMMAND, bytes(72)))
    bad = frame(
        FrameKind.ACTIVATION, (1).to_bytes(8, "little") + bytes(8) + bytes([5, 0, 0, 0, 2, 0, 0, 0])
    )
    with pytest.raises(ProtocolError):
        ActivationFrame.decode(bad)  # well framed, but seq 5 of 2


@pytest.mark.parametrize("order", ["forward", "reverse", "shuffled"])
def test_reassembly_in_any_order(order):
    data = pattern(20_000, seed=1)
    frames = packetize(11, 1, data, mtu=1500)
    if order == "reverse":
        frames.reverse()
    elif order == "shuffled":
        random.Random(5).shuffle(frames)
    r = Reassembler()
    results = [r.add(f) for f in frames]
    assert results[:-1] == [None] * (len(frames) - 1)
    assert results[-1] == data
    assert r.pending() == 0


def test_reassembly_through_the_wire():
    data = pattern(10_000, seed=2)
    wire = [f.encode() for f in packetize(4, 9, data, mtu=4096)]
    r = Reassembler()
    out = [r.add(ActivationFrame.decode(buf)) for buf in reversed(wire)]
    assert out[-1] == data


def test_duplicates_are_ignored():
    data = pattern(5000, seed=3)
    frames = packetize(1, 0, data, mtu=1000)
    r = Reassembler()
    assert r.add(frames[0]) is None
    assert r.add(frames[0]) is None  # duplicate before completion
    assert r.duplicates == 1 and r.pending() == 1
    for f in frames[1:-1]:
        assert r.add(f) is None
    assert r.add(frames[-1]) == data
    assert r.add(frames[2]) is None  # duplicate after completion does not restart the tensor
    assert r.duplicates == 2 and r.pending() == 0


def test_single_and_empty_frames_complete_immediately():
    r = Reassembler()
    assert r.add(packetize(1, 0, b"")[0]) == b""
    assert r.add(packetize(2, 0, b"abc")[0]) == b"abc"
    assert r.pending() == 0


def test_inconsistent_frames_are_rejected():
    r = Reassembler()
    frames = packetize(1, 0, bytes(3000), mtu=1000)
    r.add(frames[0])
    with pytest.raises(ProtocolError, match="disagrees"):
        r.add(ActivationFrame(1, 0, 1, 4, b"x"))  # total changed from 3 to 4
    with pytest.raises(ProtocolError):
        r.add(ActivationFrame(1, 0, 3, 3, b"x"))  # seq >= total
    with pytest.raises(ProtocolError):
        r.add(ActivationFrame(2, 0, 0, 0, b"x"))  # total must be >= 1
    assert r.pending() == 1  # the valid partial tensor is unaffected
    assert r.add(frames[1]) is None and r.add(frames[2]) == bytes(3000)


def test_version_rule():
    r = Reassembler()
    old = packetize(1, 4, pattern(3000, 4), mtu=1000)
    new = packetize(1, 5, pattern(3000, 5), mtu=1000)
    assert r.add(old[0]) is None
    assert r.add(new[0]) is None  # a newer version supersedes the incomplete older one
    assert r.superseded == 1 and r.pending() == 1
    with pytest.raises(ProtocolError, match="stale"):
        r.add(old[1])
    assert r.add(new[2]) is None and r.add(new[1]) == pattern(3000, 5)
    with pytest.raises(ProtocolError, match="stale"):
        r.add(old[2])  # still stale after completion
    newest = packetize(1, 6, b"next")
    assert r.add(newest[0]) == b"next"
    r.forget(1)
    assert r.add(old[0]) is None  # a freed tensor id may be reused from any version


def test_interleaved_tensors():
    rng = random.Random(8)
    payloads = {tid: pattern(2000 + 700 * tid, tid) for tid in range(6)}
    frames = [f for tid, data in payloads.items() for f in packetize(tid, tid % 2, data, mtu=512)]
    rng.shuffle(frames)
    r = Reassembler()
    done = {}
    for f in frames:
        out = r.add(f)
        if out is not None:
            assert f.tensor_id not in done
            done[f.tensor_id] = out
    assert done == payloads and r.pending() == 0 and r.duplicates == 0


# --------------------------------------------------------------------------- weight blocks


class RecordingBuffer:
    """Bytes-like object that records which byte ranges were read."""

    def __init__(self, data: bytes):
        self._data = data
        self.reads: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: slice) -> bytes:
        self.reads.append((key.start, key.stop))
        return self._data[key]


def expert_stream(n_blocks=6, weights=1000, tensor_id=42, fmt=CompressionFormat.INT4):
    """Blocks of different stored sizes, as an entropy coder would produce."""
    payloads = [pattern(300 + 37 * i, seed=i) for i in range(n_blocks)]
    index, stream = build_stream(tensor_id, fmt, payloads, weights)
    return payloads, index, stream


def test_build_stream_with_uniform_blocks():
    payloads, index, stream = expert_stream()
    assert stream == b"".join(payloads)
    assert index.total_bytes == len(stream) and index.n_weights == 6000
    assert (index.tensor_id, index.fmt) == (42, CompressionFormat.INT4)
    offset = 0
    for i, blk in enumerate(index.blocks):
        assert blk == WeightBlock(i, offset, len(payloads[i]), 1000 * i, 1000, crc32(payloads[i]))
        offset += blk.length


def test_build_stream_with_per_block_counts():
    payloads = [b"aaaa", b"bb", b"c"]
    index, stream = build_stream(1, CompressionFormat.ENTROPY, payloads, [64, 64, 10])
    assert [b.first_weight for b in index.blocks] == [0, 64, 128]
    assert index.n_weights == 138 and index.total_bytes == 7 and stream == b"aaaabbc"
    with pytest.raises(ValueError):
        build_stream(1, CompressionFormat.ENTROPY, payloads, [64, 64])
    with pytest.raises(ValueError):
        build_stream(1, CompressionFormat.ENTROPY, payloads, 0)  # empty blocks are meaningless
    empty, data = build_stream(1, CompressionFormat.INT4, [], 32)
    assert (empty.blocks, empty.n_weights, empty.total_bytes, data) == ((), 0, 0, b"")
    assert empty.locate(0, 0) == ()


def test_locate_edge_cases():
    _, index, _ = expert_stream(n_blocks=4, weights=100)
    ids = lambda first, n: [b.block_id for b in index.locate(first, n)]
    assert ids(0, 400) == [0, 1, 2, 3]
    assert ids(0, 1) == [0] and ids(99, 1) == [0] and ids(100, 1) == [1]
    assert ids(399, 1) == [3]
    assert ids(99, 2) == [0, 1]  # straddles a block boundary
    assert ids(100, 100) == [1] and ids(100, 101) == [1, 2]
    assert ids(150, 200) == [1, 2, 3]
    assert ids(0, 0) == [] and ids(400, 0) == []
    for first, n in [(-1, 1), (0, 401), (400, 1), (399, 2), (0, -1), (500, 0)]:
        with pytest.raises(ValueError):
            index.locate(first, n)


def test_locate_is_minimal_and_covering():
    counts = [1, 5, 32, 7, 64, 3]
    index, _ = build_stream(1, CompressionFormat.INT4, [b"x"] * len(counts), counts)
    owner = [blk.block_id for blk in index.blocks for _ in range(blk.n_weights)]
    for first in range(len(owner)):
        for n in range(1, len(owner) - first + 1):
            got = [b.block_id for b in index.locate(first, n)]
            assert got == sorted(set(owner[first : first + n]))


def test_index_round_trip():
    _, index, _ = expert_stream()
    buf = index.encode()
    assert len(buf) == 16 + 16 + 40 * 6
    assert WeightStreamIndex.decode(buf) == index and index.encode() == buf
    empty, _ = build_stream(2**64 - 1, CompressionFormat.PREDICTIVE, [], 1)
    assert WeightStreamIndex.decode(empty.encode()) == empty
    for bit in range(0, len(buf) * 8, 7):
        with pytest.raises(ProtocolError):
            WeightStreamIndex.decode(flip(buf, bit))
    with pytest.raises(ProtocolError):
        WeightStreamIndex.decode(frame(FrameKind.WEIGHT_INDEX, bytes(16 + 39)))
    with pytest.raises(ProtocolError):  # unknown format code, but valid checksums
        WeightStreamIndex.decode(
            frame(FrameKind.WEIGHT_INDEX, bytes(8) + bytes([99, 0, 0, 0]) + bytes(4))
        )


def test_index_validation():
    good = WeightBlock(0, 0, 10, 0, 32, 0)
    WeightStreamIndex(
        1, CompressionFormat.INT4, (good, WeightBlock(1, 16, 10, 32, 32, 0))
    )  # padded
    for second in (
        WeightBlock(2, 10, 10, 32, 32, 0),  # block id gap
        WeightBlock(1, 10, 10, 40, 32, 0),  # weight gap
        WeightBlock(1, 10, 10, 32, 0, 0),  # empty block
        WeightBlock(1, 9, 10, 32, 32, 0),  # overlapping bytes
    ):
        with pytest.raises(ValueError):
            WeightStreamIndex(1, CompressionFormat.INT4, (good, second))


def test_read_blocks_touches_only_requested_ranges():
    payloads, index, stream = expert_stream()
    buf = RecordingBuffer(stream)
    assert read_blocks(buf, index, [4, 1]) == [payloads[4], payloads[1]]
    wanted = [
        (index.blocks[i].offset, index.blocks[i].offset + index.blocks[i].length) for i in (4, 1)
    ]
    assert buf.reads == wanted
    assert sum(stop - start for start, stop in buf.reads) == len(payloads[4]) + len(payloads[1])
    assert read_blocks(stream, index, []) == []
    assert read_blocks(memoryview(stream), index, range(6)) == payloads
    with pytest.raises(ValueError):
        read_blocks(stream, index, [6])
    with pytest.raises(ValueError):
        read_blocks(stream, index, [-1])


def test_read_range_reads_one_expert_only():
    payloads, index, stream = expert_stream()
    buf = RecordingBuffer(stream)
    assert read_range(buf, index, 2000, 1000) == [payloads[2]]  # exactly "expert" 2
    assert buf.reads == [(index.blocks[2].offset, index.blocks[2].offset + index.blocks[2].length)]
    assert read_range(stream, index, 2999, 2) == [payloads[2], payloads[3]]


def test_block_corruption_is_detected_and_contained():
    payloads, index, stream = expert_stream()
    victim = index.blocks[3]
    for bit in (0, victim.length * 4, victim.length * 8 - 1):
        bad = flip(stream, victim.offset * 8 + bit)
        with pytest.raises(CrcError, match="block 3"):
            read_blocks(bad, index, [3])
        # Other blocks stay readable: corruption does not spread across block boundaries.
        assert read_blocks(bad, index, [0, 1, 2, 4, 5]) == payloads[:3] + payloads[4:]
    with pytest.raises(ProtocolError, match="truncated"):
        read_blocks(stream[:-1], index, [5])
    assert read_blocks(stream[:-1], index, [0, 4]) == [payloads[0], payloads[4]]


def test_formats_survive_the_index():
    for fmt in CompressionFormat:
        index, _ = build_stream(5, fmt, [b"abc", b"de"], 32)
        decoded = WeightStreamIndex.decode(index.encode())
        assert decoded.fmt is fmt and type(decoded.fmt) is CompressionFormat
