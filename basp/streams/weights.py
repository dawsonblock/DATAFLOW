"""Weight streams made of independently addressable blocks.

A stored weight tensor is a sequence of blocks, each decodable on its own and
protected by its own CRC-32, plus an index mapping weight ranges to byte ranges.
Reading one expert (or one row shard) therefore touches and decodes only its
own blocks.  The payload of a WEIGHT_INDEX frame is little-endian::

    offset  size  field
    0       8     tensor_id
    8       4     fmt        CompressionFormat
    12      4     n_blocks
    16      40*k  blocks: offset u64, length u64, first_weight u64, n_weights u64,
                          crc u32, block_id u32
"""

from __future__ import annotations

import struct
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from basp.core.tensor import CompressionFormat
from basp.streams.protocol import CrcError, FrameKind, ProtocolError, crc32, frame, unframe_exact

_HEAD = struct.Struct("<QII")
_BLOCK = struct.Struct("<QQQQII")


@dataclass(frozen=True)
class WeightBlock:
    block_id: int
    offset: int  # byte offset of the block in the stream
    length: int  # stored bytes
    first_weight: int  # index of the first weight held by the block
    n_weights: int
    crc: int  # CRC-32 of the stored bytes


@dataclass(frozen=True)
class WeightStreamIndex:
    """Block directory of one stored weight tensor."""

    tensor_id: int
    fmt: CompressionFormat
    blocks: tuple[WeightBlock, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fmt", CompressionFormat(self.fmt))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        weight = end = 0
        for i, blk in enumerate(self.blocks):
            if blk.block_id != i or blk.first_weight != weight or blk.n_weights < 1:
                raise ValueError(f"block {i}: ids and weight ranges must be contiguous")
            if blk.offset < end or blk.length < 0:
                raise ValueError(f"block {i}: byte ranges must be ordered and disjoint")
            weight += blk.n_weights
            end = blk.offset + blk.length

    @property
    def total_bytes(self) -> int:
        return sum(blk.length for blk in self.blocks)

    @property
    def n_weights(self) -> int:
        return self.blocks[-1].first_weight + self.blocks[-1].n_weights if self.blocks else 0

    def locate(self, first_weight: int, n_weights: int) -> tuple[WeightBlock, ...]:
        """The minimal run of blocks covering ``n_weights`` weights from ``first_weight``."""
        if first_weight < 0 or n_weights < 0 or first_weight + n_weights > self.n_weights:
            raise ValueError(
                f"weights [{first_weight}, {first_weight + n_weights}) outside 0..{self.n_weights}"
            )
        if n_weights == 0:
            return ()
        starts = [blk.first_weight for blk in self.blocks]
        lo = bisect_right(starts, first_weight) - 1
        hi = bisect_right(starts, first_weight + n_weights - 1)
        return self.blocks[lo:hi]

    def encode(self) -> bytes:
        parts = [_HEAD.pack(self.tensor_id, self.fmt, len(self.blocks))]
        parts += [
            _BLOCK.pack(b.offset, b.length, b.first_weight, b.n_weights, b.crc, b.block_id)
            for b in self.blocks
        ]
        return frame(FrameKind.WEIGHT_INDEX, b"".join(parts))

    @classmethod
    def decode(cls, buf: bytes) -> WeightStreamIndex:
        payload = unframe_exact(buf, FrameKind.WEIGHT_INDEX)
        if len(payload) < _HEAD.size:
            raise ProtocolError(f"weight index payload too short: {len(payload)} bytes")
        tensor_id, fmt, n_blocks = _HEAD.unpack_from(payload)
        if len(payload) != _HEAD.size + n_blocks * _BLOCK.size:
            raise ProtocolError(f"weight index declares {n_blocks} blocks, size disagrees")
        blocks = []
        for i in range(n_blocks):
            offset, length, first, count, crc, block_id = _BLOCK.unpack_from(
                payload, _HEAD.size + i * _BLOCK.size
            )
            blocks.append(WeightBlock(block_id, offset, length, first, count, crc))
        try:
            return cls(tensor_id, CompressionFormat(fmt), tuple(blocks))
        except ValueError as exc:
            raise ProtocolError(str(exc)) from None


def build_stream(
    tensor_id: int,
    fmt: CompressionFormat,
    payloads: Sequence[bytes],
    weights_per_block: Sequence[int] | int,
) -> tuple[WeightStreamIndex, bytes]:
    """Concatenate encoded blocks and build their index.

    ``weights_per_block`` is either one count shared by every block or one count
    per block (the last block of a tensor is often shorter).
    """
    if isinstance(weights_per_block, int):
        counts = [weights_per_block] * len(payloads)
    else:
        counts = list(weights_per_block)
    if len(counts) != len(payloads):
        raise ValueError(f"{len(payloads)} blocks but {len(counts)} weight counts")
    blocks, offset, weight = [], 0, 0
    for i, (data, count) in enumerate(zip(payloads, counts)):
        blocks.append(WeightBlock(i, offset, len(data), weight, count, crc32(data)))
        offset += len(data)
        weight += count
    return WeightStreamIndex(tensor_id, fmt, tuple(blocks)), b"".join(payloads)


def read_blocks(buf: bytes, index: WeightStreamIndex, block_ids: Iterable[int]) -> list[bytes]:
    """Stored bytes of the requested blocks, each verified against its CRC.

    Only the byte ranges of the requested blocks are read from ``buf`` (any
    object supporting ``len`` and slicing, e.g. bytes, a memoryview or an mmap).
    """
    out = []
    for block_id in block_ids:
        if not 0 <= block_id < len(index.blocks):
            raise ValueError(f"tensor {index.tensor_id} has no block {block_id}")
        blk = index.blocks[block_id]
        if blk.offset + blk.length > len(buf):
            raise ProtocolError(f"weight stream truncated before block {block_id}")
        data = bytes(buf[blk.offset : blk.offset + blk.length])
        if crc32(data) != blk.crc:
            raise CrcError(f"tensor {index.tensor_id} block {block_id}: checksum mismatch")
        out.append(data)
    return out


def read_range(
    buf: bytes, index: WeightStreamIndex, first_weight: int, n_weights: int
) -> list[bytes]:
    """Verified blocks covering a weight range (see :meth:`WeightStreamIndex.locate`)."""
    return read_blocks(buf, index, (blk.block_id for blk in index.locate(first_weight, n_weights)))
