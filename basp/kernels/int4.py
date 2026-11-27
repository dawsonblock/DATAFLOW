"""Signed INT4 weight format with per-group FP16 scales (4.5 bits/weight at group size 32).

Conventions
-----------
* A weight matrix is ``[rows, cols]``. Quantization groups run along the columns of a row,
  so every row is an independently decodable block: rows can be tiled, sharded and streamed.
* Codes are two's-complement nibbles in [-8, 7], two per byte, low nibble = even column.
* Columns are zero-padded to a multiple of ``group_size``; padded codes are 0.
* A group decodes as ``code * scale`` with ``scale = max|w| / 7`` rounded to FP16. The
  rounded FP16 scale is the one used to choose the codes, so encode and decode agree and
  ``|w - decode(w)| <= scale / 2`` holds for every weight.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

GROUP_SIZE = 32

_MAGIC = b"BI4T"
_VERSION = 1
_HEADER = struct.Struct("<4sHHQQ")  # magic, format version, group_size, rows, cols


def _padded(cols: int, group_size: int) -> int:
    return -(-cols // group_size) * group_size


@dataclass(frozen=True, eq=False)
class Int4Tensor:
    """Packed INT4 codes plus FP16 group scales for a ``[rows, cols]`` weight matrix."""

    packed: np.ndarray  # uint8 [rows, padded_cols // 2]
    scales: np.ndarray  # float16 [rows, padded_cols // group_size]
    rows: int
    cols: int
    group_size: int = GROUP_SIZE

    def __post_init__(self) -> None:
        if self.group_size <= 0 or self.group_size % 2:
            raise ValueError(f"group_size must be a positive even number, got {self.group_size}")
        if self.rows < 0 or self.cols < 0:
            raise ValueError(f"negative shape ({self.rows}, {self.cols})")
        want_packed = (self.rows, self.padded_cols // 2)
        want_scales = (self.rows, self.padded_cols // self.group_size)
        if self.packed.dtype != np.uint8 or self.packed.shape != want_packed:
            raise ValueError(f"packed must be uint8 {want_packed}, got "
                             f"{self.packed.dtype} {self.packed.shape}")
        if self.scales.dtype != np.float16 or self.scales.shape != want_scales:
            raise ValueError(f"scales must be float16 {want_scales}, got "
                             f"{self.scales.dtype} {self.scales.shape}")

    @property
    def padded_cols(self) -> int:
        return _padded(self.cols, self.group_size)

    @property
    def n_groups(self) -> int:
        """Quantization groups per row."""
        return self.padded_cols // self.group_size

    @property
    def n_weights(self) -> int:
        return self.rows * self.cols

    @property
    def nbytes(self) -> int:
        """Stored bytes: packed codes plus scales (what a kernel must read)."""
        return self.packed.nbytes + self.scales.nbytes

    @property
    def bits_per_weight(self) -> float:
        """Effective storage per logical weight; exactly 4.5 when ``cols % 32 == 0``."""
        return 8.0 * self.nbytes / self.n_weights if self.n_weights else 0.0

    def row_slice(self, start: int, stop: int) -> Int4Tensor:
        """Rows ``[start, stop)`` as a zero-copy view (an independently addressable block)."""
        if not 0 <= start <= stop <= self.rows:
            raise IndexError(f"row range [{start}, {stop}) outside [0, {self.rows}]")
        return Int4Tensor(self.packed[start:stop], self.scales[start:stop],
                          stop - start, self.cols, self.group_size)

    def to_bytes(self) -> bytes:
        """Serialize as a 24-byte header, then scales (little-endian FP16), then codes."""
        header = _HEADER.pack(_MAGIC, _VERSION, self.group_size, self.rows, self.cols)
        return header + self.scales.astype("<f2").tobytes() + self.packed.tobytes()

    @classmethod
    def from_bytes(cls, buf: bytes) -> Int4Tensor:
        """Inverse of :meth:`to_bytes`; the arrays are read-only views of ``buf``."""
        if len(buf) < _HEADER.size:
            raise ValueError("buffer shorter than Int4Tensor header")
        magic, version, group_size, rows, cols = _HEADER.unpack_from(buf)
        if magic != _MAGIC or version != _VERSION:
            raise ValueError(f"not an Int4Tensor v{_VERSION} buffer (magic={magic!r}, "
                             f"version={version})")
        if group_size == 0 or group_size % 2:
            raise ValueError(f"invalid group_size {group_size} in header")
        padded = _padded(cols, group_size)
        n_scales, n_packed = rows * (padded // group_size), rows * (padded // 2)
        if len(buf) != _HEADER.size + 2 * n_scales + n_packed:
            raise ValueError("buffer length does not match Int4Tensor header")
        scales = np.frombuffer(buf, dtype="<f2", count=n_scales, offset=_HEADER.size)
        packed = np.frombuffer(buf, dtype=np.uint8, count=n_packed,
                               offset=_HEADER.size + 2 * n_scales)
        return cls(packed.reshape(rows, padded // 2),
                   scales.astype(np.float16, copy=False).reshape(rows, padded // group_size),
                   rows, cols, group_size)


def pack_nibbles(q: np.ndarray) -> np.ndarray:
    """Pack int8 codes in [-8, 7] two per byte along the last axis (low nibble first)."""
    q = np.asarray(q)
    if q.dtype != np.int8:
        raise TypeError(f"codes must be int8, got {q.dtype}")
    if q.shape[-1] % 2:
        raise ValueError("last axis must have even length to pack nibble pairs")
    if q.size and (q.min() < -8 or q.max() > 7):
        raise ValueError("codes outside the signed INT4 range [-8, 7]")
    nibbles = q.view(np.uint8) & 0x0F
    return nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_nibbles`: fixed-width parallel unpack with sign extension."""
    packed = np.asarray(packed)
    if packed.dtype != np.uint8:
        raise TypeError(f"packed codes must be uint8, got {packed.dtype}")
    out = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.int8)
    out[..., 0::2] = ((packed & 0x0F) ^ 8).astype(np.int8) - 8
    out[..., 1::2] = ((packed >> 4) ^ 8).astype(np.int8) - 8
    return out


def quantize_int4(w, group_size: int = GROUP_SIZE) -> Int4Tensor:
    """Quantize a ``[rows, cols]`` float matrix to INT4 codes with symmetric FP16 group scales."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"weights must be [rows, cols], got shape {w.shape}")
    if group_size <= 0 or group_size % 2:
        raise ValueError(f"group_size must be a positive even number, got {group_size}")
    rows, cols = w.shape
    padded = _padded(cols, group_size)
    grouped = np.zeros((rows, padded), dtype=np.float64)
    grouped[:, :cols] = w
    grouped = grouped.reshape(rows, padded // group_size, group_size)
    peak = np.abs(grouped).max(axis=2)
    if not np.all(np.isfinite(peak)):
        raise ValueError("weights must be finite")
    if np.any(peak / 7.0 > float(np.finfo(np.float16).max)):
        raise ValueError("weight magnitude exceeds the FP16 scale range")
    scales = (peak / 7.0).astype(np.float16)
    # Rounding to FP16 can leave a scale too small to reach the group peak within half a
    # step; that only happens for subnormal scales, where one ulp up restores the bound.
    short = scales.astype(np.float64) * 7.5 < peak
    scales = np.where(short, np.nextafter(scales, np.float16(np.inf)), scales)
    step = scales.astype(np.float64)[:, :, None]
    codes = np.rint(np.divide(grouped, step, out=np.zeros_like(grouped), where=step > 0))
    codes = np.clip(codes, -8, 7).astype(np.int8).reshape(rows, padded)
    return Int4Tensor(pack_nibbles(codes), np.ascontiguousarray(scales), rows, cols, group_size)


def dequantize(t: Int4Tensor) -> np.ndarray:
    """Expand to float32 ``[rows, cols]``. Each ``code * scale`` product is exact in FP32."""
    codes = unpack_nibbles(t.packed).reshape(t.rows, t.n_groups, t.group_size)
    w = codes.astype(np.float32) * t.scales.astype(np.float32)[:, :, None]
    return np.ascontiguousarray(w.reshape(t.rows, t.padded_cols)[:, : t.cols])


def quantize_activations_int8(x) -> tuple[np.ndarray, np.float32]:
    """Symmetric INT8 activation quantization: ``x ~= q * scale`` with ``scale = max|x| / 127``."""
    x = np.asarray(x, dtype=np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("activations must be finite")
    scale = np.float32(np.abs(x).max() / 127.0) if x.size else np.float32(0.0)
    if not scale > 0:
        return np.zeros(x.shape, dtype=np.int8), np.float32(0.0)
    q = np.rint(x.astype(np.float64) / np.float64(scale))
    return np.clip(q, -127, 127).astype(np.int8), scale


def symbol_entropy_bits(t: Int4Tensor) -> float:
    """Shannon entropy of the INT4 codes in bits/weight (padding excluded).

    This is the order-0 lower bound for lossless entropy coding of the codes; the FP16
    scales (0.5 bits/weight at group size 32) are not included.
    """
    if t.n_weights == 0:
        return 0.0
    codes = unpack_nibbles(t.packed)[:, : t.cols]
    counts = np.bincount(codes.ravel().astype(np.int64) + 8, minlength=16)
    p = counts[counts > 0] / codes.size
    return float(-(p * np.log2(p)).sum())
