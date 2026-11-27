"""Fused streaming INT4 kernels: decoder -> MAC lanes -> accumulators, one row tile at a time.

Each kernel walks the packed weights in tiles of ``tile_rows`` rows, unpacks a tile straight
into the multiply-accumulate and discards it. The expanded weight matrix never exists as a
whole; ``KernelStats.peak_expanded_bytes`` records the largest decoded tile.

Arithmetic per output row (fixed, so results do not depend on tiling or batch size):

* FP path: activations are rounded to FP16. A lane product ``code * x`` is exact in FP32
  (4-bit times 11-bit significand). Lane products are summed within a group in ascending
  column order, the group sum is multiplied by its FP16 scale, and the scaled group sums are
  accumulated in ascending group order. Every step is one FP32 operation.
* Integer path: exact INT32 group sums of ``int4 * int8`` products (|sum| <= 32768 at group
  size 32, exact in FP32), each multiplied by its FP16 scale and accumulated in ascending
  group order in FP32; the row total is multiplied by the activation scale last.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .int4 import Int4Tensor, unpack_nibbles


@dataclass
class KernelStats:
    """Work counters; pass the same object to several calls to accumulate."""

    weight_bytes_read: int = 0  # packed codes + scales pulled through the decoder
    macs: int = 0  # logical multiply-accumulates: rows * cols * batch
    tiles: int = 0
    peak_expanded_bytes: int = 0  # largest decoded (int8) tile alive at once


def _tiles(t: Int4Tensor, tile_rows: int,
           stats: KernelStats | None) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield ``(start, stop, codes int8 [n, groups, group_size], scales FP32 [n, groups])``."""
    if tile_rows < 1:
        raise ValueError(f"tile_rows must be >= 1, got {tile_rows}")
    for start in range(0, t.rows, tile_rows):
        stop = min(start + tile_rows, t.rows)
        packed, scales = t.packed[start:stop], t.scales[start:stop]
        codes = unpack_nibbles(packed).reshape(stop - start, t.n_groups, t.group_size)
        if stats is not None:
            stats.weight_bytes_read += packed.nbytes + scales.nbytes
            stats.tiles += 1
            stats.peak_expanded_bytes = max(stats.peak_expanded_bytes, codes.nbytes)
        yield start, stop, codes, scales.astype(np.float32)


def _accumulate_groups(group_sums: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """FP32 accumulation of ``group_sum * scale`` over the last axis in ascending order."""
    acc = np.zeros(group_sums.shape[:-1], dtype=np.float32)
    for g in range(group_sums.shape[-1]):
        acc += group_sums[..., g] * scales[:, g]
    return acc


def _int8_lanes(t: Int4Tensor, x_q) -> np.ndarray:
    x_q = np.asarray(x_q)
    if x_q.dtype != np.int8 or x_q.shape != (t.cols,):
        raise ValueError(f"x_q must be int8 [{t.cols}], got {x_q.dtype} {x_q.shape}")
    lanes = np.zeros(t.padded_cols, dtype=np.int32)
    lanes[: t.cols] = x_q
    return lanes.reshape(t.n_groups, t.group_size)


def matmul_int4(t: Int4Tensor, x, *, tile_rows: int = 64,
                stats: KernelStats | None = None) -> np.ndarray:
    """FP path for a batch: ``x`` [batch, cols] -> float32 [batch, rows]."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != t.cols:
        raise ValueError(f"x must be [batch, {t.cols}], got shape {x.shape}")
    batch = x.shape[0]
    lanes = np.zeros((batch, t.padded_cols), dtype=np.float32)
    lanes[:, : t.cols] = x.astype(np.float16)
    lanes = lanes.reshape(batch, 1, t.n_groups, t.group_size)
    out = np.empty((batch, t.rows), dtype=np.float32)
    for start, stop, codes, scales in _tiles(t, tile_rows, stats):
        weights = codes.astype(np.float32)[None]
        sums = np.zeros((batch, stop - start, t.n_groups), dtype=np.float32)
        for lane in range(t.group_size):
            sums += weights[..., lane] * lanes[..., lane]
        out[:, start:stop] = _accumulate_groups(sums, scales)
    if stats is not None:
        stats.macs += batch * t.n_weights
    return out


def matvec_int4(t: Int4Tensor, x, *, tile_rows: int = 64,
                stats: KernelStats | None = None) -> np.ndarray:
    """FP path for one activation: ``x`` [cols] -> float32 [rows]."""
    x = np.asarray(x)
    if x.shape != (t.cols,):
        raise ValueError(f"x must be [{t.cols}], got shape {x.shape}")
    return matmul_int4(t, x[None, :], tile_rows=tile_rows, stats=stats)[0]


def group_sums_int32(t: Int4Tensor, x_q) -> np.ndarray:
    """Exact INT32 group sums ``sum_j code[r, g, j] * x_q[g, j]`` -> int32 [rows, groups]."""
    lanes = _int8_lanes(t, x_q)
    out = np.empty((t.rows, t.n_groups), dtype=np.int32)
    for start, stop, codes, _ in _tiles(t, 64, None):
        out[start:stop] = np.einsum("rgj,gj->rg", codes.astype(np.int32), lanes)
    return out


def matvec_int4_int8(t: Int4Tensor, x_q, x_scale, *, tile_rows: int = 64,
                     stats: KernelStats | None = None) -> np.ndarray:
    """Integer path: INT8 activations ``x_q * x_scale`` -> float32 [rows]."""
    lanes = _int8_lanes(t, x_q)
    out = np.empty(t.rows, dtype=np.float32)
    for start, stop, codes, scales in _tiles(t, tile_rows, stats):
        sums = np.einsum("rgj,gj->rg", codes.astype(np.int32), lanes)
        out[start:stop] = _accumulate_groups(sums.astype(np.float32), scales)
    out *= np.float32(x_scale)
    if stats is not None:
        stats.macs += t.n_weights
    return out
