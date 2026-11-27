"""Differential tests of the streaming INT4 kernels against the float64 reference."""

import math
from itertools import pairwise

import numpy as np
import pytest

from basp.kernels import reference
from basp.kernels.int4 import (
    Int4Tensor,
    dequantize,
    pack_nibbles,
    quantize_activations_int8,
    quantize_int4,
    unpack_nibbles,
)
from basp.kernels.matvec import (
    KernelStats,
    group_sums_int32,
    matmul_int4,
    matvec_int4,
    matvec_int4_int8,
)

U32 = 2.0**-24  # FP32 unit roundoff
U16 = 2.0**-11  # FP16 unit roundoff
SHAPES = [(48, 64), (33, 96), (7, 50), (130, 256)]


def _case(seed, rows, cols):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(rows, cols)).astype(np.float32) * 10.0 ** rng.uniform(-2, 1, (rows, 1))
    return rng, w, quantize_int4(w)


def _fp32_tol(t: Int4Tensor, abs_w: np.ndarray, abs_x: np.ndarray) -> np.ndarray:
    """Bound on FP32 accumulation error of one kernel row.

    The kernel adds ``group_size`` exact lane products per group, rounds one multiply by the
    scale, then adds ``n_groups`` group results. Sequential summation of n terms is off by
    at most about ``n * u * sum|terms|``, so the row error is below
    ``(group_size + n_groups + 2) * u * sum_j |w_j||x_j|``; the factor 2 absorbs the
    second-order terms. This is a worst-case bound, typically ~1e-6 of ``sum|w||x|``.
    """
    return 2 * (t.group_size + t.n_groups + 2) * U32 * (abs_x @ abs_w.T) + 1e-30


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_fp_matvec_matches_reference_on_dequantized_weights(rows, cols):
    rng, _, t = _case(10, rows, cols)
    w = dequantize(t).astype(np.float64)
    x = rng.normal(size=cols)
    x16 = x.astype(np.float16).astype(np.float64)
    y = matvec_int4(t, x)
    assert y.dtype == np.float32 and y.shape == (rows,)
    # Same operands as the kernel (decoded weights, FP16 activations): only FP32
    # accumulation separates the two.
    tol = _fp32_tol(t, np.abs(w), np.abs(x16))
    assert np.all(np.abs(y - reference.matvec(w, x16)) <= tol)
    # Against unrounded activations the FP16 cast adds at most u16 * |x_j| per lane.
    assert np.all(np.abs(y - reference.matvec(w, x)) <= tol + U16 * (np.abs(w) @ np.abs(x)))
    # The kernel really does consume FP16 activations.
    assert np.array_equal(y, matvec_int4(t, x.astype(np.float16)))


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_fp_matvec_within_quantization_bound_of_float_weights(rows, cols):
    rng, w, t = _case(11, rows, cols)
    x16 = rng.normal(size=cols).astype(np.float16).astype(np.float64)
    half_step = 0.5 * np.repeat(t.scales.astype(np.float64), t.group_size, axis=1)[:, :cols]
    # |w - decode(w)| <= scale/2 per weight, so the output moves by at most
    # sum_j (scale_j / 2) |x_j| on top of the accumulation bound.
    bound = half_step @ np.abs(x16) + _fp32_tol(t, np.abs(w).astype(np.float64), np.abs(x16))
    assert np.all(np.abs(matvec_int4(t, x16) - reference.matvec(w, x16)) <= bound * (1 + 1e-9))


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_fp_matmul_matches_reference_and_matvec(rows, cols):
    rng, _, t = _case(12, rows, cols)
    w = dequantize(t).astype(np.float64)
    x = rng.normal(size=(5, cols))
    x16 = x.astype(np.float16).astype(np.float64)
    y = matmul_int4(t, x)
    assert y.dtype == np.float32 and y.shape == (5, rows)
    assert np.all(np.abs(y - reference.matmul(w, x16)) <= _fp32_tol(t, np.abs(w), np.abs(x16)))
    assert np.array_equal(y, np.stack([matvec_int4(t, row) for row in x]))


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_integer_group_sums_are_exact(rows, cols):
    rng, _, t = _case(13, rows, cols)
    x_q = rng.integers(-128, 128, size=cols, dtype=np.int8)
    got = group_sums_int32(t, x_q)
    assert got.dtype == np.int32 and got.shape == (rows, t.n_groups)
    codes = unpack_nibbles(t.packed).tolist()
    acts = x_q.tolist() + [0] * (t.padded_cols - cols)
    expected = [[sum(codes[r][g * t.group_size + j] * acts[g * t.group_size + j]
                     for j in range(t.group_size)) for g in range(t.n_groups)]
                for r in range(rows)]
    assert got.tolist() == expected


def test_integer_group_sums_at_extreme_magnitude():
    codes = np.full((2, 64), -8, dtype=np.int8)
    codes[1] = 7
    t = Int4Tensor(pack_nibbles(codes), np.ones((2, 2), np.float16), rows=2, cols=64)
    x_q = np.full(64, -128, dtype=np.int8)
    assert group_sums_int32(t, x_q).tolist() == [[32768, 32768], [-28672, -28672]]
    assert matvec_int4_int8(t, x_q, 0.5).tolist() == [32768.0, -28672.0]


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_integer_path_matches_reference(rows, cols):
    rng, _, t = _case(14, rows, cols)
    w = dequantize(t).astype(np.float64)
    x_q, x_scale = quantize_activations_int8(rng.normal(size=cols))
    y = matvec_int4_int8(t, x_q, x_scale)
    assert y.dtype == np.float32 and y.shape == (rows,)
    x = x_q.astype(np.float64) * np.float64(x_scale)
    # Group sums are exact integers; rounding enters at one scale multiply per group, the
    # n_groups FP32 additions and the final multiply by x_scale.
    tol = 2 * (t.n_groups + 3) * U32 * (np.abs(w) @ np.abs(x)) + 1e-30
    assert np.all(np.abs(y - reference.matvec(w, x)) <= tol)
    exact = (group_sums_int32(t, x_q) * t.scales.astype(np.float64)).sum(axis=1)
    assert np.all(np.abs(y - exact * np.float64(x_scale)) <= tol)


def test_results_are_bit_identical_across_tile_sizes():
    rng, _, t = _case(15, 77, 160)
    x = rng.normal(size=160)
    xb = rng.normal(size=(3, 160))
    x_q, x_scale = quantize_activations_int8(x)
    base = (matvec_int4(t, x), matmul_int4(t, xb), matvec_int4_int8(t, x_q, x_scale))
    for tile_rows in (1, 3, 7, 32, 77, 82):
        assert np.array_equal(matvec_int4(t, x, tile_rows=tile_rows), base[0])
        assert np.array_equal(matmul_int4(t, xb, tile_rows=tile_rows), base[1])
        assert np.array_equal(matvec_int4_int8(t, x_q, x_scale, tile_rows=tile_rows), base[2])


def test_row_shards_concatenate_to_the_whole_bit_exact():
    rng, _, t = _case(16, 90, 96)
    x = rng.normal(size=96)
    xb = rng.normal(size=(4, 96))
    x_q, x_scale = quantize_activations_int8(x)
    bounds = [0, 5, 37, 38, 90]
    shards = [t.row_slice(a, b) for a, b in pairwise(bounds)]
    assert np.array_equal(np.concatenate([matvec_int4(s, x) for s in shards]), matvec_int4(t, x))
    assert np.array_equal(np.concatenate([matmul_int4(s, xb) for s in shards], axis=1),
                          matmul_int4(t, xb))
    assert np.array_equal(np.concatenate([matvec_int4_int8(s, x_q, x_scale) for s in shards]),
                          matvec_int4_int8(t, x_q, x_scale))


def test_every_code_and_padded_columns():
    rng = np.random.default_rng(17)
    cols = 40  # padded to 64
    codes = np.zeros((16, 64), dtype=np.int8)
    codes[:, :cols] = rng.integers(-8, 8, size=(16, cols), dtype=np.int8)
    codes[np.arange(16), np.arange(16)] = np.arange(-8, 8, dtype=np.int8)
    scales = rng.uniform(0.01, 2.0, size=(16, 2)).astype(np.float16)
    t = Int4Tensor(pack_nibbles(codes), scales, rows=16, cols=cols)
    w = dequantize(t).astype(np.float64)
    assert np.array_equal(w, (codes.reshape(16, 2, 32) * scales.astype(np.float64)[:, :, None])
                          .reshape(16, 64)[:, :cols])
    x16 = rng.normal(size=cols).astype(np.float16).astype(np.float64)
    assert np.all(np.abs(matvec_int4(t, x16) - reference.matvec(w, x16))
                  <= _fp32_tol(t, np.abs(w), np.abs(x16)))


def test_kernel_stats_accounting():
    rng, _, t = _case(18, 70, 50)
    x = rng.normal(size=50)
    stats = KernelStats()
    matvec_int4(t, x, tile_rows=16, stats=stats)
    assert stats.weight_bytes_read == t.nbytes
    assert stats.macs == 70 * 50
    assert stats.tiles == math.ceil(70 / 16)
    # Only one decoded tile exists at a time: far less than the expanded matrix.
    assert stats.peak_expanded_bytes == 16 * t.padded_cols < 70 * t.padded_cols
    matmul_int4(t, rng.normal(size=(3, 50)), tile_rows=100, stats=stats)
    x_q, x_scale = quantize_activations_int8(x)
    matvec_int4_int8(t, x_q, x_scale, tile_rows=35, stats=stats)
    assert stats.weight_bytes_read == 3 * t.nbytes
    assert stats.macs == 70 * 50 * (1 + 3 + 1)
    assert stats.tiles == 5 + 1 + 2
    assert stats.peak_expanded_bytes == 70 * t.padded_cols


def test_empty_tensor_and_argument_validation():
    empty = quantize_int4(np.zeros((0, 64)))
    stats = KernelStats()
    assert matvec_int4(empty, np.ones(64), stats=stats).shape == (0,)
    assert matmul_int4(empty, np.ones((2, 64))).shape == (2, 0)
    assert group_sums_int32(empty, np.ones(64, np.int8)).shape == (0, 2)
    assert (stats.tiles, stats.macs, stats.weight_bytes_read) == (0, 0, 0)
    _, _, t = _case(19, 4, 64)
    assert matmul_int4(t, np.zeros((0, 64))).shape == (0, 4)
    with pytest.raises(ValueError):
        matvec_int4(t, np.ones(64), tile_rows=0)
    with pytest.raises(ValueError):
        matvec_int4(t, np.ones(63))
    with pytest.raises(ValueError):
        matmul_int4(t, np.ones(64))
    with pytest.raises(ValueError):
        matvec_int4_int8(t, np.ones(64, np.int16), 1.0)
    with pytest.raises(ValueError):
        group_sums_int32(t, np.ones(32, np.int8))
