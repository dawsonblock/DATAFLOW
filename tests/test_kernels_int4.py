"""INT4 weight format: packing, quantization bounds, addressing and serialization."""

import warnings
from itertools import pairwise

import numpy as np
import pytest

from basp.kernels.int4 import (
    GROUP_SIZE,
    Int4Tensor,
    dequantize,
    pack_nibbles,
    quantize_activations_int8,
    quantize_int4,
    symbol_entropy_bits,
    unpack_nibbles,
)


def _weights(rng, rows, cols, dtype=np.float32):
    """Gaussian weights with per-row magnitudes spread over five decades."""
    return (rng.normal(size=(rows, cols)) * 10.0 ** rng.uniform(-3, 2, size=(rows, 1))).astype(
        dtype)


def _group_scales(t: Int4Tensor) -> np.ndarray:
    """FP16 scale of each weight's group, float64 [rows, cols]."""
    return np.repeat(t.scales.astype(np.float64), t.group_size, axis=1)[:, : t.cols]


def test_nibble_round_trip_covers_all_code_pairs():
    codes = np.arange(-8, 8, dtype=np.int8)
    pairs = np.array([(lo, hi) for lo in codes for hi in codes], dtype=np.int8)
    packed = pack_nibbles(pairs)
    assert packed.dtype == np.uint8 and packed.shape == (256, 1)
    expected = [(int(lo) & 0xF) | ((int(hi) & 0xF) << 4) for lo, hi in pairs]
    assert packed[:, 0].tolist() == expected
    assert np.array_equal(unpack_nibbles(packed), pairs)
    # Low nibble holds the even column: (-8, 7) -> 0x8 | 0x7 << 4.
    assert pack_nibbles(np.array([-8, 7], dtype=np.int8)).tolist() == [0x78]


def test_pack_rejects_invalid_codes():
    with pytest.raises(ValueError):
        pack_nibbles(np.zeros(3, dtype=np.int8))
    with pytest.raises(ValueError):
        pack_nibbles(np.array([8, 0], dtype=np.int8))
    with pytest.raises(ValueError):
        pack_nibbles(np.array([0, -9], dtype=np.int8))
    with pytest.raises(TypeError):
        pack_nibbles(np.zeros(2, dtype=np.int32))
    with pytest.raises(TypeError):
        unpack_nibbles(np.zeros(2, dtype=np.int8))


def test_bits_per_weight_and_expert_size_arithmetic():
    # Small analogue of the spec's example expert (hidden 4096, intermediate 14336 = 3.5x).
    hidden, inter = 64, 224
    rng = np.random.default_rng(0)
    mats = [quantize_int4(_weights(rng, r, c)) for r, c in
            [(inter, hidden), (inter, hidden), (hidden, inter)]]
    assert all(m.bits_per_weight == 4.5 for m in mats)
    n_weights = sum(m.n_weights for m in mats)
    assert n_weights == 3 * hidden * inter
    assert sum(m.nbytes for m in mats) == n_weights * 4.5 / 8 == 24192
    # The same arithmetic at full size reproduces the spec: 176.2M weights, 94.5 MiB.
    full = 3 * 4096 * 14336
    assert full == 176_160_768 and full * 4.5 / 8 == 99_090_432 == 94.5 * 2**20


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("magnitude", [1.0, 1e3, 1e-4, 1e-7])
def test_quantization_error_within_half_step(dtype, magnitude):
    rng = np.random.default_rng(1)
    w = (_weights(rng, 37, 96, np.float64) * magnitude).astype(dtype)
    t = quantize_int4(w)
    err = np.abs(w.astype(np.float64) - dequantize(t).astype(np.float64))
    # Round-to-nearest gives |w - code*scale| <= scale/2 exactly; 1e-9 relative slack covers
    # the float64 division used to pick the code. magnitude=1e-7 exercises subnormal scales.
    assert np.all(err <= 0.5 * _group_scales(t) * (1 + 1e-9))
    codes = unpack_nibbles(t.packed)
    assert codes.min() >= -7 and codes.max() <= 7  # symmetric scale never needs -8


def test_quantization_is_idempotent_on_decoded_weights():
    # The stored FP16 scale is the one the codes were chosen with, so quantizing the decoded
    # matrix must reproduce codes and scales bit for bit.
    rng = np.random.default_rng(2)
    t = quantize_int4(_weights(rng, 20, 128))
    again = quantize_int4(dequantize(t))
    assert np.array_equal(again.packed, t.packed)
    assert np.array_equal(again.scales, t.scales)


def test_columns_not_multiple_of_group_size_are_zero_padded():
    rng = np.random.default_rng(3)
    w = _weights(rng, 9, 50)
    t = quantize_int4(w)
    assert (t.rows, t.cols, t.padded_cols, t.n_groups) == (9, 50, 64, 2)
    assert t.packed.shape == (9, 32) and t.scales.shape == (9, 2)
    assert np.all(unpack_nibbles(t.packed)[:, 50:] == 0)
    assert dequantize(t).shape == (9, 50)
    assert t.bits_per_weight == pytest.approx(8 * (32 + 4) / 50)
    assert np.all(np.abs(w - dequantize(t)) <= 0.5 * _group_scales(t) * (1 + 1e-9))


def test_zero_groups_rows_and_cols_quantize_without_warnings():
    rng = np.random.default_rng(4)
    w = _weights(rng, 4, 96)
    w[1] = 0.0  # a whole row of zero groups
    w[2, 32:64] = 0.0  # one zero group between live ones
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = quantize_int4(w)
        deq = dequantize(t)
        empty_rows = quantize_int4(np.zeros((0, 64)))
        empty_cols = quantize_int4(np.zeros((3, 0)))
    assert np.all(t.scales[1] == 0) and t.scales[2, 1] == 0
    assert np.all(deq[1] == 0) and np.all(deq[2, 32:64] == 0)
    assert np.all(np.isfinite(deq))
    assert (empty_rows.rows, empty_rows.nbytes, empty_rows.bits_per_weight) == (0, 0, 0.0)
    assert dequantize(empty_rows).shape == (0, 64)
    assert empty_cols.nbytes == 0 and dequantize(empty_cols).shape == (3, 0)


def test_quantize_rejects_bad_input():
    with pytest.raises(ValueError):
        quantize_int4(np.zeros(8))
    with pytest.raises(ValueError):
        quantize_int4(np.zeros((2, 8)), group_size=3)
    with pytest.raises(ValueError):
        quantize_int4(np.array([[np.nan, 1.0]]))
    with pytest.raises(ValueError):
        quantize_int4(np.full((1, 32), 1e6))  # scale would overflow FP16
    with pytest.raises(ValueError):
        Int4Tensor(np.zeros((2, 16), np.uint8), np.zeros((2, 2), np.float16), rows=2, cols=32)


def test_row_slice_is_an_independently_addressable_view():
    rng = np.random.default_rng(5)
    t = quantize_int4(_weights(rng, 40, 64))
    bounds = [0, 7, 7, 25, 40]
    shards = [t.row_slice(a, b) for a, b in pairwise(bounds)]
    assert sum(s.nbytes for s in shards) == t.nbytes
    assert sum(s.n_weights for s in shards) == t.n_weights
    full = dequantize(t)
    for (a, b), shard in zip(pairwise(bounds), shards):
        assert shard.rows == b - a and shard.cols == t.cols
        assert np.array_equal(dequantize(shard), full[a:b])
        if shard.rows:
            assert np.shares_memory(shard.packed, t.packed)
            assert np.shares_memory(shard.scales, t.scales)
    with pytest.raises(IndexError):
        t.row_slice(10, 41)
    with pytest.raises(IndexError):
        t.row_slice(5, 4)


def test_bytes_round_trip():
    rng = np.random.default_rng(6)
    for t in (quantize_int4(_weights(rng, 11, 50)),
              quantize_int4(_weights(rng, 16, 64)).row_slice(3, 9),
              quantize_int4(_weights(rng, 5, 24), group_size=8)):
        buf = t.to_bytes()
        assert len(buf) == 24 + t.nbytes
        back = Int4Tensor.from_bytes(buf)
        assert (back.rows, back.cols, back.group_size) == (t.rows, t.cols, t.group_size)
        assert np.array_equal(back.packed, t.packed)
        assert np.array_equal(back.scales, t.scales)
        assert np.array_equal(dequantize(back), dequantize(t))
    with pytest.raises(ValueError):
        Int4Tensor.from_bytes(b"XXXX" + buf[4:])
    with pytest.raises(ValueError):
        Int4Tensor.from_bytes(buf[:-1])
    with pytest.raises(ValueError):
        Int4Tensor.from_bytes(buf[:10])


def test_quantize_activations_int8():
    rng = np.random.default_rng(7)
    x = rng.normal(size=300).astype(np.float32) * 3
    q, scale = quantize_activations_int8(x)
    assert q.dtype == np.int8 and isinstance(scale, np.float32)
    assert np.abs(q).max() == 127
    err = np.abs(x.astype(np.float64) - q.astype(np.float64) * np.float64(scale))
    assert np.all(err <= 0.5 * np.float64(scale) * (1 + 1e-6))
    q0, s0 = quantize_activations_int8(np.zeros(5))
    assert np.all(q0 == 0) and s0 == 0
    with pytest.raises(ValueError):
        quantize_activations_int8(np.array([1.0, np.inf]))


def test_symbol_entropy_bits():
    assert symbol_entropy_bits(quantize_int4(np.zeros((4, 64)))) == 0.0
    uniform = np.tile(np.arange(-8, 8, dtype=np.int8), (3, 4))
    t = Int4Tensor(pack_nibbles(uniform), np.ones((3, 2), np.float16), rows=3, cols=64)
    assert symbol_entropy_bits(t) == pytest.approx(4.0)
    rng = np.random.default_rng(8)
    gaussian = quantize_int4(rng.normal(size=(256, 1024)))
    # Max-abs scaling over groups of 32 Gaussian weights uses the 15 symmetric codes
    # unevenly, so the order-0 entropy sits below 4 bits but not by much.
    assert 3.5 < symbol_entropy_bits(gaussian) < 4.0
    assert GROUP_SIZE == 32
