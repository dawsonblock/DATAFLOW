"""Dequantisation of GGUF tensor data (ggml block formats) to float32, in NumPy.

Implemented: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K -- every format a
Qwen3 ``Q4_K_M``, ``Q5_K_M``, ``Q6_K`` or ``Q8_0`` file contains.  Other formats raise
``UnsupportedFormat``: a tensor is never silently read as something it is not.

The block layouts follow ggml (``ggml-quants.c``, ``dequantize_row_*``); every function is
cross-checked against the reference ``gguf`` package in the tests.  ``quantize_q8_0`` is
provided so tests can write small quantised models.
"""

from __future__ import annotations

import numpy as np

from .gguf import GGML_TYPES


class UnsupportedFormat(NotImplementedError):
    """A ggml type this runtime cannot dequantise."""


QK_K = 256


def _f16(raw: np.ndarray) -> np.ndarray:
    """Little-endian float16 from a (..., 2) uint8 array, as float32."""
    return np.ascontiguousarray(raw).view("<f2")[..., 0].astype(np.float32)


def _blocks(data: np.ndarray, block_bytes: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.uint8).reshape(-1)
    if data.size % block_bytes:
        raise ValueError(f"{data.size} bytes are not a whole number of {block_bytes}-byte blocks")
    return data.reshape(-1, block_bytes)


def _q8_0(b):
    d = _f16(b[:, 0:2].reshape(-1, 1, 2))
    return d * b[:, 2:34].view(np.int8).astype(np.float32)


def _q4_0(b):
    d = _f16(b[:, 0:2].reshape(-1, 1, 2))
    qs = b[:, 2:18]
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32) - 8.0
    return d * q


def _q4_1(b):
    d, m = _f16(b[:, 0:2].reshape(-1, 1, 2)), _f16(b[:, 2:4].reshape(-1, 1, 2))
    qs = b[:, 4:20]
    return d * np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.float32) + m


def _q5(b, *, with_min: bool):
    d = _f16(b[:, 0:2].reshape(-1, 1, 2))
    start = 4 if with_min else 2
    m = _f16(b[:, 2:4].reshape(-1, 1, 2)) if with_min else None
    qh = np.ascontiguousarray(b[:, start:start + 4]).view("<u4")[:, 0:1]  # (n, 1)
    qs = b[:, start + 4:start + 20]
    bits = (qh >> np.arange(32, dtype=np.uint32)) & 1  # (n, 32)
    q = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.uint32) | (bits << 4)
    if with_min:
        return d * q.astype(np.float32) + m
    return d * (q.astype(np.float32) - 16.0)


def _scale_min_k4(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The eight 6-bit (scale, min) pairs packed in a K-quant block's 12 scale bytes."""
    s = scales.astype(np.uint8)
    sc = np.empty((s.shape[0], 8), np.uint8)
    mn = np.empty((s.shape[0], 8), np.uint8)
    sc[:, :4] = s[:, 0:4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc.astype(np.float32), mn.astype(np.float32)


def _q4_k(b):
    n = b.shape[0]
    d, dmin = _f16(b[:, 0:2].reshape(-1, 1, 2)), _f16(b[:, 2:4].reshape(-1, 1, 2))
    sc, mn = _scale_min_k4(b[:, 4:16])
    qs = b[:, 16:144].reshape(n, 4, 32)  # four 64-weight groups, 32 bytes each
    q = np.stack([qs & 0x0F, qs >> 4], axis=2).reshape(n, 8, 32).astype(np.float32)  # sub-blocks of 32
    return (d[:, :, None] * sc[:, :, None] * q - dmin[:, :, None] * mn[:, :, None]).reshape(n, QK_K)


def _q5_k(b):
    n = b.shape[0]
    d, dmin = _f16(b[:, 0:2].reshape(-1, 1, 2)), _f16(b[:, 2:4].reshape(-1, 1, 2))
    sc, mn = _scale_min_k4(b[:, 4:16])
    qh = b[:, 16:48]  # (n, 32)
    qs = b[:, 48:176].reshape(n, 4, 32)
    low = np.stack([qs & 0x0F, qs >> 4], axis=2).reshape(n, 8, 32).astype(np.uint8)
    high = ((qh[:, None, :] >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1) << 4  # bit j of sub-block j
    q = (low | high).astype(np.float32)
    return (d[:, :, None] * sc[:, :, None] * q - dmin[:, :, None] * mn[:, :, None]).reshape(n, QK_K)


def _q6_k(b):
    n = b.shape[0]
    ql = b[:, 0:128].reshape(n, 2, 64)
    qh = b[:, 128:192].reshape(n, 2, 32)
    scales = b[:, 192:208].view(np.int8).astype(np.float32).reshape(n, 2, 8)
    d = _f16(b[:, 208:210].reshape(-1, 1, 2))[:, :, None]  # (n, 1, 1)
    l1, l2 = ql[:, :, :32], ql[:, :, 32:]
    q1 = (l1 & 0x0F) | (((qh >> 0) & 3) << 4)
    q2 = (l2 & 0x0F) | (((qh >> 2) & 3) << 4)
    q3 = (l1 >> 4) | (((qh >> 4) & 3) << 4)
    q4 = (l2 >> 4) | (((qh >> 6) & 3) << 4)
    q = np.stack([q1, q2, q3, q4], axis=2).astype(np.float32) - 32.0  # (n, 2, 4, 32)
    # scale index: is = l // 16 (+0, +2, +4, +6 for the four quarters) within each 128-half
    sc = scales.reshape(n, 2, 4, 2)  # [half, quarter, l // 16]
    sc = np.repeat(sc, 16, axis=3)  # (n, 2, 4, 32)
    return (d[:, :, :, None] * sc * q).reshape(n, QK_K)


_DEQUANT = {
    "Q8_0": _q8_0, "Q4_0": _q4_0, "Q4_1": _q4_1,
    "Q5_0": lambda b: _q5(b, with_min=False), "Q5_1": lambda b: _q5(b, with_min=True),
    "Q4_K": _q4_k, "Q5_K": _q5_k, "Q6_K": _q6_k,
}  # fmt: skip

SUPPORTED = ("F32", "F16", "BF16", *_DEQUANT)

#: Scratch memory ``dequantize`` needs while it runs, as a multiple of the float32 result
#: (temporaries beyond the result itself).  Upper bounds on measured peaks (tracemalloc; the
#: test suite re-measures every format).  A loader reserves this much before it starts.
TRANSIENT_FACTOR = {"F32": 0.1, "F16": 0.1, "BF16": 1.25, "Q8_0": 1.25, "Q4_0": 1.25, "Q4_1": 1.25,
                    "Q5_0": 3.25, "Q5_1": 3.25, "Q4_K": 2.25, "Q5_K": 2.75, "Q6_K": 2.5}  # fmt: skip


def transient_bytes(ggml_type: int, n_elements: int) -> int:
    """Scratch bytes to reserve for dequantising ``n_elements`` weights of ``ggml_type``."""
    name = type_name(ggml_type)
    if name not in TRANSIENT_FACTOR:
        raise UnsupportedFormat(f"ggml type {name} is not supported by the NumPy runtime")
    return int(TRANSIENT_FACTOR[name] * n_elements * 4) + 4096


def type_name(ggml_type: int) -> str:
    return GGML_TYPES.get(ggml_type, (f"type{ggml_type}", 0, 0))[0]


def dequantize(data: np.ndarray, ggml_type: int, n_elements: int) -> np.ndarray:
    """Float32 values of ``n_elements`` weights stored as ``ggml_type`` in ``data`` (bytes)."""
    name = type_name(ggml_type)
    data = np.asarray(data, dtype=np.uint8).reshape(-1)
    if name == "F32":
        out = data.view("<f4").astype(np.float32, copy=True)
    elif name == "F16":
        out = data.view("<f2").astype(np.float32)
    elif name == "BF16":
        out = (data.view("<u2").astype(np.uint32) << 16).view(np.float32)
    elif name in _DEQUANT:
        size = GGML_TYPES[ggml_type][2]
        out = _DEQUANT[name](_blocks(data, size)).reshape(-1)
    else:
        raise UnsupportedFormat(f"ggml type {name} is not supported by the NumPy runtime "
                                f"(supported: {', '.join(SUPPORTED)})")  # fmt: skip
    if out.size != n_elements:
        raise ValueError(f"{name}: {out.size} values decoded, {n_elements} expected")
    return out


def quantize_q8_0(values: np.ndarray) -> np.ndarray:
    """Q8_0 blocks (bytes) for float values (length a multiple of 32), as ggml rounds them."""
    x = np.asarray(values, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1, keepdims=True)
    d = amax / 127.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    scaled = x * inv  # ggml rounds half away from zero (roundf), not to even
    q = (np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)).clip(-128, 127).astype(np.int8)
    out = np.empty((x.shape[0], 34), np.uint8)
    out[:, 0:2] = d.astype("<f2").view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(-1)
