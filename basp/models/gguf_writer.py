"""Write GGUF files with tensor data (F32, F16, Q8_0): small test models, and exports.

Tensors are given in NumPy order (a weight matrix as ``(out, in)``); the GGUF shape is the
reverse.  Files written here are read back by ``basp.models.gguf`` and by the reference
``gguf`` package (tested).
"""

from __future__ import annotations

import os
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .gguf import _ARRAY, _STRING, MAGIC
from .quants import quantize_q8_0

TYPE_IDS = {"F32": 0, "F16": 1, "Q8_0": 8}


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _value(v: Any) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I?", 7, v)
    if isinstance(v, int):
        return struct.pack("<II", 4, v) if 0 <= v < 2**32 else struct.pack("<Iq", 11, v)
    if isinstance(v, float):
        return struct.pack("<If", 6, v)
    if isinstance(v, str):
        return struct.pack("<I", _STRING) + _string(v)
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, str) for x in v):
            return struct.pack("<IIQ", _ARRAY, _STRING, len(v)) + b"".join(_string(x) for x in v)
        return struct.pack("<IIQ", _ARRAY, 5, len(v)) + b"".join(struct.pack("<i", x) for x in v)
    raise TypeError(f"unsupported metadata value {v!r}")


def encode(array: np.ndarray, type_name: str) -> bytes:
    a = np.ascontiguousarray(array, dtype=np.float32)
    if type_name == "F32":
        return a.astype("<f4").tobytes()
    if type_name == "F16":
        return a.astype("<f2").tobytes()
    if type_name == "Q8_0":
        if a.shape[-1] % 32:
            raise ValueError(f"Q8_0 needs rows that are a multiple of 32 values, not {a.shape[-1]}")
        return quantize_q8_0(a.reshape(-1)).tobytes()
    raise ValueError(f"cannot write tensor type {type_name}")


def write_gguf(path: str | os.PathLike, metadata: Mapping[str, Any],
               tensors: Sequence[tuple[str, np.ndarray, str]], *, alignment: int = 32) -> None:  # fmt: skip
    """``tensors``: (name, NumPy array, type name) in file order."""
    head = MAGIC + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    head += b"".join(_string(k) + _value(v) for k, v in metadata.items())
    blobs, offset, infos = [], 0, []
    for name, array, type_name in tensors:
        data = encode(array, type_name)
        shape = tuple(reversed(np.shape(array)))
        infos.append(_string(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
                     + struct.pack("<IQ", TYPE_IDS[type_name], offset))  # fmt: skip
        padded = data + b"\0" * (-len(data) % alignment)
        blobs.append(padded)
        offset += len(padded)
    head += b"".join(infos)
    with open(path, "wb") as f:
        f.write(head + b"\0" * (-len(head) % alignment))
        f.writelines(blobs)


def tiny_qwen3(path: str | os.PathLike, *, moe: bool = False, seed: int = 0, weight_type: str = "F32",
               layers: int = 2, hidden: int = 64, heads: int = 4, kv_heads: int = 2, head_dim: int = 32,
               ffn: int = 128, vocab: int = 96, experts: int = 8, top_k: int = 2, expert_ffn: int = 32,
               tied: bool = True, scale: float = 0.08) -> None:  # fmt: skip
    """A small random-weight Qwen3 (or Qwen3-MoE) GGUF with the real tensor layout."""
    rng = np.random.default_rng(seed)
    arch = "qwen3moe" if moe else "qwen3"

    def w(*shape):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    meta: dict[str, Any] = {"general.architecture": arch, "general.name": f"tiny-random-{arch}",
                            f"{arch}.block_count": layers, f"{arch}.context_length": 256,
                            f"{arch}.embedding_length": hidden, f"{arch}.attention.head_count": heads,
                            f"{arch}.attention.head_count_kv": kv_heads, f"{arch}.attention.key_length": head_dim,
                            f"{arch}.attention.value_length": head_dim, f"{arch}.rope.freq_base": 10000.0,
                            f"{arch}.attention.layer_norm_rms_epsilon": 1e-6}  # fmt: skip
    if moe:
        meta.update({f"{arch}.expert_count": experts, f"{arch}.expert_used_count": top_k,
                     f"{arch}.expert_feed_forward_length": expert_ffn})  # fmt: skip
    else:
        meta[f"{arch}.feed_forward_length"] = ffn
    mat = weight_type  # matrices quantised as asked; norms stay F32 (as in real files)
    tensors = [("token_embd.weight", w(vocab, hidden), mat), ("output_norm.weight", norm(hidden), "F32")]
    if not tied:
        tensors.append(("output.weight", w(vocab, hidden), mat))
    for i in range(layers):
        p = f"blk.{i}."
        tensors += [(p + "attn_norm.weight", norm(hidden), "F32"), (p + "ffn_norm.weight", norm(hidden), "F32"),
                    (p + "attn_q.weight", w(heads * head_dim, hidden), mat),
                    (p + "attn_k.weight", w(kv_heads * head_dim, hidden), mat),
                    (p + "attn_v.weight", w(kv_heads * head_dim, hidden), mat),
                    (p + "attn_output.weight", w(hidden, heads * head_dim), mat),
                    (p + "attn_q_norm.weight", norm(head_dim), "F32"), (p + "attn_k_norm.weight", norm(head_dim), "F32")]  # fmt: skip
        if moe:
            tensors += [(p + "ffn_gate_inp.weight", w(experts, hidden) * 8, "F32"),
                        (p + "ffn_gate_exps.weight", w(experts, expert_ffn, hidden), mat),
                        (p + "ffn_up_exps.weight", w(experts, expert_ffn, hidden), mat),
                        (p + "ffn_down_exps.weight", w(experts, hidden, expert_ffn), mat)]  # fmt: skip
        else:
            tensors += [(p + "ffn_gate.weight", w(ffn, hidden), mat), (p + "ffn_up.weight", w(ffn, hidden), mat),
                        (p + "ffn_down.weight", w(hidden, ffn), mat)]  # fmt: skip
    write_gguf(path, meta, tensors)
