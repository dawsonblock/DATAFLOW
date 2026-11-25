"""Read the header of a GGUF model file: metadata, tensor index, exact size per weight.

Only the header and tensor index are read (the first few megabytes); tensor data is
never loaded.  That is enough to

* compute the file's exact **bits per weight** (bytes of tensor data / parameters), which
  the simulator needs to model the same weight traffic as the measured run,
* check that a file **is the model it claims to be**, by deriving a model shape from its
  metadata and comparing it with the shape the simulator uses, and
* **fingerprint** the file cheaply (SHA-256 of the header and tensor index), so a
  measurement records exactly which file produced it.

The format follows the GGUF specification (versions 2 and 3) as implemented by the
``gguf`` Python package shipped with llama.cpp; the block sizes below were copied from that
package's ``GGML_QUANT_SIZES`` and are cross-checked against it in the tests when it is
installed.  GGUF quantisation formats (Q4_K and friends) are **not** the BASP INT4 format:
only their sizes are used here, never their arithmetic.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from basp.workloads.model import MoEModelSpec

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

#: ggml tensor type id -> (name, elements per block, bytes per block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18),
}  # fmt: skip

#: llama.cpp ``general.file_type`` values (llama_ftype) for the common quantisations
FILE_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K",
}  # fmt: skip

# value types: id -> struct format (None: variable-length)
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
            12: "<d"}  # fmt: skip
_STRING, _ARRAY = 8, 9
#: arrays longer than this are summarised (length and type) instead of being kept in memory
KEEP_ARRAY = 4096
# sanity limits: a corrupt or hostile file must fail cleanly instead of exhausting memory
_MAX_COUNT = 10_000_000
_MAX_STRING = 64 * 1024 * 1024
_MAX_DIMS = 8


class GGUFError(ValueError):
    """The file is not a GGUF file this reader understands."""


@dataclass(frozen=True)
class ArraySummary:
    """A long metadata array (e.g. a tokenizer vocabulary), kept as its type and length only."""

    item_type: int
    length: int


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # GGUF order: the first dimension is the contiguous one
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.ggml_type, (f"type{self.ggml_type}", 0, 0))[0]

    @property
    def n_elements(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        if self.ggml_type not in GGML_TYPES:
            raise GGUFError(f"tensor {self.name}: unknown ggml type {self.ggml_type}")
        _, block, size = GGML_TYPES[self.ggml_type]
        if self.n_elements % block:
            raise GGUFError(f"tensor {self.name}: {self.n_elements} elements are not a whole number "
                            f"of {self.type_name} blocks of {block}")  # fmt: skip
        return self.n_elements // block * size


@dataclass
class GGUFFile:
    path: str
    version: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    data_offset: int  # start of the tensor data (aligned)
    file_size: int
    header_sha256: str  # of the bytes before the tensor data: metadata and tensor index
    _by_name: dict[str, TensorInfo] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_name = {t.name: t for t in self.tensors}

    # -- identity ----------------------------------------------------------------------
    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    @property
    def name(self) -> str:
        return str(self.metadata.get("general.name", Path(self.path).stem))

    @property
    def file_type(self) -> str:
        value = self.metadata.get("general.file_type")
        return FILE_TYPES.get(int(value), f"ftype{value}") if value is not None else "unknown"

    def fingerprint(self) -> dict[str, Any]:
        """What a measurement records about the file that produced it."""
        return {"file": Path(self.path).name, "size": self.file_size, "header_sha256": self.header_sha256,
                "architecture": self.architecture, "file_type": self.file_type,
                "parameters": self.parameter_count, "bits_per_weight": round(self.bits_per_weight, 4)}  # fmt: skip

    # -- sizes -------------------------------------------------------------------------
    def tensor(self, name: str) -> TensorInfo | None:
        return self._by_name.get(name)

    @property
    def parameter_count(self) -> int:
        return sum(t.n_elements for t in self.tensors)

    @property
    def data_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def bits_per_weight(self) -> float:
        """Tensor data bits per parameter, all tensors included (what llama.cpp reads)."""
        params = self.parameter_count
        return 8.0 * self.data_bytes / params if params else 0.0

    def type_breakdown(self) -> dict[str, dict[str, float]]:
        """Parameters and bytes per tensor type, largest first."""
        out: dict[str, dict[str, float]] = {}
        for t in self.tensors:
            row = out.setdefault(t.type_name, {"parameters": 0, "bytes": 0})
            row["parameters"] += t.n_elements
            row["bytes"] += t.nbytes
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["bytes"]))

    # -- metadata helpers --------------------------------------------------------------
    def arch_value(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(f"{self.architecture}.{key}", default)


# -- reading -------------------------------------------------------------------------------
def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    buf = f.read(size)
    if len(buf) != size:
        raise GGUFError("unexpected end of file in the GGUF header")
    return struct.unpack(fmt, buf)[0]


def _read_string(f: BinaryIO) -> str:
    length = _read(f, "<Q")
    if length > _MAX_STRING:
        raise GGUFError(f"string of {length} bytes in the header: file is corrupt")
    buf = f.read(length)
    if len(buf) != length:
        raise GGUFError("unexpected end of file in a header string")
    return buf.decode("utf-8", errors="replace")


def _skip_string(f: BinaryIO) -> None:
    length = _read(f, "<Q")
    if length > _MAX_STRING:
        raise GGUFError(f"string of {length} bytes in the header: file is corrupt")
    f.seek(length, os.SEEK_CUR)


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALARS:
        return _read(f, _SCALARS[vtype])
    if vtype == _STRING:
        return _read_string(f)
    if vtype == _ARRAY:
        item_type, length = _read(f, "<I"), _read(f, "<Q")
        if length > _MAX_COUNT:
            raise GGUFError(f"array of {length} items in the header: file is corrupt")
        if length > KEEP_ARRAY:
            if item_type in _SCALARS:
                f.seek(length * struct.calcsize(_SCALARS[item_type]), os.SEEK_CUR)
            elif item_type == _STRING:
                for _ in range(length):
                    _skip_string(f)
            else:
                for _ in range(length):
                    _read_value(f, item_type)
            return ArraySummary(item_type, length)
        return [_read_value(f, item_type) for _ in range(length)]
    raise GGUFError(f"unknown metadata value type {vtype}")


def read_gguf(path: str | os.PathLike) -> GGUFFile:
    """Read the header and tensor index of a GGUF file (no tensor data)."""
    path = os.fspath(path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise GGUFError(f"{path} is not a GGUF file (bad magic)")
        version = _read(f, "<I")
        if version not in (2, 3):
            raise GGUFError(f"{path}: GGUF version {version} is not supported (2 and 3 are)")
        n_tensors, n_kv = _read(f, "<Q"), _read(f, "<Q")
        if n_tensors > _MAX_COUNT or n_kv > _MAX_COUNT:
            raise GGUFError(f"{path}: implausible header counts ({n_tensors} tensors, {n_kv} keys)")
        metadata: dict[str, Any] = {}
        for _ in range(n_kv):
            key = _read_string(f)
            metadata[key] = _read_value(f, _read(f, "<I"))
        tensors = []
        for _ in range(n_tensors):
            name = _read_string(f)
            n_dims = _read(f, "<I")
            if n_dims > _MAX_DIMS:
                raise GGUFError(f"{path}: tensor {name} has {n_dims} dimensions")
            shape = tuple(_read(f, "<Q") for _ in range(n_dims))
            tensors.append(TensorInfo(name, shape, _read(f, "<I"), _read(f, "<Q")))
        alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        if alignment <= 0 or alignment & (alignment - 1):
            raise GGUFError(f"{path}: alignment {alignment} is not a power of two")
        header_end = f.tell()
        data_offset = -(-header_end // alignment) * alignment
        f.seek(0)
        digest = hashlib.sha256(f.read(header_end)).hexdigest()
    gguf = GGUFFile(path, version, metadata, tensors, data_offset, size, digest)
    data_end = data_offset + max((t.offset + t.nbytes for t in tensors), default=0)
    if data_end > size:
        raise GGUFError(f"{path}: tensor data extends to byte {data_end} but the file has {size} "
                        "(truncated download, or a split file -- give the first part)")  # fmt: skip
    return gguf


# -- model shape ------------------------------------------------------------------------------
def spec_from_gguf(gguf: GGUFFile, *, name: str | None = None) -> MoEModelSpec:
    """The simulator's model shape implied by the file's metadata.

    Understands decoder-only transformers described with the standard llama.cpp keys
    (``llama``, ``qwen2``, ``qwen3``, ``qwen2moe``, ``qwen3moe`` and similar).  Dense models
    become one always-selected expert per layer, as in ``basp.workloads.zoo``.
    """
    arch = gguf.architecture
    if not arch:
        raise GGUFError("the file has no general.architecture")

    def need(key: str) -> int:
        value = gguf.arch_value(key)
        if value is None:
            raise GGUFError(f"{arch} metadata lacks {arch}.{key}")
        return int(value)

    hidden, layers, heads = need("embedding_length"), need("block_count"), need("attention.head_count")
    kv_heads = int(gguf.arch_value("attention.head_count_kv", heads))
    head_dim = int(gguf.arch_value("attention.key_length", hidden // heads))
    experts = int(gguf.arch_value("expert_count", 0) or 0)
    if experts > 1:
        top_k = need("expert_used_count")
        inter = int(gguf.arch_value("expert_feed_forward_length", 0) or 0) or need("feed_forward_length")
    else:
        experts, top_k, inter = 1, 1, need("feed_forward_length")
    tokens = gguf.metadata.get("tokenizer.ggml.tokens")
    vocab = tokens.length if isinstance(tokens, ArraySummary) else len(tokens or [])
    embedding = gguf.tensor("token_embd.weight")
    if embedding is not None and len(embedding.shape) == 2:
        vocab = int(embedding.shape[1])  # the embedding matrix is authoritative (padding)
    if not vocab:
        vocab = int(gguf.arch_value("vocab_size", 0) or 0)
    return MoEModelSpec(
        name or gguf.name, hidden=hidden, n_layers=layers, n_experts=experts, top_k=top_k,
        expert_inter=inter, n_heads=heads, n_kv_heads=kv_heads, head_dim=head_dim, vocab=vocab,
        tied_embeddings=gguf.tensor("output.weight") is None,
        n_shared_experts=int(gguf.arch_value("expert_shared_count", 0) or 0),
        source=f"GGUF {Path(gguf.path).name}",
    )  # fmt: skip


_SHAPE_FIELDS = ("hidden", "n_layers", "n_experts", "top_k", "expert_inter", "n_heads", "n_kv_heads",
                 "head_dim", "vocab", "tied_embeddings", "n_shared_experts")  # fmt: skip


def compare_shapes(found: MoEModelSpec, expected: MoEModelSpec) -> list[str]:
    """Differences between two model shapes, as readable strings (empty: same shape)."""
    return [f"{f}: file {getattr(found, f)}, expected {getattr(expected, f)}"
            for f in _SHAPE_FIELDS if getattr(found, f) != getattr(expected, f)]  # fmt: skip


def check_against(gguf: GGUFFile, expected: MoEModelSpec, *, param_tolerance: float = 0.01) -> dict[str, Any]:
    """Shape and parameter-count agreement between a file and the shape the simulator uses."""
    found = spec_from_gguf(gguf)
    params = gguf.parameter_count
    rel = abs(params - expected.total_params) / expected.total_params
    problems = compare_shapes(found, expected)
    if rel > param_tolerance:
        problems.append(f"parameters: file {params:,}, simulator {expected.total_params:,} "
                        f"({rel:.1%} apart)")  # fmt: skip
    return {"ok": not problems, "problems": problems, "file_parameters": params,
            "simulator_parameters": expected.total_params, "relative_difference": rel}  # fmt: skip


def write_minimal_gguf(path: str | os.PathLike, metadata: Mapping[str, Any],
                       tensors: list[tuple[str, tuple[int, ...], int]], *, alignment: int = 32,
                       sparse: bool = False) -> None:
    """Write a structurally valid GGUF file with zero-filled tensor data (for tests and examples).

    ``sparse=True`` extends the file to its full size without writing the data, so a header with
    a real model's tensor layout costs no disk space on file systems with sparse files.

    ``metadata`` values may be ``int`` (stored as UINT32 when non-negative and small, else INT64),
    ``float``, ``bool``, ``str`` or lists of strings or ints.  ``tensors`` are
    ``(name, shape, ggml_type)`` triples.
    """

    def string(s: str) -> bytes:
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    def value(v: Any) -> bytes:
        if isinstance(v, bool):
            return struct.pack("<I?", 7, v)
        if isinstance(v, int):
            return struct.pack("<II", 4, v) if 0 <= v < 2**32 else struct.pack("<Iq", 11, v)
        if isinstance(v, float):
            return struct.pack("<If", 6, v)
        if isinstance(v, str):
            return struct.pack("<I", _STRING) + string(v)
        if isinstance(v, list):
            if all(isinstance(x, str) for x in v):
                return struct.pack("<IIQ", _ARRAY, _STRING, len(v)) + b"".join(string(x) for x in v)
            return struct.pack("<IIQ", _ARRAY, 5, len(v)) + b"".join(struct.pack("<i", x) for x in v)
        raise TypeError(f"unsupported metadata value {v!r}")

    head = MAGIC + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    head += b"".join(string(k) + value(v) for k, v in metadata.items())
    offset, infos, sizes = 0, [], []
    for name, shape, ggml_type in tensors:
        nbytes = TensorInfo(name, tuple(shape), ggml_type, 0).nbytes
        infos.append(string(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
                     + struct.pack("<IQ", ggml_type, offset))  # fmt: skip
        sizes.append(nbytes)
        offset += -(-nbytes // alignment) * alignment
    head += b"".join(infos)
    pad = -len(head) % alignment
    with open(path, "wb") as f:
        f.write(head + b"\0" * pad)
        if sparse:
            f.truncate(len(head) + pad + offset)
            return
        for nbytes in sizes:
            f.write(b"\0" * (-(-nbytes // alignment) * alignment))
