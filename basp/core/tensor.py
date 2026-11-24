"""Tensor descriptors, data types, stream formats and memory policies.

The :class:`TensorDescriptor` mirrors the wire/C layout from the architecture
specification so that the same record can later be shared with the C++ runtime
and with hardware command streams.
"""

from __future__ import annotations

import enum
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass, replace


class DType(enum.IntEnum):
    """Element data types."""

    FP32 = 1
    FP16 = 2
    INT32 = 3
    INT8 = 4
    INT4 = 5
    INT2 = 6
    TERNARY = 7
    BYTES = 8  # opaque byte stream (compressed blocks)


#: Storage bits per element (ternary packs five trits into one byte).
DTYPE_BITS: Mapping[DType, float] = {
    DType.FP32: 32.0,
    DType.FP16: 16.0,
    DType.INT32: 32.0,
    DType.INT8: 8.0,
    DType.INT4: 4.0,
    DType.INT2: 2.0,
    DType.TERNARY: 1.6,
    DType.BYTES: 8.0,
}


class CompressionFormat(enum.IntEnum):
    """Weight stream formats (section 12 of the architecture)."""

    RAW_FP16 = 1
    INT8 = 2
    INT4 = 3
    INT2 = 4
    TERNARY = 5
    SPARSE_BLOCK = 6
    ENTROPY = 7
    PREDICTIVE = 8


@dataclass(frozen=True)
class FormatSpec:
    """Static properties of a weight stream format.

    ``bits_per_weight`` is the *effective* storage cost including per-group
    metadata (one FP16 scale per 32 weights adds 0.5 bit).  ``decodes_to`` names
    the format that results from decoding; two formats with the same
    ``lossless_base`` produce bit-identical model outputs and may be exchanged
    by the runtime without changing model behaviour.  ``assumed`` marks ratios
    that are modelling assumptions rather than properties of the format.
    """

    fmt: CompressionFormat
    bits_per_weight: float
    decodes_to: CompressionFormat | None = None
    assumed: bool = False
    description: str = ""

    @property
    def lossless_base(self) -> CompressionFormat:
        return self.decodes_to if self.decodes_to is not None else self.fmt


_GROUP_SCALE_BITS = 16.0 / 32.0  # one FP16 scale per 32-weight group

DEFAULT_FORMATS: Mapping[CompressionFormat, FormatSpec] = {
    f.fmt: f
    for f in (
        FormatSpec(CompressionFormat.RAW_FP16, 16.0, description="reference execution"),
        FormatSpec(CompressionFormat.INT8, 8.0 + _GROUP_SCALE_BITS, description="higher precision"),
        FormatSpec(CompressionFormat.INT4, 4.0 + _GROUP_SCALE_BITS, description="primary format"),
        FormatSpec(CompressionFormat.INT2, 2.0 + _GROUP_SCALE_BITS, description="highly compressed"),
        FormatSpec(CompressionFormat.TERNARY, 1.6 + _GROUP_SCALE_BITS, description="5 trits/byte"),
        # The three ratios below depend on the weights being stored; the defaults
        # are placeholders to be replaced by measurements (see docs/limitations.md).
        FormatSpec(
            CompressionFormat.SPARSE_BLOCK,
            0.5 * 4.0 + _GROUP_SCALE_BITS + 1.0 / 32.0,
            decodes_to=CompressionFormat.INT4,
            assumed=True,
            description="INT4 blocks, 50% block density, 1-bit block mask",
        ),
        FormatSpec(
            CompressionFormat.ENTROPY,
            3.62 + _GROUP_SCALE_BITS,
            decodes_to=CompressionFormat.INT4,
            assumed=True,
            description="lossless entropy-coded INT4 (order-0 entropy of Gaussian weights)",
        ),
        FormatSpec(
            CompressionFormat.PREDICTIVE,
            3.3 + _GROUP_SCALE_BITS,
            decodes_to=CompressionFormat.INT4,
            assumed=True,
            description="experimental predictive block coding",
        ),
    )
}


class FormatTable:
    """Format specifications with optional per-experiment overrides."""

    def __init__(self, overrides: Mapping[CompressionFormat, float] | None = None):
        self._specs = dict(DEFAULT_FORMATS)
        for fmt, bits in (overrides or {}).items():
            fmt = CompressionFormat(fmt)
            if bits <= 0:
                raise ValueError(f"bits_per_weight must be positive, got {bits}")
            self._specs[fmt] = replace(self._specs[fmt], bits_per_weight=float(bits))

    def spec(self, fmt: CompressionFormat) -> FormatSpec:
        return self._specs[CompressionFormat(fmt)]

    def bits(self, fmt: CompressionFormat) -> float:
        return self.spec(fmt).bits_per_weight

    def weight_bytes(self, n_weights: int, fmt: CompressionFormat) -> int:
        """Stored size in bytes of ``n_weights`` weights in ``fmt``."""
        return math.ceil(n_weights * self.bits(fmt) / 8.0)

    def interchangeable(self, a: CompressionFormat, b: CompressionFormat) -> bool:
        """True if switching between ``a`` and ``b`` cannot change model outputs."""
        return self.spec(a).lossless_base == self.spec(b).lossless_base


class AccessFlags(enum.IntFlag):
    NONE = 0
    READ = 1
    WRITE = 2
    IMMUTABLE = 4
    REPLICABLE = 8
    PINNED = 16
    TEMPORARY = 32
    PAGED = 64
    VERSIONED = 128


class TensorKind(enum.IntEnum):
    """Tensor classes with distinct memory policies (section 8)."""

    WEIGHT = 1
    EXPERT_WEIGHT = 2
    ACTIVATION = 3
    KV_CACHE = 4
    INTERMEDIATE = 5
    LEARNING_UPDATE = 6
    COLD_WEIGHT = 7


@dataclass(frozen=True)
class MemoryPolicy:
    flags: AccessFlags
    summary: str

    @property
    def immutable(self) -> bool:
        return bool(self.flags & AccessFlags.IMMUTABLE)

    @property
    def replicable(self) -> bool:
        return bool(self.flags & AccessFlags.REPLICABLE)

    @property
    def temporary(self) -> bool:
        return bool(self.flags & AccessFlags.TEMPORARY)


_R, _W = AccessFlags.READ, AccessFlags.WRITE

MEMORY_POLICIES: Mapping[TensorKind, MemoryPolicy] = {
    TensorKind.WEIGHT: MemoryPolicy(
        _R | AccessFlags.IMMUTABLE | AccessFlags.PINNED, "immutable, locally resident"
    ),
    TensorKind.EXPERT_WEIGHT: MemoryPolicy(
        _R | AccessFlags.IMMUTABLE | AccessFlags.REPLICABLE, "replicate when beneficial"
    ),
    TensorKind.ACTIVATION: MemoryPolicy(
        _R | _W | AccessFlags.TEMPORARY, "temporary, dependency-managed"
    ),
    TensorKind.KV_CACHE: MemoryPolicy(
        _R | _W | AccessFlags.PAGED | AccessFlags.PINNED, "paged and dynamically allocated"
    ),
    TensorKind.INTERMEDIATE: MemoryPolicy(
        _R | _W | AccessFlags.TEMPORARY, "reusable scratch memory"
    ),
    TensorKind.LEARNING_UPDATE: MemoryPolicy(
        _R | _W | AccessFlags.VERSIONED, "versioned, explicitly synchronized"
    ),
    TensorKind.COLD_WEIGHT: MemoryPolicy(
        _R | AccessFlags.IMMUTABLE | AccessFlags.REPLICABLE, "stored in DDR or NVMe"
    ),
}

_DESCRIPTOR_STRUCT = struct.Struct("<QQQIIII4Q4Q")


@dataclass(frozen=True)
class TensorDescriptor:
    """Logical tensor record; field-for-field the ``TensorDescriptor`` C struct."""

    tensor_id: int
    version: int
    byte_length: int
    owner_device: int
    data_type: int
    compression_format: int
    access_flags: int
    shape: tuple[int, int, int, int]
    strides: tuple[int, int, int, int]

    WIRE_SIZE = _DESCRIPTOR_STRUCT.size  # 104 bytes

    def __post_init__(self) -> None:
        if len(self.shape) != 4 or len(self.strides) != 4:
            raise ValueError("shape and strides must have exactly four entries")
        if self.byte_length < 0 or self.version < 0:
            raise ValueError("byte_length and version must be non-negative")

    @property
    def n_elements(self) -> int:
        n = 1
        for dim in self.shape:
            n *= max(dim, 1)
        return n

    def pack(self) -> bytes:
        return _DESCRIPTOR_STRUCT.pack(
            self.tensor_id,
            self.version,
            self.byte_length,
            self.owner_device,
            int(self.data_type),
            int(self.compression_format),
            int(self.access_flags),
            *self.shape,
            *self.strides,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> TensorDescriptor:
        if len(buf) != _DESCRIPTOR_STRUCT.size:
            raise ValueError(f"descriptor must be {_DESCRIPTOR_STRUCT.size} bytes, got {len(buf)}")
        f = _DESCRIPTOR_STRUCT.unpack(buf)
        return cls(f[0], f[1], f[2], f[3], f[4], f[5], f[6], tuple(f[7:11]), tuple(f[11:15]))

    def bumped(self) -> TensorDescriptor:
        """Descriptor for the next version of a mutable tensor."""
        return replace(self, version=self.version + 1)


def pad_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Pad a shape of up to four dimensions with trailing zeros (0 = unused)."""
    if not 0 < len(shape) <= 4:
        raise ValueError("tensors have between one and four dimensions")
    return tuple(shape) + (0,) * (4 - len(shape))  # type: ignore[return-value]


def contiguous_strides(shape: tuple[int, ...], item_bytes: int) -> tuple[int, int, int, int]:
    """Row-major byte strides for ``shape``; unused dimensions get stride 0."""
    strides = [0, 0, 0, 0]
    step = item_bytes
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = step
        step *= shape[i]
    return tuple(strides)  # type: ignore[return-value]


def activation_bytes(n_elements: int, dtype: DType = DType.FP16) -> int:
    return math.ceil(n_elements * DTYPE_BITS[dtype] / 8.0)


@dataclass(frozen=True)
class TensorSpec:
    """Compile-time description of a tensor in the logical namespace."""

    tensor_id: int
    name: str
    kind: TensorKind
    shape: tuple[int, ...]
    dtype: DType
    nbytes: int
    fmt: CompressionFormat | None = None  # weights only: canonical stored format
    n_weights: int = 0

    @property
    def policy(self) -> MemoryPolicy:
        return MEMORY_POLICIES[self.kind]

    @property
    def is_weight(self) -> bool:
        return self.kind in (TensorKind.WEIGHT, TensorKind.EXPERT_WEIGHT, TensorKind.COLD_WEIGHT)

    def descriptor(self, owner_device: int, version: int = 0) -> TensorDescriptor:
        item = max(1, math.ceil(DTYPE_BITS[self.dtype] / 8.0))
        return TensorDescriptor(
            tensor_id=self.tensor_id,
            version=version,
            byte_length=self.nbytes,
            owner_device=owner_device,
            data_type=int(self.dtype),
            compression_format=int(self.fmt) if self.fmt is not None else 0,
            access_flags=int(self.policy.flags),
            shape=pad_shape(self.shape),
            strides=contiguous_strides(self.shape, item),
        )
