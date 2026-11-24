"""Device capability descriptions used by the compiler and the scheduler."""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from basp.core.graph import OpCode
from basp.core.tensor import CompressionFormat


class DeviceKind(enum.IntEnum):
    CPU = 1
    GPU = 2
    NEAR_MEMORY = 3
    STORAGE = 4


ALL_OPCODES = frozenset(OpCode)


@dataclass(frozen=True)
class DeviceCapabilities:
    """What a device can execute and how it consumes weight streams.

    ``decode_rate`` maps each consumable format to its decode throughput in
    *compressed input bytes per second* (``math.inf`` when decoding never limits
    the pipeline).  Formats in ``stream_formats`` feed the arithmetic units
    through a fused decode->MAC pipeline; other consumable formats need a staged
    pass that materialises ``expanded_format`` in device memory first.
    ``stream_ingest`` allows consuming weights that arrive over a link as an
    overlapped stream instead of storing them first.
    """

    opcodes: frozenset[OpCode] = ALL_OPCODES
    decode_rate: Mapping[CompressionFormat, float] = field(default_factory=dict)
    stream_formats: frozenset[CompressionFormat] = frozenset()
    expanded_format: CompressionFormat = CompressionFormat.INT4
    stream_ingest: bool = False

    def supports(self, opcode: OpCode) -> bool:
        return opcode in self.opcodes

    def can_consume(self, fmt: CompressionFormat) -> bool:
        return fmt in self.decode_rate

    def fused(self, fmt: CompressionFormat) -> bool:
        return fmt in self.stream_formats and fmt in self.decode_rate

    def decode_limited(self, fmt: CompressionFormat) -> bool:
        return math.isfinite(self.decode_rate.get(fmt, math.inf))

    @property
    def can_compute(self) -> bool:
        return bool(self.opcodes)


NO_COMPUTE = DeviceCapabilities(opcodes=frozenset())
