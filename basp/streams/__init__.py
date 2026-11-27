"""BASP Stream Protocol (BSP).

Device-independent command, completion, activation and weight streams.  The
same frames are meant to travel over PCIe, CXL or a network link without the
execution graph changing; nothing here depends on the simulator.
"""

from basp.streams.activation import ActivationFrame, Reassembler, activation_bytes, packetize
from basp.streams.commands import (
    MAX_DEPS,
    MAX_TENSORS,
    Command,
    decode_stream,
    encode_stream,
    fan_in,
    validate,
)
from basp.streams.completion import CompletionEvent, CompletionQueue
from basp.streams.protocol import (
    BSP_MAGIC,
    BSP_VERSION,
    FRAME_OVERHEAD,
    NO_DEVICE,
    BoundedQueue,
    CrcError,
    Flags,
    FrameKind,
    Opcode,
    ProtocolError,
    QueueFull,
    Status,
    crc32,
    frame,
    iter_frames,
    unframe,
)
from basp.streams.weights import (
    WeightBlock,
    WeightStreamIndex,
    build_stream,
    read_blocks,
    read_range,
)

__all__ = [
    "BSP_MAGIC",
    "BSP_VERSION",
    "FRAME_OVERHEAD",
    "MAX_DEPS",
    "MAX_TENSORS",
    "NO_DEVICE",
    "ActivationFrame",
    "BoundedQueue",
    "Command",
    "CompletionEvent",
    "CompletionQueue",
    "CrcError",
    "Flags",
    "FrameKind",
    "Opcode",
    "ProtocolError",
    "QueueFull",
    "Reassembler",
    "Status",
    "WeightBlock",
    "WeightStreamIndex",
    "activation_bytes",
    "build_stream",
    "crc32",
    "decode_stream",
    "encode_stream",
    "fan_in",
    "frame",
    "iter_frames",
    "packetize",
    "read_blocks",
    "read_range",
    "unframe",
    "validate",
]  # fmt: skip
