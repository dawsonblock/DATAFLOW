"""BSP command descriptors and their wire encoding.

One :class:`Command` describes one operation (transfer, kernel execution,
reduction, ...) so that a device can run a dependent chain without the host
driving each step.  The payload of a COMMAND frame is little-endian::

    offset  size  field
    0       8     op_id          operation ID
    8       8     completion_id  result synchronisation tag
    16      8     version        expected tensor version
    24      8     arg0           immediate (ALLOCATE: bytes; CANCEL/QUERY_STATUS: target op)
    32      8     arg1           immediate
    40      8     timeout_ns     0 = no timeout
    48      4     device_id      execution / destination device
    52      4     src_device     source device (TRANSFER, LOAD, PREFETCH)
    56      2     priority       signed; lower is more urgent
    58      2     precision      numerical representation code
    60      2     flags          Flags
    62      1     opcode
    63      1     status
    64      1     n_inputs       <= MAX_TENSORS
    65      1     n_outputs      <= MAX_TENSORS
    66      1     n_deps         <= MAX_DEPS
    67      5     reserved       must be zero
    72      8*k   input tensor IDs, output tensor IDs, dependency op IDs (u64 each)

A minimal command is 72 payload bytes (88 framed); the largest is 328 (344 framed).
"""

from __future__ import annotations

import operator
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from basp.streams.protocol import (
    FRAME_OVERHEAD,
    NO_DEVICE,
    Flags,
    FrameKind,
    Opcode,
    ProtocolError,
    Status,
    frame,
    iter_frames,
    unframe_exact,
)

MAX_TENSORS = 8
MAX_DEPS = 16

_FIXED = struct.Struct("<6Q2IhHH6BI")
_U64 = (1 << 64) - 1
_ALL_FLAGS = sum(int(flag) for flag in Flags.__members__.values())
_MOVES = (Opcode.TRANSFER, Opcode.LOAD, Opcode.PREFETCH)
_ON_DEVICE = _MOVES + (Opcode.EXECUTE, Opcode.REDUCE, Opcode.ALLOCATE, Opcode.FREE)
_INT_FIELDS = (
    "op_id", "device_id", "precision", "priority", "version", "completion_id",
    "src_device", "arg0", "arg1", "timeout_ns",
)  # fmt: skip
_TUPLE_FIELDS = ("inputs", "outputs", "deps")


@dataclass(frozen=True)
class Command:
    """One BSP operation descriptor (immutable; see module docstring for the layout)."""

    op_id: int
    opcode: Opcode
    device_id: int
    inputs: tuple[int, ...] = ()
    outputs: tuple[int, ...] = ()
    deps: tuple[int, ...] = ()
    precision: int = 0
    priority: int = 0
    version: int = 0
    completion_id: int = 0
    status: Status = Status.PENDING
    src_device: int = NO_DEVICE
    flags: Flags = Flags.NONE
    arg0: int = 0
    arg1: int = 0
    timeout_ns: int = 0

    def __post_init__(self) -> None:
        put = object.__setattr__
        for name in _INT_FIELDS:
            put(self, name, operator.index(getattr(self, name)))
        for name in _TUPLE_FIELDS:
            put(self, name, tuple(operator.index(v) for v in getattr(self, name)))
        put(self, "opcode", Opcode(self.opcode))
        put(self, "status", Status(self.status))
        put(self, "flags", Flags(self.flags))

    @property
    def wire_size(self) -> int:
        """Encoded size in bytes, including the frame."""
        n_ids = len(self.inputs) + len(self.outputs) + len(self.deps)
        return FRAME_OVERHEAD + _FIXED.size + 8 * n_ids

    def with_status(self, status: Status) -> Command:
        return replace(self, status=status)

    def encode(self) -> bytes:
        """Validate and serialise as one COMMAND frame."""
        validate(self)
        ids = self.inputs + self.outputs + self.deps
        fixed = _FIXED.pack(
            self.op_id, self.completion_id, self.version, self.arg0, self.arg1, self.timeout_ns,
            self.device_id, self.src_device, self.priority, self.precision, int(self.flags),
            self.opcode, self.status, len(self.inputs), len(self.outputs), len(self.deps), 0, 0,
        )  # fmt: skip
        return frame(FrameKind.COMMAND, fixed + struct.pack(f"<{len(ids)}Q", *ids))

    @classmethod
    def decode(cls, buf: bytes) -> Command:
        """Parse a buffer holding exactly one COMMAND frame."""
        return _from_payload(unframe_exact(buf, FrameKind.COMMAND))


def _from_payload(payload: bytes) -> Command:
    if len(payload) < _FIXED.size:
        raise ProtocolError(f"command payload too short: {len(payload)} bytes")
    (op_id, completion_id, version, arg0, arg1, timeout_ns, device_id, src_device, priority,
     precision, flags, opcode, status, n_in, n_out, n_deps, pad8, pad32) = _FIXED.unpack_from(
        payload
    )  # fmt: skip
    if pad8 or pad32:
        raise ProtocolError("reserved command bytes must be zero")
    n_ids = n_in + n_out + n_deps
    if len(payload) != _FIXED.size + 8 * n_ids:
        raise ProtocolError(f"command payload is {len(payload)} bytes but declares {n_ids} ids")
    ids = struct.unpack_from(f"<{n_ids}Q", payload, _FIXED.size)
    try:
        cmd = Command(
            op_id, opcode, device_id, ids[:n_in], ids[n_in : n_in + n_out], ids[n_in + n_out :],
            precision, priority, version, completion_id, status, src_device, flags,
            arg0, arg1, timeout_ns,
        )  # fmt: skip
    except ValueError as exc:
        raise ProtocolError(str(exc)) from None
    validate(cmd)
    return cmd


def _need(cond: bool, cmd: Command, message: str) -> None:
    if not cond:
        raise ProtocolError(f"{cmd.opcode.name} op {cmd.op_id}: {message}")


def validate(cmd: Command) -> None:
    """Raise :class:`ProtocolError` unless ``cmd`` is well formed and encodable."""
    for name in ("op_id", "completion_id", "version", "arg0", "arg1", "timeout_ns"):
        _need(0 <= getattr(cmd, name) <= _U64, cmd, f"{name} does not fit in 64 bits")
    for name in ("device_id", "src_device"):
        _need(0 <= getattr(cmd, name) <= NO_DEVICE, cmd, f"{name} does not fit in 32 bits")
    _need(-0x8000 <= cmd.priority <= 0x7FFF, cmd, "priority does not fit in 16 bits")
    _need(0 <= cmd.precision <= 0xFFFF, cmd, "precision does not fit in 16 bits")
    _need(not int(cmd.flags) & ~_ALL_FLAGS, cmd, f"undefined flag bits 0x{int(cmd.flags):x}")
    _need(len(cmd.inputs) <= MAX_TENSORS, cmd, f"more than {MAX_TENSORS} input tensors")
    _need(len(cmd.outputs) <= MAX_TENSORS, cmd, f"more than {MAX_TENSORS} output tensors")
    _need(len(cmd.deps) <= MAX_DEPS, cmd, f"more than {MAX_DEPS} dependencies")
    for value in cmd.inputs + cmd.outputs + cmd.deps:
        _need(0 <= value <= _U64, cmd, "tensor or dependency id does not fit in 64 bits")
    _need(cmd.op_id not in cmd.deps, cmd, "command depends on itself")

    op = cmd.opcode
    if op in _ON_DEVICE:
        _need(cmd.device_id != NO_DEVICE, cmd, "needs a target device")
    if op in _MOVES:
        _need(cmd.src_device != NO_DEVICE, cmd, "needs a source device")
        _need(cmd.src_device != cmd.device_id, cmd, "source and destination are the same device")
        _need(len(cmd.inputs) == 1, cmd, "moves exactly one tensor")
    elif op in (Opcode.EXECUTE, Opcode.REDUCE):
        _need(len(cmd.outputs) >= 1, cmd, "needs at least one output tensor")
    elif op in (Opcode.ALLOCATE, Opcode.FREE):
        _need(len(cmd.inputs) == 1, cmd, "names exactly one tensor")
        _need(op is Opcode.FREE or cmd.arg0 > 0, cmd, "byte length (arg0) must be positive")
    elif op in (Opcode.CANCEL, Opcode.QUERY_STATUS):
        _need(cmd.arg0 != cmd.op_id, cmd, "target op (arg0) is the command itself")
    elif op is Opcode.BARRIER:
        _need(len(cmd.deps) >= 1, cmd, "needs at least one dependency")


def encode_stream(cmds: Iterable[Command]) -> bytes:
    """Concatenated COMMAND frames, in order."""
    return b"".join(cmd.encode() for cmd in cmds)


def decode_stream(buf: bytes) -> list[Command]:
    """Parse a buffer of concatenated COMMAND frames."""
    out = []
    for kind, payload in iter_frames(buf):
        if kind != FrameKind.COMMAND:
            raise ProtocolError(f"unexpected {kind.name} frame in a command stream")
        out.append(_from_payload(payload))
    return out


def fan_in(
    deps: Sequence[int],
    new_op_id: Callable[[], int],
    device_id: int = NO_DEVICE,
    priority: int = 0,
) -> tuple[tuple[Command, ...], tuple[int, ...]]:
    """Narrow a dependency list to at most ``MAX_DEPS`` ids using BARRIER commands.

    Returns ``(barriers, deps)``: submit the barriers first, then use the returned
    ids as the ``deps`` of the waiting command.  Lists that already fit are
    returned unchanged with no barriers.
    """
    barriers: list[Command] = []
    level = tuple(dict.fromkeys(deps))
    while len(level) > MAX_DEPS:
        merged = []
        for i in range(0, len(level), MAX_DEPS):
            chunk = level[i : i + MAX_DEPS]
            if len(chunk) == 1:
                merged.append(chunk[0])
                continue
            barrier = Command(new_op_id(), Opcode.BARRIER, device_id, deps=chunk, priority=priority)
            barriers.append(barrier)
            merged.append(barrier.op_id)
        level = tuple(merged)
    return tuple(barriers), level
