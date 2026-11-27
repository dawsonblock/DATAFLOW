"""BSP command descriptors and completion events."""

import dataclasses
import itertools
import random
import struct

import pytest

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
    NO_DEVICE,
    CrcError,
    Flags,
    FrameKind,
    Opcode,
    ProtocolError,
    Status,
    frame,
)

U64 = 2**64 - 1

#: One representative valid command per opcode.
VALID = {
    Opcode.ALLOCATE: Command(1, Opcode.ALLOCATE, 3, inputs=(70,), arg0=4096),
    Opcode.FREE: Command(2, Opcode.FREE, 3, inputs=(70,)),
    Opcode.LOAD: Command(3, Opcode.LOAD, 2, inputs=(71,), src_device=9, flags=Flags.RETAIN),
    Opcode.TRANSFER: Command(4, Opcode.TRANSFER, 2, inputs=(72,), src_device=0, deps=(3,)),
    Opcode.PREFETCH: Command(
        5,
        Opcode.PREFETCH,
        1,
        inputs=(73,),
        src_device=0,
        priority=1,
        flags=Flags.RETAIN | Flags.SPECULATIVE,
    ),
    Opcode.EXECUTE: Command(
        6,
        Opcode.EXECUTE,
        2,
        inputs=(72, 74),
        outputs=(75,),
        deps=(4,),
        precision=5,
        version=3,
        completion_id=600,
        flags=Flags.STREAM,
        timeout_ns=5_000_000,
    ),
    Opcode.REDUCE: Command(7, Opcode.REDUCE, 0, inputs=(75, 76), outputs=(77,), deps=(6,)),
    Opcode.BARRIER: Command(8, Opcode.BARRIER, NO_DEVICE, deps=(6, 7)),
    Opcode.COMPLETE: Command(9, Opcode.COMPLETE, 0, status=Status.SUCCESS, completion_id=600),
    Opcode.CANCEL: Command(10, Opcode.CANCEL, 1, arg0=5),
    Opcode.QUERY_STATUS: Command(11, Opcode.QUERY_STATUS, 1, arg0=6),
}

MAXIMAL = Command(
    op_id=U64, opcode=Opcode.EXECUTE, device_id=NO_DEVICE - 1,
    inputs=tuple(U64 - i for i in range(MAX_TENSORS)),
    outputs=tuple(range(MAX_TENSORS)),
    deps=tuple(U64 - 100 - i for i in range(MAX_DEPS)),
    precision=0xFFFF, priority=-0x8000, version=U64, completion_id=U64, status=Status.CRC_ERROR,
    src_device=NO_DEVICE - 1, flags=Flags(63), arg0=U64, arg1=U64, timeout_ns=U64,
)  # fmt: skip


def flip(buf: bytes, bit: int) -> bytes:
    out = bytearray(buf)
    out[bit // 8] ^= 1 << (bit % 8)
    return bytes(out)


def test_field_order_and_defaults_match_the_spec():
    names = [f.name for f in dataclasses.fields(Command)]
    assert names == [
        "op_id", "opcode", "device_id", "inputs", "outputs", "deps", "precision", "priority",
        "version", "completion_id", "status", "src_device", "flags", "arg0", "arg1", "timeout_ns",
    ]  # fmt: skip
    cmd = Command(1, Opcode.COMPLETE, 0)
    assert dataclasses.astuple(cmd)[3:] == (
        (), (), (), 0, 0, 0, 0, Status.PENDING, NO_DEVICE, Flags.NONE, 0, 0, 0,
    )  # fmt: skip
    assert (MAX_TENSORS, MAX_DEPS) == (8, 16)


def test_all_opcodes_are_covered():
    assert set(VALID) == set(Opcode)


@pytest.mark.parametrize("cmd", list(VALID.values()) + [MAXIMAL], ids=lambda c: c.opcode.name)
def test_command_round_trip(cmd):
    buf = cmd.encode()
    assert len(buf) == cmd.wire_size
    assert Command.decode(buf) == cmd
    assert cmd.encode() == buf  # pure function
    decoded = Command.decode(buf)
    assert type(decoded.opcode) is Opcode and type(decoded.status) is Status
    assert type(decoded.flags) is Flags and type(decoded.inputs) is tuple


def test_wire_sizes():
    assert Command(1, Opcode.COMPLETE, 0).wire_size == 88  # 72-byte descriptor + 16 framing
    assert VALID[Opcode.TRANSFER].wire_size == 88 + 8 * 2
    assert MAXIMAL.wire_size == 344  # 72 + 8 * (8 + 8 + 16) + 16
    assert len(MAXIMAL.encode()) == 344


def test_wire_layout_is_stable():
    """Golden bytes for one command: the layout is an interface, not an implementation detail."""
    payload = VALID[Opcode.EXECUTE].encode()[12:-4]
    fixed = struct.unpack_from("<6Q2IhHH6BI", payload)
    assert fixed == (6, 600, 3, 0, 0, 5_000_000, 2, NO_DEVICE, 0, 5, 2, 6, 0, 2, 1, 1, 0, 0)
    assert struct.unpack_from("<4Q", payload, 72) == (72, 74, 75, 4)
    assert len(payload) == 72 + 8 * 4


@pytest.mark.parametrize("cmd", [VALID[Opcode.EXECUTE], MAXIMAL], ids=["execute", "maximal"])
def test_every_single_bit_flip_is_detected(cmd):
    buf = cmd.encode()
    for bit in range(len(buf) * 8):
        with pytest.raises(ProtocolError):
            Command.decode(flip(buf, bit))


def test_random_multi_bit_corruption_is_detected():
    rng = random.Random(1234)
    buf = VALID[Opcode.EXECUTE].encode()
    for _ in range(3000):
        bits = rng.sample(range(len(buf) * 8), rng.choice((2, 3)))
        corrupted = buf
        for bit in bits:
            corrupted = flip(corrupted, bit)
        with pytest.raises(ProtocolError):
            Command.decode(corrupted)


def test_payload_corruption_is_a_crc_error():
    buf = VALID[Opcode.REDUCE].encode()
    with pytest.raises(CrcError):
        Command.decode(flip(buf, 12 * 8 + 3))  # op_id bit


def test_coercion_and_immutability():
    cmd = Command(1, 6, 2, inputs=[4, 5], outputs=[6], status=2, flags=3)
    assert cmd.opcode is Opcode.EXECUTE and cmd.status is Status.RUNNING
    assert cmd.flags == Flags.RETAIN | Flags.STREAM
    assert cmd.inputs == (4, 5) and cmd.outputs == (6,)
    assert cmd == Command.decode(cmd.encode())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.status = Status.SUCCESS
    with pytest.raises(ValueError):
        Command(1, 99, 2)
    with pytest.raises(TypeError):
        Command(1.5, Opcode.COMPLETE, 2)
    done = cmd.with_status(Status.SUCCESS)
    assert done.status is Status.SUCCESS and cmd.status is Status.RUNNING
    assert dataclasses.replace(done, status=Status.RUNNING) == cmd
    assert hash(cmd) == hash(Command(1, 6, 2, inputs=(4, 5), outputs=(6,), status=2, flags=3))


def but(opcode: Opcode, **changes) -> Command:
    return dataclasses.replace(VALID[opcode], **changes)


INVALID = {
    "too many inputs": but(Opcode.EXECUTE, inputs=tuple(range(MAX_TENSORS + 1))),
    "too many outputs": but(Opcode.EXECUTE, outputs=tuple(range(MAX_TENSORS + 1))),
    "too many deps": but(Opcode.EXECUTE, deps=tuple(range(100, 101 + MAX_DEPS))),
    "self dependency": but(Opcode.EXECUTE, deps=(4, 6)),
    "negative op id": but(Opcode.EXECUTE, op_id=-1),
    "op id too wide": but(Opcode.EXECUTE, op_id=U64 + 1),
    "tensor id too wide": but(Opcode.EXECUTE, inputs=(U64 + 1,)),
    "negative dep": but(Opcode.EXECUTE, deps=(-2,)),
    "device too wide": but(Opcode.EXECUTE, device_id=NO_DEVICE + 1),
    "src device too wide": but(Opcode.EXECUTE, src_device=NO_DEVICE + 1),
    "priority too wide": but(Opcode.EXECUTE, priority=0x8000),
    "precision too wide": but(Opcode.EXECUTE, precision=0x10000),
    "negative precision": but(Opcode.EXECUTE, precision=-1),
    "undefined flag": but(Opcode.EXECUTE, flags=64),
    "version too wide": but(Opcode.EXECUTE, version=U64 + 1),
    "timeout negative": but(Opcode.EXECUTE, timeout_ns=-1),
    "execute without output": but(Opcode.EXECUTE, outputs=()),
    "execute without device": but(Opcode.EXECUTE, device_id=NO_DEVICE),
    "reduce without output": but(Opcode.REDUCE, outputs=()),
    "transfer without source": but(Opcode.TRANSFER, src_device=NO_DEVICE),
    "transfer to itself": but(Opcode.TRANSFER, src_device=2),
    "transfer without tensor": but(Opcode.TRANSFER, inputs=()),
    "transfer of two tensors": but(Opcode.TRANSFER, inputs=(1, 2)),
    "transfer without destination": but(Opcode.TRANSFER, device_id=NO_DEVICE),
    "load without source": but(Opcode.LOAD, src_device=NO_DEVICE),
    "load of two tensors": but(Opcode.LOAD, inputs=(1, 2)),
    "prefetch to itself": but(Opcode.PREFETCH, src_device=1),
    "prefetch without tensor": but(Opcode.PREFETCH, inputs=()),
    "allocate zero bytes": but(Opcode.ALLOCATE, arg0=0),
    "allocate without tensor": but(Opcode.ALLOCATE, inputs=()),
    "allocate two tensors": but(Opcode.ALLOCATE, inputs=(1, 2)),
    "free without tensor": but(Opcode.FREE, inputs=()),
    "free without device": but(Opcode.FREE, device_id=NO_DEVICE),
    "cancel itself": but(Opcode.CANCEL, arg0=10),
    "query itself": but(Opcode.QUERY_STATUS, arg0=11),
    "barrier without deps": but(Opcode.BARRIER, deps=()),
}


@pytest.mark.parametrize("name", INVALID)
def test_validate_rejects(name):
    with pytest.raises(ProtocolError):
        validate(INVALID[name])
    with pytest.raises(ProtocolError):
        INVALID[name].encode()


def test_validate_accepts_boundaries():
    for cmd in list(VALID.values()) + [MAXIMAL]:
        validate(cmd)
    validate(but(Opcode.EXECUTE, priority=0x7FFF))
    validate(but(Opcode.FREE, arg0=0))  # only ALLOCATE carries a byte length
    validate(but(Opcode.CANCEL, arg0=0))  # op id 0 is a legal target
    validate(but(Opcode.BARRIER, device_id=0))  # a barrier may also live on a device
    validate(but(Opcode.REDUCE, inputs=()))  # fully fused reduction has no inputs left


def command_frame(fixed: tuple, ids: tuple = ()) -> bytes:
    payload = struct.pack("<6Q2IhHH6BI", *fixed) + struct.pack(f"<{len(ids)}Q", *ids)
    return frame(FrameKind.COMMAND, payload)


def test_decode_rejects_well_framed_garbage():
    """Checksums pass, so these exercise the semantic checks of the decoder."""
    good = (6, 0, 0, 0, 0, 0, 2, NO_DEVICE, 0, 0, 0, Opcode.EXECUTE, 0, 1, 1, 0, 0, 0)
    assert Command.decode(command_frame(good, (4, 5))) == Command(6, 6, 2, (4,), (5,))

    def changed(index, value):
        return good[:index] + (value,) + good[index + 1 :]

    cases = {
        "unknown opcode": command_frame(changed(11, 0), (4, 5)),
        "opcode past the table": command_frame(changed(11, 12), (4, 5)),
        "unknown status": command_frame(changed(12, 10), (4, 5)),
        "undefined flag": command_frame(changed(10, 0x40), (4, 5)),
        "reserved byte set": command_frame(changed(16, 1), (4, 5)),
        "reserved word set": command_frame(changed(17, 1), (4, 5)),
        "count beyond payload": command_frame(changed(13, 2), (4, 5)),
        "payload beyond count": command_frame(good, (4, 5, 6)),
        "too many inputs": command_frame(changed(13, 9), tuple(range(10))),
        "invalid semantics": command_frame(changed(14, 0), (4,)),  # EXECUTE without output
        "short payload": frame(FrameKind.COMMAND, bytes(71)),
        "wrong frame kind": frame(FrameKind.COMPLETION, struct.pack("<6Q2IhHH6BI", *good)),
        "trailing bytes": command_frame(good, (4, 5)) + b"\0",
        "two frames": command_frame(good, (4, 5)) * 2,
    }
    for name, buf in cases.items():
        with pytest.raises(ProtocolError):
            Command.decode(buf)
        assert name  # keeps the case label visible in failure output


def test_command_stream_round_trip():
    cmds = list(VALID.values()) + [MAXIMAL]
    buf = encode_stream(cmds)
    assert len(buf) == sum(c.wire_size for c in cmds)
    assert decode_stream(buf) == cmds
    assert decode_stream(b"") == [] and encode_stream([]) == b""
    with pytest.raises(ProtocolError):
        decode_stream(buf[:-1])
    with pytest.raises(ProtocolError, match="COMPLETION"):
        decode_stream(buf + CompletionEvent(1, 1, 0, Status.SUCCESS).encode())
    with pytest.raises(CrcError):
        decode_stream(flip(buf, len(buf) * 4))


@pytest.mark.parametrize("n", [0, 1, 16, 17, 32, 33, 256, 257, 1000])
def test_fan_in_narrows_wide_dependency_lists(n):
    ids = itertools.count(10_000)
    deps = list(range(n))
    barriers, narrowed = fan_in(deps, lambda: next(ids), device_id=4, priority=2)
    assert len(narrowed) <= MAX_DEPS
    if n <= MAX_DEPS:
        assert barriers == () and narrowed == tuple(deps)
    # Every barrier is encodable and only waits on earlier barriers or original deps.
    seen = set(deps)
    for barrier in barriers:
        validate(barrier)
        assert barrier.opcode is Opcode.BARRIER and set(barrier.deps) <= seen
        assert (barrier.device_id, barrier.priority) == (4, 2)
        seen.add(barrier.op_id)
    # Following the barriers from the narrowed list reaches exactly the original deps.
    by_id = {b.op_id: b for b in barriers}
    frontier, leaves = list(narrowed), set()
    while frontier:
        op = frontier.pop()
        frontier.extend(by_id[op].deps) if op in by_id else leaves.add(op)
    assert leaves == set(deps)
    waiter = Command(99_999, Opcode.REDUCE, 4, outputs=(1,), deps=narrowed)
    assert decode_stream(encode_stream([*barriers, waiter]))[-1] == waiter


def test_fan_in_removes_duplicates():
    barriers, narrowed = fan_in([3, 3, 5, 3], lambda: 1)
    assert barriers == () and narrowed == (3, 5)


def test_completion_event_round_trip():
    event = CompletionEvent(600, 6, 2, Status.SUCCESS, 1_234_567, 99_090_432, 7)
    buf = event.encode()
    assert len(buf) == 64  # 48-byte record + 16 framing
    assert CompletionEvent.decode(buf) == event and event.encode() == buf
    widest = CompletionEvent(U64, U64, NO_DEVICE, Status.CRC_ERROR, U64, U64, U64)
    assert CompletionEvent.decode(widest.encode()) == widest
    assert CompletionEvent(1, 2, 3, 6).status is Status.TIMEOUT
    for bit in range(len(buf) * 8):
        with pytest.raises(ProtocolError):
            CompletionEvent.decode(flip(buf, bit))


def test_completion_event_rejects_bad_input():
    with pytest.raises(ProtocolError):
        CompletionEvent(-1, 1, 0, Status.SUCCESS).encode()
    with pytest.raises(ProtocolError):
        CompletionEvent(1, 1, NO_DEVICE + 1, Status.SUCCESS).encode()
    with pytest.raises(ProtocolError):
        CompletionEvent.decode(VALID[Opcode.COMPLETE].encode())  # a COMMAND frame
    record = struct.pack("<5QIB3s", 1, 2, 3, 4, 5, 0, 10, b"\0\0\0")  # status 10 is undefined
    with pytest.raises(ProtocolError):
        CompletionEvent.decode(frame(FrameKind.COMPLETION, record))
    record = struct.pack("<5QIB3s", 1, 2, 3, 4, 5, 0, 3, b"\0\1\0")  # reserved byte set
    with pytest.raises(ProtocolError):
        CompletionEvent.decode(frame(FrameKind.COMPLETION, record))
    with pytest.raises(ProtocolError):
        CompletionEvent.decode(frame(FrameKind.COMPLETION, bytes(47)))


def test_completion_queue_overflow_is_never_silent():
    q = CompletionQueue(capacity=2)
    events = [CompletionEvent(i, i, 0, Status.SUCCESS) for i in range(4)]
    assert q.post(events[0]) and q.post(events[1])
    assert not q.overflowed and len(q) == 2
    assert q.post(events[2]) is False and q.post(events[3]) is False
    assert (q.dropped, q.overflowed, q.posted, len(q), q.high_water) == (2, True, 2, 2, 2)
    assert q.poll() == events[0]
    assert q.overflowed  # sticky until the consumer has resynchronised
    assert q.post(events[2]) and q.overflowed
    assert q.acknowledge_overflow() == 2 and not q.overflowed
    assert q.drain() == [events[1], events[2]]
    assert q.poll() is None and q.drain() == [] and len(q) == 0
    assert CompletionQueue().capacity == 1024
    with pytest.raises(ValueError):
        CompletionQueue(0)
