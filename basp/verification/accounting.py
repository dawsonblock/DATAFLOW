"""Event and bandwidth accounting checks (the evidence required for v0.1)."""

from __future__ import annotations

from basp.backends.virtual.system import VirtualSystem
from basp.streams.protocol import Opcode, Status

_REL = 1e-9


def check_accounting(system: VirtualSystem) -> list[str]:
    """Return every accounting inconsistency found (an empty list means the run is clean).

    * No resource served more work than capacity x elapsed time, and its busy
      time does not exceed the elapsed time.
    * Per-tag work on every resource sums to its total.
    * Bytes carried by each link direction equal the bytes of the transfers and
      streamed kernels routed over it (conservation: nothing appears or is lost).
    * Every submitted command reached a terminal state and was logged once.
    """
    problems: list[str] = []
    system.net.settle()
    elapsed = system.sim.now
    for key, res in system.budget.items():
        link = system.interconnect.link_res[key]
        background = sum(v for tag, v in link.by_tag.items() if tag in ("replicate", "prefetch"))
        if res.served > background * (1 + _REL) + 1e-6:
            problems.append(f"{res.name}: budget accounting exceeds background traffic")
    for res in system.net.resources.values():
        limit = res.capacity * elapsed
        if res.served > limit * (1 + _REL) + 1e-6:
            problems.append(f"{res.name}: served {res.served:.6g} exceeds capacity x time {limit:.6g}")
        if res.busy_time > elapsed * (1 + _REL) + 1e-15:
            problems.append(f"{res.name}: busy {res.busy_time} s exceeds elapsed {elapsed} s")
        tagged = sum(res.by_tag.values())
        if abs(tagged - res.served) > max(abs(res.served), 1.0) * 1e-9:
            problems.append(f"{res.name}: per-tag work {tagged:.6g} != total {res.served:.6g}")

    expected: dict[tuple[int, bool], float] = {}
    for state in system.cmds.values():
        cmd = state.cmd
        if cmd.opcode in (Opcode.TRANSFER, Opcode.LOAD, Opcode.PREFETCH) and state.moved:
            for hop in system.topology.route(cmd.src_device, cmd.device_id):
                key = (hop.link, hop.forward)
                expected[key] = expected.get(key, 0.0) + state.moved
        elif cmd.opcode in (Opcode.EXECUTE, Opcode.REDUCE) and state.status is Status.SUCCESS:
            for phase in state.kernel.phases:
                for key, work in phase.demands:
                    if key[0] == "link":
                        k = (key[1], key[2])
                        expected[k] = expected.get(k, 0.0) + work
    for key, res in system.interconnect.link_res.items():
        want = expected.get(key, 0.0)
        if abs(res.served - want) > max(want, 1.0) * 1e-6:
            problems.append(f"{res.name}: carried {res.served:.6g} bytes, transfers account for {want:.6g}")

    logged = [r.cmd_id for r in system.log]
    if len(logged) != len(set(logged)):
        problems.append("a command was logged more than once")
    for cmd_id, state in system.cmds.items():
        if not state.status.is_terminal:
            problems.append(f"command {cmd_id} ({state.cmd.opcode.name}) never reached a terminal state")
    if len(logged) != sum(1 for s in system.cmds.values() if s.status.is_terminal):
        problems.append("execution log does not cover every finished command")
    return problems
