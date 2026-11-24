"""Dependency verification of an executed schedule."""

from __future__ import annotations

from collections.abc import Sequence

from basp.backends.virtual.runtime import OpRecord
from basp.backends.virtual.system import VirtualSystem
from basp.streams.protocol import Status

_TOL = 1e-15


def check_dependencies(op_log: Sequence[OpRecord], system: VirtualSystem | None = None) -> list[str]:
    """Return every dependency violation found (an empty list means the run is clean).

    Checks that each operation ran exactly once, finished, and did not start
    before all of its graph dependencies had finished; and, given the system,
    that no command started before the commands it depends on completed and
    that the devices reported no residency violation.
    """
    problems: list[str] = []
    seen: dict[int, OpRecord] = {}
    for record in op_log:
        if record.op_id in seen:
            problems.append(f"op {record.name} was scheduled twice")
        seen[record.op_id] = record
        if record.started is None or record.finished is None:
            problems.append(f"op {record.name} never completed")
    for record in op_log:
        if record.started is None:
            continue
        for dep in record.deps:
            before = seen.get(dep)
            if before is None:
                continue  # dependency belongs to an earlier, already verified batch
            if before.finished is None or before.finished > record.started + _TOL:
                problems.append(f"op {record.name} started before its dependency {before.name} finished")
    if system is not None:
        problems.extend(system.violations)
        for state in system.cmds.values():
            if state.status is not Status.SUCCESS or state.started is None:
                continue
            for dep in state.cmd.deps:
                other = system.cmds[dep]
                if other.finished is None or other.finished > state.started + _TOL:
                    problems.append(
                        f"command {state.cmd.op_id} started before dependency {dep} completed"
                    )
    return problems
