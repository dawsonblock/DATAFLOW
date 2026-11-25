"""Completion handles (re-exported from the backend contract) and helpers."""

from __future__ import annotations

from collections.abc import Iterable

from basp.backends.execution import Completion


def wait_all(completions: Iterable[Completion]) -> list[Completion]:
    out = list(completions)
    failed = [c for c in out if c.error]
    if failed:
        raise RuntimeError(f"{len(failed)} operation(s) failed: {failed[0].op}: {failed[0].error}")
    pending = [c for c in out if not c.done]
    if pending:
        raise RuntimeError(f"{len(pending)} operation(s) not complete; synchronize their backend first")
    return out


__all__ = ["Completion", "wait_all"]
