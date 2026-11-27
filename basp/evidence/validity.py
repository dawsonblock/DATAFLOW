"""Is a recorded measurement still applicable?  One rule and one rendering, used everywhere.

A measurement record (a benchmark result, a parity result, a study) carries an ``identity``:
the runtime code that executed it, the tool that measured it, the machine, the model files
(full-content hashes) and the parameters.  ``check`` compares it with the present:

* the identity must be complete -- a record that does not say what produced it cannot be
  shown to apply to anything;
* the runtime fingerprint must be the current one (the code that ran is the code here);
* the measuring tool must be unchanged;
* model files, when the caller can name the current ones, must be the same files.

``note`` renders the outcome the same way for every kind of record, so no table can show a
stale result without saying so.  (Value-level checks -- NaN, ranges -- are ``validation``.)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .identity import (
    environment,
    file_identity,
    files_fingerprint,
    machine_identity,
    runtime_fingerprint,
)

REQUIRED = ("runtime", "tool", "machine", "models", "parameters")


def measurement_identity(*, tool: str, tool_files, models: Mapping[str, str | Path],
                         parameters: Mapping[str, Any], backend: str | None = None) -> dict[str, Any]:  # fmt: skip
    """The identity to store with a measurement made now."""
    software = environment()
    try:
        import torch

        software["torch"] = torch.__version__
    except ImportError:
        pass
    return {"runtime": runtime_fingerprint(), "tool": {"name": tool, "sha256": files_fingerprint(tool_files)},
            "machine": machine_identity(), "software": software, "backend": backend,
            "models": {role: file_identity(path) for role, path in sorted(models.items())},
            "parameters": dict(parameters)}  # fmt: skip


def check(record: Mapping[str, Any], *, tool_files=None,
          models: Mapping[str, str | Path] | None = None) -> dict[str, Any]:  # fmt: skip
    """``{"valid": bool, "reasons": [...]}`` for a stored record, against the code (and, if
    ``models`` is given, the model files) present now."""
    identity = record.get("identity")
    reasons: list[str] = []
    if not isinstance(identity, Mapping):
        return {"valid": False, "reasons": ["no identity recorded: it cannot be shown what produced this result"]}
    missing = [k for k in REQUIRED if not identity.get(k) and identity.get(k) != {}]
    if missing:
        reasons.append("identity incomplete: " + ", ".join(missing) + " not recorded")
    if identity.get("runtime") and identity["runtime"] != runtime_fingerprint():
        reasons.append("the runtime code changed since this was measured")
    tool = identity.get("tool") or {}
    if tool_files is not None and tool.get("sha256") and tool["sha256"] != files_fingerprint(tool_files):
        reasons.append("the measuring tool changed since this was measured")
    for role, entry in (identity.get("models") or {}).items():
        if not (isinstance(entry, Mapping) and entry.get("sha256")):
            reasons.append(f"model file {role!r} is not identified by its content")
        elif models and role in models and file_identity(models[role])["sha256"] != entry["sha256"]:
            reasons.append(f"model file {role!r} changed since this was measured")
    return {"valid": not reasons, "reasons": reasons}


def note(validity: Mapping[str, Any]) -> str:
    """Empty for an applicable record; otherwise the marker every table shows."""
    if validity["valid"]:
        return ""
    return " -- NOT APPLICABLE (" + "; ".join(validity["reasons"]) + ")"
