"""Provenance records and deterministic manifests of files and directories.

``directory_manifest`` lists every file under a directory with its size and SHA-256, sorted,
and hashes the list: the same content gives the same root hash on any machine, whatever the
file order on disk or the modification times.  ``provenance`` records everything a stored
result depends on; ``stale_reasons`` says why a stored result no longer matches.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .identity import (
    config_path,
    environment,
    git_state,
    sha256_file,
    sha256_json,
    source_fingerprint,
)

_SKIP = ("__pycache__", ".git", "*.pyc", ".pytest_cache", ".ruff_cache")


def directory_manifest(path: str | Path, *, exclude: Iterable[str] = _SKIP) -> dict[str, Any]:
    """Every file under ``path`` (relative POSIX paths, sorted) with size and SHA-256, and a root hash."""
    root = Path(path)
    patterns = tuple(exclude)
    files = []
    for file in sorted(root.rglob("*")):
        rel = file.relative_to(root)
        if any(fnmatch.fnmatch(part, pat) for part in rel.parts for pat in patterns) or not file.is_file():
            continue
        files.append({"path": rel.as_posix(), "size": file.stat().st_size, "sha256": sha256_file(file)})
    return {"root": sha256_json(files), "files": files}


def path_hash(path: str | Path | None) -> str | None:
    """Content hash of a file, or the manifest root hash of a directory; ``None`` if absent."""
    if path is None:
        return None
    p = Path(path)
    if p.is_dir():
        return directory_manifest(p)["root"]
    return sha256_file(p) if p.exists() else None


def provenance(*, configs: Iterable[str | Path] = (), inputs: Mapping[str, str | Path | None] | None = None,
               parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """Everything a stored result depends on.

    ``configs`` are profile/estimate names or paths; ``inputs`` maps a role ("trace",
    "measurements", "evidence", ...) to a file or a directory (hashed by its manifest) or
    ``None`` (not used); ``parameters`` are the run's own settings, hashed together.
    """
    record: dict[str, Any] = {
        "source": source_fingerprint(),
        "configs": {Path(config_path(c)).stem: sha256_file(config_path(c)) for c in configs},
        "inputs": {role: path_hash(p) for role, p in (inputs or {}).items()},
        "parameters": sha256_json(parameters or {}),
        "environment": environment(),
    }
    git = git_state()
    if git is not None:
        record["git"] = git
    return record


def stale_reasons(recorded: Mapping[str, Any] | None, current: Mapping[str, Any]) -> list[str]:
    """Why a stored result no longer corresponds to ``current`` (empty: it still does).

    The environment and git commit are informational; a change to the simulator, a
    configuration, an input file or directory, or the run's parameters makes a result stale.
    """
    if not recorded:
        return ["no provenance recorded"]
    reasons = []
    if recorded.get("source") != current.get("source"):
        reasons.append("simulator source changed")
    for key in ("configs", "inputs"):
        old, new = recorded.get(key, {}), current.get(key, {})
        for name in sorted(set(old) | set(new)):
            if old.get(name) != new.get(name):
                reasons.append(f"{key[:-1]} {name!r} changed")
    if recorded.get("parameters") != current.get("parameters"):
        reasons.append("run parameters changed")
    return reasons
