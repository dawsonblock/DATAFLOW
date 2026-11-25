"""Evidence levels and provenance: what a number is, and exactly what produced it.

Every figure the project reports carries one of four evidence levels:

=====  ==========================================================================
E0     theoretical estimate, argument by construction, or a guessed parameter
E1     simulation that has not been validated against measurements
E2     simulation whose model of *existing* hardware passed held-out validation
E3     direct measurement of the thing being claimed
=====  ==========================================================================

The level of a claim about hypothetical hardware can never exceed E2 before a prototype
exists: calibrating the simulator on a GPU validates the model of that GPU, not of a
compute-memory module.

``provenance()`` records what a result was computed from -- a fingerprint of the simulator's
source, hashes of every configuration and input file, the software environment -- so a later
reader (or the report generator) can tell whether a stored result is **stale**: produced by
different code, profiles, estimates or traces than the ones in the repository now.
"""

from __future__ import annotations

import enum
import hashlib
import json
import platform
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

import basp

PACKAGE_DIR = Path(basp.__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
CONFIG_DIR = REPO_DIR / "configs"

#: source files whose change cannot alter a simulated number (rendering and decision logic)
_NOT_SIMULATION = {"experiments/report.py", "experiments/report_v3.py", "experiments/gate.py", "evidence.py"}


class EvidenceLevel(enum.IntEnum):
    E0 = 0
    E1 = 1
    E2 = 2
    E3 = 3

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]

    def __str__(self) -> str:
        return f"{self.name} ({self.description})"


_DESCRIPTIONS = {
    EvidenceLevel.E0: "estimate or argument",
    EvidenceLevel.E1: "uncalibrated simulation",
    EvidenceLevel.E2: "simulation validated on existing hardware",
    EvidenceLevel.E3: "measurement",
}


# -- hashing ----------------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash of a JSON-serialisable object, independent of key order and formatting."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)
    return sha256_bytes(text.encode())


def source_fingerprint() -> str:
    """Hash of the simulator's source code (``basp/``, excluding rendering and decision code)."""
    h = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        rel = path.relative_to(PACKAGE_DIR).as_posix()
        if rel in _NOT_SIMULATION:
            continue
        h.update(rel.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def config_path(name: str | Path) -> Path:
    path = Path(name)
    return path if path.suffix == ".json" else CONFIG_DIR / f"{name}.json"


def git_state() -> dict[str, Any] | None:
    """Commit and dirty flag of the working tree, or ``None`` outside a git checkout."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip()  # fmt: skip
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_DIR,
                               capture_output=True, text=True, timeout=10, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": bool(dirty)}


def environment() -> dict[str, str]:
    return {"basp": basp.__version__, "python": platform.python_version(), "numpy": np.__version__,
            "platform": platform.platform()}  # fmt: skip


# -- provenance --------------------------------------------------------------------------------
def provenance(*, configs: Iterable[str | Path] = (), inputs: Mapping[str, str | Path | None] | None = None,
               parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """Everything a stored result depends on.

    ``configs`` are profile/estimate names or paths; ``inputs`` maps a role ("trace",
    "measurements", ...) to a file path (``None``: not used); ``parameters`` are the run's
    own settings (workload, seeds, budgets), hashed together.
    """
    record: dict[str, Any] = {
        "source": source_fingerprint(),
        "configs": {Path(config_path(c)).stem: sha256_file(config_path(c)) for c in configs},
        "inputs": {role: (sha256_file(p) if p is not None and Path(p).exists() else None)
                   for role, p in (inputs or {}).items()},
        "parameters": sha256_json(parameters or {}),
        "environment": environment(),
    }
    git = git_state()
    if git is not None:
        record["git"] = git
    return record


def stale_reasons(recorded: Mapping[str, Any] | None, current: Mapping[str, Any]) -> list[str]:
    """Why a stored result no longer corresponds to ``current`` (empty: it still does).

    The environment and git commit are informational: a documentation commit or a different
    machine does not make a simulation stale; a change to the simulator, a configuration, an
    input file or the run's parameters does.
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
