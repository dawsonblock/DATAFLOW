"""Identities: what exactly produced a number -- source code, configuration, files, machine,
software environment.  (Formerly ``basp/evidence.py``; everything is re-exported from
``basp.evidence``.)
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

import basp

PACKAGE_DIR = Path(basp.__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
CONFIG_DIR = REPO_DIR / "configs"

#: source files whose change cannot alter a simulated number: rendering and decision logic, and
#: the physical runtime with its verification and memory-domain code, which the simulator never
#: imports (the runtime has its own fingerprint, ``basp.verification.parity_report``)
_NOT_SIMULATION_FILES = {
    "experiments/report.py", "experiments/report_v4.py", "experiments/gate.py",
    "backends/execution.py", "backends/cpu/backend.py", "backends/cpu/kernels.py",
    "models/quants.py", "models/gguf_tensors.py", "models/gguf_writer.py",
    "memory/domains.py", "memory/admission.py", "memory/migration.py", "memory/expert_cache.py",
    "memory/hierarchy.py", "memory/prefetch.py",
    "verification/logits.py", "verification/routing.py", "verification/generation.py",
    "verification/perplexity.py", "verification/intermediate_tensors.py", "verification/parity_report.py",
    "verification/reference_runtime.py",
}  # fmt: skip
_NOT_SIMULATION_DIRS = ("evidence/", "runtime/", "backends/cuda/", "backends/metal/", "backends/ggml/",
                        "backends/remote/")  # fmt: skip


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
        if rel in _NOT_SIMULATION_FILES or rel.startswith(_NOT_SIMULATION_DIRS):
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


# -- files, executables, machines --------------------------------------------------------------
_FILE_CACHE: dict[tuple[str, int, int], str] = {}


def file_identity(path: str | Path) -> dict[str, Any]:
    """Size and SHA-256 of a file's full content (cached by path, size and modification time).

    A model's weights or a benchmark executable is identified by its whole content: a header
    fingerprint would not notice changed weights.
    """
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _FILE_CACHE:
        _FILE_CACHE[key] = sha256_file(path)
    return {"name": path.name, "size": stat.st_size, "sha256": _FILE_CACHE[key]}


def profile_identity(profile: Mapping[str, Any]) -> str:
    """Fingerprint of a hardware profile's content (comments and calibration notes excluded)."""

    def strip(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {k: strip(v) for k, v in obj.items() if not str(k).startswith("_")}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    return sha256_json(strip(profile))
