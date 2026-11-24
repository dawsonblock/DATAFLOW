"""Loading of hardware profiles and workload descriptions from ``configs/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from basp.core.topology import Topology
from basp.workloads.model import QWEN3_30B_A3B, MoEModelSpec, tiny_spec

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

MODELS: dict[str, MoEModelSpec] = {
    "qwen3-30b-a3b": QWEN3_30B_A3B,
    "tiny": tiny_spec(),
}


def _resolve(name: str | Path) -> Path:
    path = Path(name)
    if path.suffix != ".json":
        path = CONFIG_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no configuration file {path}")
    return path


def load_json(name: str | Path) -> dict[str, Any]:
    with _resolve(name).open() as handle:
        return json.load(handle)


def load_topology(name: str | Path) -> Topology:
    """Load a hardware profile, e.g. ``desktop_gpu12`` or ``basp_4module``."""
    return Topology.from_dict(load_json(name))


def load_workload(name: str | Path) -> dict[str, Any]:
    data = load_json(name)
    if data["model"] not in MODELS:
        raise ValueError(f"unknown model {data['model']!r}; known: {sorted(MODELS)}")
    return data
