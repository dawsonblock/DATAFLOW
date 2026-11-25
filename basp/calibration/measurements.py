"""Benchmark measurements with a declared calibration/validation split.

A :class:`Measurement` is one measured throughput figure from a real machine (for example one
``llama-bench`` result).  Each record declares, at the moment it is recorded, whether it belongs
to the *calibration* split (used to tune simulator parameters) or the *validation* split (used
only to judge predictive accuracy).

The split is never chosen after seeing results: there is deliberately no ``split_by(predicate)``
helper, and a :class:`MeasurementSet` refuses to hold the same benchmark in both splits, because
matching a benchmark that was used to tune the simulator does not establish predictive accuracy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SPLITS = ("calibration", "validation")
KINDS = ("decode", "prefill", "batched")

_FORMAT = "basp.measurements"
_VERSION = 1


@dataclass(frozen=True)
class Measurement:
    """One measured throughput figure.

    ``n_gpu_layers`` follows llama.cpp ``-ngl``: the number of layers offloaded to the GPU, with
    ``-1`` meaning all of them; ``n_cpu_moe`` follows ``--n-cpu-moe``: the expert weights of the
    first N layers stay on the CPU.  ``tokens_per_s`` is generated tokens per second for
    ``decode`` and ``batched`` (aggregate over the batch) and prompt tokens per second for
    ``prefill``.  ``context_len`` is the KV-cache length the measured tokens see: for ``decode``
    the average over the generated tokens, for ``prefill`` the depth before the prompt (the
    prompt's own tokens come on top).  ``provenance`` records where the number came from (tool
    build, model file fingerprint, plan) and never affects comparisons.
    """

    id: str
    machine: str
    split: str
    kind: str
    model: str
    tokens_per_s: float
    bits_per_weight: float = 4.5
    n_gpu_layers: int = 0
    context_len: int = 0
    batch: int = 1
    threads: int = 0
    stddev_tokens_per_s: float = 0.0
    source: str = ""
    notes: str = ""
    n_cpu_moe: int = 0
    provenance: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        for name in ("id", "machine", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"measurement: {name} must be a non-empty string, got {value!r}")
        where = f"measurement {self.id!r}"
        if self.split not in SPLITS:
            raise ValueError(f"{where}: split must be one of {SPLITS}, got {self.split!r}")
        if self.kind not in KINDS:
            raise ValueError(f"{where}: kind must be one of {KINDS}, got {self.kind!r}")
        if not (isinstance(self.tokens_per_s, (int, float)) and math.isfinite(self.tokens_per_s)
                and self.tokens_per_s > 0):  # fmt: skip
            raise ValueError(f"{where}: tokens_per_s must be positive and finite, "
                             f"got {self.tokens_per_s!r}")  # fmt: skip
        if not (math.isfinite(self.bits_per_weight) and self.bits_per_weight > 0):
            raise ValueError(f"{where}: bits_per_weight must be positive, "
                             f"got {self.bits_per_weight!r}")  # fmt: skip
        if self.n_gpu_layers < -1:
            raise ValueError(f"{where}: n_gpu_layers must be >= -1 (-1 = all layers)")
        if self.n_cpu_moe < 0:
            raise ValueError(f"{where}: n_cpu_moe must be >= 0")
        if not isinstance(self.provenance, dict):
            raise TypeError(f"{where}: provenance must be a mapping")
        if self.context_len < 0 or self.threads < 0:
            raise ValueError(f"{where}: context_len and threads must be >= 0")
        if self.batch < 1:
            raise ValueError(f"{where}: batch must be >= 1")
        if not (math.isfinite(self.stddev_tokens_per_s) and self.stddev_tokens_per_s >= 0):
            raise ValueError(f"{where}: stddev_tokens_per_s must be >= 0")

    @property
    def workload_key(self) -> tuple:
        """What is computed: model, phase, context and batch -- the same on every machine."""
        return (self.model, self.kind, self.context_len, self.batch)

    @property
    def execution_key(self) -> tuple:
        """How and where: machine, weight file size per parameter, offload split and threads."""
        return (self.machine, self.bits_per_weight, self.n_gpu_layers, self.n_cpu_moe, self.threads)

    @property
    def signature(self) -> tuple:
        """Identity of the benchmark (workload on an execution setup), independent of when or how
        often it was run: repetitions share a signature."""
        return (*self.execution_key, *self.workload_key)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Measurement:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"measurement {data.get('id')!r}: unknown fields {unknown}")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"measurement {data.get('id')!r}: {exc}") from None


def find_problems(measurements: Iterable[Measurement]) -> list[str]:
    """Integrity problems of a collection of measurements (an empty list means it is sound).

    * every id is unique;
    * no benchmark signature appears in both the calibration and the validation split.
    """
    problems: list[str] = []
    by_id: dict[str, int] = {}
    by_signature: dict[tuple, dict[str, list[str]]] = {}
    for m in measurements:
        by_id[m.id] = by_id.get(m.id, 0) + 1
        by_signature.setdefault(m.signature, {}).setdefault(m.split, []).append(m.id)
    for mid, count in by_id.items():
        if count > 1:
            problems.append(f"measurement id {mid!r} appears {count} times")
    for splits in by_signature.values():
        if len(splits) > 1:
            problems.append(
                "the same benchmark is in both splits: calibration ids "
                f"{sorted(splits['calibration'])} and validation ids {sorted(splits['validation'])}"
            )
    return problems


class MeasurementSet:
    """An ordered collection of measurements that enforces the integrity rules on every add."""

    def __init__(self, measurements: Iterable[Measurement] = ()) -> None:
        self._items: list[Measurement] = []
        self._ids: set[str] = set()
        self._split_of: dict[tuple, tuple[str, str]] = {}  # signature -> (split, first id)
        for m in measurements:
            self.add(m)

    def add(self, measurement: Measurement) -> Measurement:
        """Add a measurement; raises ``ValueError`` naming the ids involved if a rule is broken."""
        m = measurement
        if m.id in self._ids:
            raise ValueError(f"duplicate measurement id {m.id!r}")
        seen = self._split_of.get(m.signature)
        if seen is not None and seen[0] != m.split:
            raise ValueError(
                f"measurement {m.id!r} ({m.split}) repeats the benchmark of {seen[1]!r} "
                f"({seen[0]}): a benchmark may not appear in both splits"
            )
        self._items.append(m)
        self._ids.add(m.id)
        self._split_of.setdefault(m.signature, (m.split, m.id))
        return m

    def check(self) -> list[str]:
        """Recompute the integrity problems without raising (always empty for a sound set)."""
        return find_problems(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Measurement]:
        return iter(self._items)

    def __contains__(self, measurement_id: object) -> bool:
        return measurement_id in self._ids

    @property
    def calibration(self) -> tuple[Measurement, ...]:
        return tuple(m for m in self._items if m.split == "calibration")

    @property
    def validation(self) -> tuple[Measurement, ...]:
        return tuple(m for m in self._items if m.split == "validation")

    def machines(self) -> tuple[str, ...]:
        return tuple(sorted({m.machine for m in self._items}))

    def to_json(self) -> dict[str, Any]:
        """Stable representation: measurements ordered by id."""
        ordered = sorted(self._items, key=lambda m: m.id)
        return {"format": _FORMAT, "version": _VERSION,
                "measurements": [m.to_json() for m in ordered]}  # fmt: skip

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> MeasurementSet:
        if not isinstance(data, dict) or "measurements" not in data:
            raise ValueError("measurement set: expected an object with a 'measurements' list")
        if data.get("format", _FORMAT) != _FORMAT or data.get("version", _VERSION) != _VERSION:
            raise ValueError("measurement set: unsupported format/version "
                             f"{data.get('format')!r}/{data.get('version')!r}")  # fmt: skip
        items = [Measurement.from_json(d) for d in data["measurements"]]
        problems = find_problems(items)
        if problems:
            raise ValueError("measurement set is inconsistent: " + "; ".join(problems))
        return cls(items)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> MeasurementSet:
        try:
            data = json.loads(Path(path).read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"measurement set: {path} is not valid JSON ({exc})") from None
        return cls.from_json(data)
