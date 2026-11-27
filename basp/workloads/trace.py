"""MoE routing traces: which experts each token stream selects at each layer.

A :class:`RoutingTrace` records, for every decode step and layer, the ``top_k`` distinct experts
(and their routing coefficients) chosen by each of ``batch`` independent token streams. Traces
serialise to plain JSON, so router decisions recorded from a real model can replace synthetic ones.

:func:`synth_trace` is a SYNTHETIC generator, not measured router behaviour. Its model:

* every layer has its own random popularity permutation with a Zipf(``zipf_s``) base distribution
  over popularity ranks (``zipf_s=0`` is uniform);
* every stream has ``top_k`` slots per layer. At each step a slot keeps its previous expert with
  probability ``stickiness``; the remaining slots are resampled from the base distribution without
  replacement, excluding the kept experts, so a selection is always ``top_k`` distinct experts;
* coefficients are i.i.d. exponential draws, sorted descending and normalised to sum to 1.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_FORMAT = "basp.routing_trace"
_VERSION = 1
_COEFF_TOL = 1e-6


@dataclass(frozen=True)
class RoutingStep:
    """Routing decision of all batch rows at one (step, layer)."""

    experts: tuple[tuple[int, ...], ...]
    coeffs: tuple[tuple[float, ...], ...]

    def unique_experts(self) -> tuple[int, ...]:
        return tuple(sorted({e for row in self.experts for e in row}))

    def rows_for(self, expert: int) -> tuple[int, ...]:
        return tuple(i for i, row in enumerate(self.experts) if expert in row)


@dataclass
class RoutingTrace:
    """Routing decisions indexed ``steps[step][layer]``."""

    n_layers: int
    n_experts: int
    top_k: int
    batch: int
    steps: list[list[RoutingStep]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if min(self.n_layers, self.n_experts, self.top_k, self.batch) < 1:
            raise ValueError("routing trace: n_layers, n_experts, top_k and batch must be >= 1")
        if self.top_k > self.n_experts:
            raise ValueError(f"routing trace: top_k={self.top_k} exceeds n_experts={self.n_experts}")

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def at(self, step: int, layer: int) -> RoutingStep:
        return self.steps[step][layer]

    def expert_frequencies(self) -> np.ndarray:
        """``[n_layers, n_experts]`` fraction of (step, row) visits that selected each expert."""
        freq = np.zeros((self.n_layers, self.n_experts), dtype=np.float64)
        if not self.steps:
            return freq
        ids = np.array([[st.experts for st in layers] for layers in self.steps], dtype=np.intp)
        for layer in range(self.n_layers):
            freq[layer] = np.bincount(ids[:, layer].ravel(), minlength=self.n_experts)
        return freq / (self.n_steps * self.batch)

    def split(self, n_steps_first: int) -> tuple[RoutingTrace, RoutingTrace]:
        """Split into the first ``n_steps_first`` steps and the rest (train / held-out)."""
        if not 0 <= n_steps_first <= self.n_steps:
            raise ValueError(f"split point {n_steps_first} outside [0, {self.n_steps}]")
        head = RoutingTrace(self.n_layers, self.n_experts, self.top_k, self.batch,
                            self.steps[:n_steps_first])
        tail = RoutingTrace(self.n_layers, self.n_experts, self.top_k, self.batch,
                            self.steps[n_steps_first:])
        return head, tail

    def validate(self) -> None:
        """Check shapes, id ranges, distinctness and coefficient normalisation."""
        for s, layers in enumerate(self.steps):
            if len(layers) != self.n_layers:
                raise ValueError(f"step {s}: expected {self.n_layers} layers, got {len(layers)}")
            for layer, st in enumerate(layers):
                where = f"step {s} layer {layer}"
                if len(st.experts) != self.batch or len(st.coeffs) != self.batch:
                    raise ValueError(f"{where}: expected {self.batch} batch rows of experts/coeffs")
                for row, (ids, co) in enumerate(zip(st.experts, st.coeffs)):
                    if len(ids) != self.top_k or len(co) != self.top_k:
                        raise ValueError(f"{where} row {row}: expected top_k={self.top_k} entries")
                    if not all(isinstance(e, numbers.Integral) for e in ids):
                        raise ValueError(f"{where} row {row}: expert ids must be integers")
                    if min(ids) < 0 or max(ids) >= self.n_experts:
                        raise ValueError(
                            f"{where} row {row}: expert id outside [0, {self.n_experts})")
                    if len(set(ids)) != self.top_k:
                        raise ValueError(f"{where} row {row}: duplicate expert ids {ids}")
                    if not all(isinstance(c, numbers.Real) and math.isfinite(c) for c in co):
                        raise ValueError(f"{where} row {row}: coefficients must be finite numbers")
                    if min(co) < 0 or abs(sum(co) - 1.0) > _COEFF_TOL:
                        raise ValueError(
                            f"{where} row {row}: coefficients must be >= 0 and sum to 1")

    def to_json(self) -> dict:
        return {
            "format": _FORMAT,
            "version": _VERSION,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "batch": self.batch,
            "steps": [[{"experts": [list(r) for r in st.experts],
                        "coeffs": [list(r) for r in st.coeffs]} for st in layers]
                      for layers in self.steps],
        }

    @classmethod
    def from_json(cls, d: dict) -> RoutingTrace:
        try:
            header = (d["n_layers"], d["n_experts"], d["top_k"], d["batch"])
            raw = d["steps"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"routing trace: missing field {exc}") from None
        if d.get("format", _FORMAT) != _FORMAT or d.get("version", _VERSION) != _VERSION:
            raise ValueError("routing trace: unsupported format/version "
                             f"{d.get('format')!r}/{d.get('version')!r}")
        if not all(isinstance(h, int) and not isinstance(h, bool) for h in header):
            raise ValueError("routing trace: n_layers, n_experts, top_k, batch must be integers")
        try:
            steps = [[RoutingStep(tuple(tuple(r) for r in st["experts"]),
                                  tuple(tuple(float(c) for c in r) for r in st["coeffs"]))
                      for st in layers] for layers in raw]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"routing trace: malformed steps ({exc})") from None
        trace = cls(*header, steps)
        trace.validate()
        return trace

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), separators=(",", ":")))

    @classmethod
    def load(cls, path: str | Path) -> RoutingTrace:
        try:
            data = json.loads(Path(path).read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"routing trace: {path} is not valid JSON ({exc})") from None
        return cls.from_json(data)


def synth_trace(n_steps: int, n_layers: int, n_experts: int, top_k: int, *, batch: int = 1,
                zipf_s: float = 1.0, stickiness: float = 0.5, seed: int = 0) -> RoutingTrace:
    """Seeded synthetic routing trace (see module docstring for the generative model)."""
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if not 0.0 <= stickiness <= 1.0:
        raise ValueError("stickiness must be within [0, 1]")
    if zipf_s < 0:
        raise ValueError("zipf_s must be >= 0")
    trace = RoutingTrace(n_layers, n_experts, top_k, batch)

    rng = np.random.default_rng(seed)
    base = np.arange(1, n_experts + 1, dtype=np.float64) ** -float(zipf_s)
    log_base = np.log(base / base.sum())
    logp = np.empty((n_layers, 1, n_experts))
    for layer in range(n_layers):
        logp[layer, 0, rng.permutation(n_experts)] = log_base

    shape = (n_layers, batch, top_k)
    prev: np.ndarray | None = None
    for _ in range(n_steps):
        # Gumbel-top-k: ordering experts by log p + Gumbel noise is successive sampling without
        # replacement from p; masking the kept experts conditions on them being excluded.
        keys = logp + rng.gumbel(size=(n_layers, batch, n_experts))
        if prev is None:
            cur = np.argsort(-keys, axis=-1)[..., :top_k]
        else:
            keep = rng.random(shape) < stickiness
            li, bi, ki = np.nonzero(keep)
            keys[li, bi, prev[li, bi, ki]] = -np.inf
            fresh = np.argsort(-keys, axis=-1)[..., :top_k]
            rank = np.maximum(np.cumsum(~keep, axis=-1) - 1, 0)
            cur = np.where(keep, prev, np.take_along_axis(fresh, rank, axis=-1))
        co = -np.sort(-(rng.exponential(size=shape) + 1e-12), axis=-1)
        co /= co.sum(axis=-1, keepdims=True)
        ids_l, co_l = cur.tolist(), co.tolist()
        trace.steps.append([RoutingStep(tuple(map(tuple, ids_l[layer])),
                                        tuple(map(tuple, co_l[layer])))
                            for layer in range(n_layers)])
        prev = cur
    return trace


def cyclic_trace(n_steps: int, n_layers: int, n_experts: int, top_k: int, *, period: int = 4,
                 batch: int = 1, seed: int = 0) -> RoutingTrace:
    """Constructed trace with purely *transitional* structure (synthetic best case).

    Every layer cycles deterministically through ``period`` disjoint expert sets,
    so the selection changes completely at each step: recency carries no
    information, while first-order transitions predict the next selection
    exactly.  Real routers lie somewhere between this and :func:`synth_trace`.
    """
    if period * top_k > n_experts:
        raise ValueError("period * top_k must not exceed n_experts")
    rng = np.random.default_rng(seed)
    coeff = tuple(float(c) for c in np.full(top_k, 1.0 / top_k))
    sets = []
    for _ in range(n_layers):
        perm = rng.permutation(n_experts)
        sets.append([tuple(sorted(int(e) for e in perm[i * top_k:(i + 1) * top_k]))
                     for i in range(period)])
    offsets = rng.integers(0, period, size=(n_layers, batch))
    steps = [
        [
            RoutingStep(
                tuple(sets[layer][(step + int(offsets[layer][row])) % period] for row in range(batch)),
                tuple(coeff for _ in range(batch)),
            )
            for layer in range(n_layers)
        ]
        for step in range(n_steps)
    ]
    return RoutingTrace(n_layers, n_experts, top_k, batch, steps)
