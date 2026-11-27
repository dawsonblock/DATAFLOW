"""Lightweight statistical expert predictor (spec section 11).

Predicts which experts a token stream will select at its next visit to a layer. It predicts data
needs only and never influences routing. Three signals, all maintained online with exponential
decay, are combined into one probability per expert:

* frequency ``f_e``: decayed rate at which expert ``e`` is selected per visit by one stream;
* recency: whether ``e`` was in the stream's last selection ``S`` at this layer. With a per-layer
  stickiness ``s`` fitted online by least squares on observed repeats, the estimate is
  ``f_e + s * (1[e in S] - f_e)``; ``s = 0`` reduces to frequency;
* transitions: first-order rates ``T[a -> b] = P(b selected next | a selected now)`` with additive
  smoothing towards ``f_b``. The cross-expert excess
  ``x_b = sum_{a in S, a != b} (T[a -> b] - f_b)`` is added with a per-layer weight ``w`` that is
  also fitted online by least squares on the remaining residual, so the term vanishes when
  transitions carry no information beyond recency. (With recency disabled the ``a = b`` repeat
  term is included in ``x_b``, so transitions alone can still express repeats.)

Both fitted coefficients and the final probability are clipped to [0, 1]. A stream with no history
at a layer falls back to frequency.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from basp.workloads.trace import RoutingStep, RoutingTrace

_RIDGE = 1.0  # shrinks the fitted coefficients towards 0 until evidence accumulates


class ExpertPredictor:
    def __init__(self, n_layers: int, n_experts: int, *, decay: float = 0.98,
                 use_frequency: bool = True, use_recency: bool = True,
                 use_transitions: bool = True, smoothing: float = 8.0) -> None:
        if n_layers < 1 or n_experts < 1:
            raise ValueError("n_layers and n_experts must be >= 1")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be within (0, 1]")
        if smoothing <= 0:
            raise ValueError("smoothing must be > 0")
        self.n_layers, self.n_experts = n_layers, n_experts
        self.decay, self.smoothing = float(decay), float(smoothing)
        self.use_frequency = use_frequency
        self.use_recency = use_recency
        self.use_transitions = use_transitions
        self._freq = np.zeros((n_layers, n_experts))
        self._visits = np.zeros(n_layers)
        # Decayed least-squares sums per layer: [x1.r, x1.x1, x2.r, x2.x2].
        self._fit = np.zeros((n_layers, 4))
        self._tc = np.zeros((n_layers, n_experts, n_experts), np.float32) if use_transitions \
            else None
        self._tn = np.zeros((n_layers, n_experts)) if use_transitions else None
        self._last: list[dict[int, np.ndarray]] = [{} for _ in range(n_layers)]
        self._known: set[int] = set()
        self._cache: list[dict] = [{} for _ in range(n_layers)]

    # ---- learning -------------------------------------------------------------------------

    def observe(self, layer: int, experts: Iterable[int], stream: int = 0) -> None:
        """Record one routing decision of ``stream`` at ``layer``."""
        idx = np.array(sorted(set(experts)), dtype=np.intp)
        if idx.size == 0 or idx[0] < 0 or idx[-1] >= self.n_experts:
            raise ValueError(f"experts must be non-empty ids within [0, {self.n_experts})")
        d = self.decay
        prev = self._last[layer].get(stream)
        if prev is not None:
            f, x1, x2 = self._signals(layer, stream)
            fit = self._fit[layer]
            resid = -f
            resid[idx] += 1.0
            if x1 is not None:
                s = self._coeffs(layer)[0]
                fit[0] = fit[0] * d + x1 @ resid
                fit[1] = fit[1] * d + x1 @ x1
                resid -= s * x1
            if x2 is not None:
                fit[2] = fit[2] * d + x2 @ resid
                fit[3] = fit[3] * d + x2 @ x2
                tc, tn = self._tc[layer], self._tn[layer]
                if d != 1.0:
                    tc[prev] *= d
                tc[prev[:, None], idx] += 1.0
                tn[prev] = tn[prev] * d + 1.0
        freq = self._freq[layer]
        if d != 1.0:
            freq *= d
        freq[idx] += 1.0
        self._visits[layer] = self._visits[layer] * d + 1.0
        self._last[layer][stream] = idx
        self._known.add(stream)
        self._cache[layer].clear()

    def observe_step(self, layer: int, step: RoutingStep) -> None:
        for row, ids in enumerate(step.experts):
            self.observe(layer, ids, stream=row)

    def fit(self, trace: RoutingTrace) -> None:
        self._check(trace)
        for layers in trace.steps:
            for layer, step in enumerate(layers):
                self.observe_step(layer, step)

    def reset_streams(self) -> None:
        """Forget every stream's last selections; keep the learned statistics and stream ids."""
        for layer in range(self.n_layers):
            self._last[layer].clear()
            self._cache[layer].clear()

    # ---- queries --------------------------------------------------------------------------

    def frequency(self, layer: int, expert: int) -> float:
        visits = self._visits[layer]
        return float(self._freq[layer, expert] / visits) if visits > 0 else 0.0

    def probability(self, layer: int, expert: int, stream: int = 0) -> float:
        return float(self._probs(layer, stream)[expert])

    def probability_any(self, layer: int, expert: int) -> float:
        return float(self._probs_any(layer)[expert])

    def predict(self, layer: int, n: int, stream: int | None = None) -> list[tuple[int, float]]:
        p = self._probs_any(layer) if stream is None else self._probs(layer, stream)
        order = np.argsort(-p, kind="stable")[:max(n, 0)]
        return list(zip(order.tolist(), p[order].tolist()))

    def evaluate(self, trace: RoutingTrace, n: int) -> dict[str, float]:
        """Online evaluation: predict ``n`` experts per stream, score, then observe."""
        self._check(trace)
        hits = selected = predicted = full = visits = 0
        for layers in trace.steps:
            for layer, step in enumerate(layers):
                for row, ids in enumerate(step.experts):
                    guess = {e for e, _ in self.predict(layer, n, stream=row)}
                    h = len(guess.intersection(ids))
                    hits += h
                    selected += len(ids)
                    predicted += len(guess)
                    full += h == len(ids)
                    visits += 1
                self.observe_step(layer, step)
        return {"hit_rate": hits / selected if selected else 0.0,
                "precision": hits / predicted if predicted else 0.0,
                "coverage_full": full / visits if visits else 0.0}

    # ---- internals ------------------------------------------------------------------------

    def _check(self, trace: RoutingTrace) -> None:
        if trace.n_layers != self.n_layers or trace.n_experts != self.n_experts:
            raise ValueError("trace dimensions do not match the predictor")

    def _prior(self, layer: int) -> np.ndarray:
        cache = self._cache[layer]
        f = cache.get("prior")
        if f is None:
            visits = self._visits[layer]
            f = self._freq[layer] / visits if visits > 0 else np.zeros(self.n_experts)
            if not self.use_frequency:
                f = np.full(self.n_experts, f.mean())
            cache["prior"] = f
        return f

    def _coeffs(self, layer: int) -> tuple[float, float]:
        a1, b1, a2, b2 = self._fit[layer]
        s = min(max(float(a1 / (b1 + _RIDGE)), 0.0), 1.0) if self.use_recency else 0.0
        w = min(max(float(a2 / (b2 + _RIDGE)), 0.0), 1.0) if self.use_transitions else 0.0
        return s, w

    def _signals(self, layer: int, stream: int):
        """``(f, x1, x2)``: prior, recency regressor and transition excess (None if unused)."""
        cache = self._cache[layer]
        sig = cache.get(stream)
        if sig is None:
            f = self._prior(layer)
            last = self._last[layer].get(stream)
            x1 = x2 = None
            if last is not None:
                if self.use_recency:
                    x1 = -f
                    x1[last] += 1.0
                if self.use_transitions:
                    # sum_a (C[a, :] - n_a f) / (n_a + alpha), as one small matvec.
                    tc, n = self._tc[layer], self._tn[layer][last]
                    wts = 1.0 / (n + self.smoothing)
                    x2 = wts @ tc[last] - (n @ wts) * f
                    if self.use_recency:  # recency owns the a -> a term; keep the signals apart
                        x2[last] -= (tc[last, last] - n * f[last]) * wts
            sig = cache[stream] = (f, x1, x2)
        return sig

    def _probs(self, layer: int, stream: int) -> np.ndarray:
        f, x1, x2 = self._signals(layer, stream)
        s, w = self._coeffs(layer)
        p = f
        if x1 is not None and s:
            p = p + s * x1
        if x2 is not None and w:
            p = p + w * x2
        return p.clip(0.0, 1.0)

    def _probs_any(self, layer: int) -> np.ndarray:
        cache = self._cache[layer]
        p = cache.get("any")
        if p is None:
            miss = np.ones(self.n_experts)
            cold = 0
            # Before any stream is known, behave as a single anonymous stream (frequency).
            for stream in sorted(self._known) or (0,):
                if stream in self._last[layer]:
                    miss *= 1.0 - self._probs(layer, stream)
                else:
                    cold += 1
            if cold:
                miss *= (1.0 - self._prior(layer)) ** cold
            p = cache["any"] = 1.0 - miss
        return p
