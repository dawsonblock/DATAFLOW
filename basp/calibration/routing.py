"""Turning recorded router logits into routing traces, and summarising real traces.

``trace_statistics`` reports the three properties the synthetic trace generator assumes
(popularity skew, stickiness, concentration), so a recorded trace can be compared with it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from basp.kernels import reference
from basp.workloads.trace import RoutingStep, RoutingTrace


def logits_to_routing(logits: np.ndarray, top_k: int, normalize: bool = True) -> RoutingStep:
    """Routing decision of every row of ``logits`` ``[rows, n_experts]``.

    Softmax over all logits, keep the ``top_k`` largest (ties go to the lower index); with
    ``normalize`` the kept coefficients are rescaled to sum to 1 (``norm_topk_prob``).
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[None, :]
    if logits.ndim != 2:
        raise ValueError(f"logits must be [rows, n_experts], got shape {logits.shape}")
    if not np.all(np.isfinite(logits)):
        raise ValueError("router logits contain non-finite values")
    indices, coeffs = reference.route_topk(logits, top_k, normalize=normalize)
    return RoutingStep(
        tuple(tuple(int(e) for e in row) for row in indices),
        tuple(tuple(float(c) for c in row) for row in coeffs),
    )


def build_trace(per_step_logits: Sequence[Sequence[np.ndarray]], top_k: int) -> RoutingTrace:
    """Trace from router logits indexed ``[step][layer] -> [rows, n_experts]``.

    Every step must have the same number of layers and every logits array the same shape.
    """
    if len(per_step_logits) == 0:
        raise ValueError("build_trace needs at least one step")
    n_layers = len(per_step_logits[0])
    if n_layers == 0:
        raise ValueError("build_trace: a step has no layers")
    first = np.asarray(per_step_logits[0][0])
    shape = (1, first.shape[0]) if first.ndim == 1 else first.shape
    steps: list[list[RoutingStep]] = []
    for s, layers in enumerate(per_step_logits):
        if len(layers) != n_layers:
            raise ValueError(f"step {s} has {len(layers)} layers, expected {n_layers}")
        row: list[RoutingStep] = []
        for layer, logits in enumerate(layers):
            arr = np.asarray(logits)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape != shape:
                raise ValueError(f"step {s} layer {layer}: logits shape {arr.shape} differs from "
                                 f"{shape} (all steps need the same rows and experts)")  # fmt: skip
            row.append(logits_to_routing(arr, top_k))
        steps.append(row)
    trace = RoutingTrace(n_layers, int(shape[1]), top_k, int(shape[0]), steps)
    trace.validate()
    return trace


def trace_statistics(trace: RoutingTrace) -> dict[str, Any]:
    """Popularity entropy, stickiness and concentration of a routing trace.

    * ``usage_entropy_bits`` -- per layer, the entropy of the distribution of selections over
      experts (``log2(n_experts)`` when usage is uniform; lower means more skew);
    * ``stickiness`` -- probability that an expert selected by a row at step ``t`` is selected by
      the same row at ``t + 1`` (``None`` for traces with fewer than two steps).  Note that the
      synthetic generator's ``stickiness`` parameter is a per-slot *keep* probability, which is
      a lower bound of this quantity;
    * ``top_decile_share`` -- share of all selections going to the most-used tenth of the
      experts of each layer, averaged over layers (0.1-ish when usage is uniform).
    """
    freq = trace.expert_frequencies()  # [layers, experts], rows sum to top_k
    entropies: list[float] = []
    decile_shares: list[float] = []
    n_top = max(1, math.ceil(trace.n_experts / 10))
    for layer in range(trace.n_layers):
        total = freq[layer].sum()
        if total <= 0:
            entropies.append(0.0)
            decile_shares.append(0.0)
            continue
        p = freq[layer] / total
        nz = p[p > 0]
        entropies.append(float(-(nz * np.log2(nz)).sum()))
        decile_shares.append(float(np.sort(p)[::-1][:n_top].sum()))

    stickiness: float | None = None
    if trace.n_steps >= 2:
        kept = 0
        for step in range(trace.n_steps - 1):
            for layer in range(trace.n_layers):
                now, nxt = trace.at(step, layer).experts, trace.at(step + 1, layer).experts
                kept += sum(len(set(a) & set(b)) for a, b in zip(now, nxt))
        possible = (trace.n_steps - 1) * trace.n_layers * trace.batch * trace.top_k
        stickiness = kept / possible

    return {
        "n_steps": trace.n_steps,
        "n_layers": trace.n_layers,
        "n_experts": trace.n_experts,
        "top_k": trace.top_k,
        "batch": trace.batch,
        "usage_entropy_bits": entropies,
        "usage_entropy_bits_mean": float(np.mean(entropies)) if entropies else 0.0,
        "usage_entropy_bits_max": math.log2(trace.n_experts),
        "stickiness": stickiness,
        "top_decile_share": float(np.mean(decile_shares)) if decile_shares else 0.0,
        "top_decile_share_per_layer": decile_shares,
    }
