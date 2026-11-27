"""Numerical verification against the independent float64 reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from basp.backends.cpu.executor import NumericPayload
from basp.workloads.tiny import TinyModel


@dataclass
class NumericReport:
    steps: int
    max_rel_l2: float  # worst relative L2 error of the logits over all steps
    routing_matches: bool  # the reference's own routing equals the executed routing
    tokens_match: bool  # arg-max tokens equal the reference's at every step
    tolerance: float

    @property
    def ok(self) -> bool:
        return self.max_rel_l2 <= self.tolerance and self.routing_matches and self.tokens_match


def compare_to_reference(model: TinyModel, payload: NumericPayload, *, tolerance: float = 1e-2) -> NumericReport:
    """Replay every executed step through the float64 reference and compare.

    The reference runs on dequantized weights with unrounded activations, so the
    remaining difference is the FP16 activation rounding and FP32 accumulation
    of the streaming kernels (about 5e-4 per expert stage).  Values are compared
    under the executed routing; whether the reference would have routed
    identically is reported separately.
    """
    worst = 0.0
    routing_ok = tokens_ok = True
    for step, logits in enumerate(payload.logits):
        expected, own = model.reference_step(payload.inputs[step], forced=payload.routing[step])
        err = np.linalg.norm(logits - expected) / max(np.linalg.norm(expected), 1e-30)
        worst = max(worst, float(err))
        tokens_ok &= bool(np.array_equal(np.argmax(expected, axis=1), payload.tokens[step]))
        for mine, theirs in zip(payload.routing[step], own):
            routing_ok &= [set(r) for r in mine.experts] == [set(r) for r in theirs.experts]
    return NumericReport(len(payload.logits), worst, routing_ok, tokens_ok, tolerance)
