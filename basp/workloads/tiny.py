"""A small attention-free MoE language model with real (random) weights.

Used for numerical co-simulation: the same model is evaluated (a) by an
independent float64 reference on dequantized weights and (b) by the INT4
streaming kernels following the plans chosen by the scheduler.  Each layer is

    h      = rms_norm(x)
    logits = R h                         (router)
    y      = x + sum_i c_i * E_i(h)      (top-k SwiGLU experts, c = softmax over top-k)

followed by an output projection; the next input is the embedding of the
arg-max token, so decoding is genuinely autoregressive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from basp.kernels import Int4Tensor, QuantizedExpert, dequantize, quantize_int4, reference
from basp.workloads.model import MoEModelSpec
from basp.workloads.trace import RoutingStep


@dataclass
class TinyModel:
    spec: MoEModelSpec
    norm: np.ndarray  # [layers, hidden] float32
    router: np.ndarray  # [layers, experts, hidden] float32
    experts: list[list[QuantizedExpert]]
    head: Int4Tensor  # [vocab, hidden]
    embed: np.ndarray  # [vocab, hidden] float32

    @classmethod
    def random(cls, spec: MoEModelSpec, seed: int = 0) -> TinyModel:
        if spec.n_heads:
            raise ValueError("the numeric tiny model has no attention")
        rng = np.random.default_rng(seed)
        h, m = spec.hidden, spec.expert_inter
        scale_in, scale_mid = 1.0 / np.sqrt(h), 1.0 / np.sqrt(m)
        experts = [
            [
                QuantizedExpert.from_float(
                    rng.standard_normal((m, h)) * scale_in,
                    rng.standard_normal((m, h)) * scale_in,
                    rng.standard_normal((h, m)) * scale_mid,
                )
                for _ in range(spec.n_experts)
            ]
            for _ in range(spec.n_layers)
        ]
        return cls(
            spec=spec,
            norm=(1.0 + 0.1 * rng.standard_normal((spec.n_layers, h))).astype(np.float32),
            router=(rng.standard_normal((spec.n_layers, spec.n_experts, h)) * scale_in).astype(np.float32),
            experts=experts,
            head=quantize_int4(rng.standard_normal((spec.vocab, h)) * scale_in),
            embed=rng.standard_normal((spec.vocab, h)).astype(np.float32),
        )

    def initial_input(self, batch: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed + 10_007)
        tokens = rng.integers(0, self.spec.vocab, size=batch)
        return self.embed[tokens].copy()

    def reference_step(self, x: np.ndarray, forced: list[RoutingStep] | None = None):
        """One decoding step in float64 on dequantized weights.

        Returns ``(logits [batch, vocab], routing per layer)``.  With ``forced``
        the given routing decisions are used instead of the reference's own, so
        values can be compared even if a near-tie resolved differently.
        """
        spec = self.spec
        x = np.asarray(x, dtype=np.float64)
        routing: list[RoutingStep] = []
        for layer in range(spec.n_layers):
            h = reference.rms_norm(x, self.norm[layer])
            logits = reference.matmul(self.router[layer], h)
            idx, coeffs = reference.route_topk(logits, spec.top_k)
            own = RoutingStep(
                tuple(tuple(int(e) for e in row) for row in idx),
                tuple(tuple(float(c) for c in row) for row in coeffs),
            )
            step = forced[layer] if forced is not None else own
            routing.append(own)
            y = x.copy()
            for row in range(x.shape[0]):
                for expert, coeff in zip(step.experts[row], step.coeffs[row]):
                    gate, up, down = self.experts[layer][expert].dequantized()
                    y[row] += coeff * reference.swiglu_expert(h[row], gate, up, down)
            x = y
        return reference.matmul(dequantize(self.head), x), routing
