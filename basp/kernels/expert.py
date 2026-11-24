"""MoE expert execution on the streaming INT4 kernels.

An expert is a SwiGLU block ``W_down @ (silu(W_gate x) * (W_up x))`` with ``W_gate`` and
``W_up`` of shape ``[intermediate, hidden]`` and ``W_down`` of shape ``[hidden, intermediate]``.
Activations enter every matrix as FP16 (the kernels round them); the gate/up outputs, the
activation function and the weighted reduction are FP32.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .int4 import GROUP_SIZE, Int4Tensor, dequantize, quantize_int4
from .matvec import KernelStats, matmul_int4


@dataclass(frozen=True, eq=False)
class QuantizedExpert:
    """The three INT4 matrices of one expert."""

    gate: Int4Tensor
    up: Int4Tensor
    down: Int4Tensor

    def __post_init__(self) -> None:
        gate, up, down = self.gate, self.up, self.down
        if (gate.rows, gate.cols) != (up.rows, up.cols) or down.cols != gate.rows:
            raise ValueError(
                f"inconsistent expert shapes: gate [{gate.rows}, {gate.cols}], "
                f"up [{up.rows}, {up.cols}], down [{down.rows}, {down.cols}]")

    @classmethod
    def from_float(cls, w_gate, w_up, w_down, group_size: int = GROUP_SIZE) -> QuantizedExpert:
        return cls(quantize_int4(w_gate, group_size), quantize_int4(w_up, group_size),
                   quantize_int4(w_down, group_size))

    @property
    def nbytes(self) -> int:
        return self.gate.nbytes + self.up.nbytes + self.down.nbytes

    @property
    def n_weights(self) -> int:
        return self.gate.n_weights + self.up.n_weights + self.down.n_weights

    def dequantized(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(w_gate, w_up, w_down)`` expanded to float32."""
        return dequantize(self.gate), dequantize(self.up), dequantize(self.down)


def silu_fp32(x: np.ndarray) -> np.ndarray:
    """FP32 ``x * sigmoid(x)``, evaluated without overflow for large ``|x|``."""
    x = np.asarray(x, dtype=np.float32)
    return x * np.exp(-np.logaddexp(np.float32(0.0), -x))


def run_expert(expert: QuantizedExpert, x, *, tile_rows: int = 64,
               stats: KernelStats | None = None) -> np.ndarray:
    """Run one expert on ``x`` [hidden] or [batch, hidden] -> float32 of the matching shape."""
    x = np.asarray(x)
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be [hidden] or [batch, hidden], got shape {x.shape}")
    batch = x[None, :] if x.ndim == 1 else x
    gate = matmul_int4(expert.gate, batch, tile_rows=tile_rows, stats=stats)
    up = matmul_int4(expert.up, batch, tile_rows=tile_rows, stats=stats)
    out = matmul_int4(expert.down, silu_fp32(gate) * up, tile_rows=tile_rows, stats=stats)
    return out[0] if x.ndim == 1 else out


def moe_combine(outputs: Sequence, coeffs: Sequence[float]) -> np.ndarray:
    """Weighted reduction ``sum_i coeffs[i] * outputs[i]``, FP32 accumulation in list order."""
    if len(outputs) != len(coeffs) or len(outputs) == 0:
        raise ValueError("moe_combine needs one coefficient per output and at least one")
    acc = np.zeros(np.shape(outputs[0]), dtype=np.float32)
    for out, coeff in zip(outputs, coeffs):
        acc += np.float32(coeff) * np.asarray(out, dtype=np.float32)
    return acc
