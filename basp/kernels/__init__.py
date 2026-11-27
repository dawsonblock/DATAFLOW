"""Low-bit reference kernels.

``reference`` is the float64 golden model. ``int4`` defines the INT4 + FP16-group-scale
weight format (4.5 bits/weight), ``matvec`` the fused streaming decode->MAC kernels and
``expert`` MoE expert execution built on them.
"""

from . import reference
from .expert import (
    QuantizedExpert,
    moe_combine,
    run_expert,
    shard_bounds,
    silu_fp32,
    split_expert,
)
from .int4 import (
    GROUP_SIZE,
    Int4Tensor,
    dequantize,
    pack_nibbles,
    quantize_activations_int8,
    quantize_int4,
    symbol_entropy_bits,
    unpack_nibbles,
)
from .matvec import KernelStats, group_sums_int32, matmul_int4, matvec_int4, matvec_int4_int8

__all__ = [
    "GROUP_SIZE",
    "Int4Tensor",
    "KernelStats",
    "QuantizedExpert",
    "dequantize",
    "group_sums_int32",
    "matmul_int4",
    "matvec_int4",
    "matvec_int4_int8",
    "moe_combine",
    "pack_nibbles",
    "quantize_activations_int8",
    "quantize_int4",
    "reference",
    "run_expert",
    "shard_bounds",
    "silu_fp32",
    "split_expert",
    "symbol_entropy_bits",
    "unpack_nibbles",
]
