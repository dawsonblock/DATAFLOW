"""Closed-form dataflow comparison (section 6 of the architecture).

Single-token roofline model of one expert: moving the weights to a remote
processor versus executing next to the memory that holds them.  The assumptions
are those of the specification: one weight read, FP16 activations in and out,
ideal overlap of memory access, decoding and computation; no queueing, cache
reuse, protocol latency or remote execution time.  It is an analytical
illustration, not a measurement, and it is what the discrete-event simulator
must reproduce on an idle system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from basp.simulator.dram import ddr_bandwidth, lane_mac_rate, stream_mac_rate
from basp.units import KiB, MiB


@dataclass(frozen=True)
class PathComparison:
    params: int
    weight_bytes: float
    activation_bytes: int
    traffic_reduction: float
    cold_transfer_s: float  # weights -> remote processor, excluding its execution time
    near_memory_s: float  # execute in place, including activation transfer
    near_memory_bound: str  # which stage limits the near-memory kernel

    def as_rows(self) -> list[tuple[str, str]]:
        return [
            ("Expert parameters", f"{self.params / 1e6:.1f}M"),
            ("Weight transfer size", f"{self.weight_bytes / MiB:.1f} MiB"),
            ("Input/output activations", f"{self.activation_bytes / KiB:.1f} KiB"),
            ("Interconnect traffic reduction", f"{self.traffic_reduction:,.0f}x"),
            ("Cold weight transfer, excluding execution", f"{self.cold_transfer_s * 1e3:.3f} ms"),
            ("Near-memory execution, including activations", f"{self.near_memory_s * 1e3:.3f} ms"),
            ("Near-memory kernel limited by", self.near_memory_bound),
        ]


def compare_paths(
    hidden: int = 4096,
    intermediate: int = 14336,
    bits_per_weight: float = 4.5,
    link_bandwidth: float = 24e9,
    memory_bandwidth: float = 54e9,
    compute_rate: float = 128e9,
    *,
    matrices: int = 3,
    decode_rate: float = math.inf,
    activation_bits: int = 16,
) -> PathComparison:
    """Compare path A (move weights) with path B (move activations) for one expert."""
    params = matrices * hidden * intermediate
    weight_bytes = params * bits_per_weight / 8.0
    act_bytes = 2 * hidden * activation_bits // 8  # one input and one output vector
    stages = {
        "memory bandwidth": weight_bytes / memory_bandwidth,
        "compute throughput": params / compute_rate,
        "decode throughput": weight_bytes / decode_rate,
    }
    bound = max(stages, key=stages.get)
    return PathComparison(
        params=params,
        weight_bytes=weight_bytes,
        activation_bytes=act_bytes,
        traffic_reduction=weight_bytes / act_bytes,
        cold_transfer_s=weight_bytes / link_bandwidth,
        near_memory_s=stages[bound] + act_bytes / link_bandwidth,
        near_memory_bound=bound,
    )


@dataclass(frozen=True)
class ArithmeticSizing:
    memory_bandwidth: float  # bytes/s
    required_mac_rate: float  # MAC/s to consume the full weight stream
    lane_mac_rate: float  # MAC/s provided by the lane array
    headroom: float  # provided / required


def size_arithmetic(
    ddr_mt_per_s: float = 5600,
    channels: int = 2,
    bits_per_weight: float = 4.5,
    lanes: int = 1024,
    clock_hz: float = 250e6,
    efficiency: float = 1.0,
) -> ArithmeticSizing:
    """Section 10 sizing: can the MAC lanes keep up with the memory channels?"""
    bandwidth = ddr_bandwidth(ddr_mt_per_s, channels, efficiency)
    required = stream_mac_rate(bandwidth, bits_per_weight)
    provided = lane_mac_rate(lanes, clock_hz)
    return ArithmeticSizing(bandwidth, required, provided, provided / required)
