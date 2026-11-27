"""Memory-channel arithmetic and the SRAM tile cache.

Channel bandwidth here is a *bandwidth model* (theoretical rate times a
sustained-efficiency factor).  Row-buffer behaviour, refresh and scheduling
effects need a cycle-level DRAM simulator such as Ramulator2, which is planned
as an external calibration source rather than re-implemented.
"""

from __future__ import annotations

from collections import OrderedDict


def ddr_channel_bandwidth(mega_transfers_per_s: float, bus_bytes: int = 8) -> float:
    """Theoretical bytes/s of one DDR channel (DDR5-5600 -> 44.8e9)."""
    return mega_transfers_per_s * 1e6 * bus_bytes


def ddr_bandwidth(mega_transfers_per_s: float, channels: int, efficiency: float = 1.0) -> float:
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    return ddr_channel_bandwidth(mega_transfers_per_s) * channels * efficiency


def stream_mac_rate(bandwidth_bytes_per_s: float, bits_per_weight: float) -> float:
    """MAC/s needed to consume a weight stream arriving at ``bandwidth`` (single token)."""
    return bandwidth_bytes_per_s * 8.0 / bits_per_weight


def lane_mac_rate(lanes: int, clock_hz: float) -> float:
    """MAC/s of an array with ``lanes`` MAC lanes, one MAC per lane per cycle."""
    return lanes * clock_hz


class TileCache:
    """Device-local SRAM holding decoded weight tiles, LRU by tensor.

    Models operand reuse for the row-stationary mode: a tensor whose tiles are
    all in SRAM is served without touching the external memory channels.
    """

    def __init__(self, capacity_bytes: int) -> None:
        self.capacity = int(capacity_bytes)
        self.used = 0
        self._entries: OrderedDict[int, int] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __contains__(self, tensor_id: int) -> bool:
        return tensor_id in self._entries

    def contains_all(self, tensor_ids) -> bool:
        return all(t in self._entries for t in tensor_ids)

    def touch(self, tensor_id: int) -> None:
        self._entries.move_to_end(tensor_id)

    def insert(self, tensor_id: int, nbytes: int) -> list[int]:
        """Insert (or refresh) a tensor; returns the tensors evicted to make room."""
        if nbytes > self.capacity:
            return []
        if tensor_id in self._entries:
            self._entries.move_to_end(tensor_id)
            return []
        evicted: list[int] = []
        while self.used + nbytes > self.capacity:
            victim, size = self._entries.popitem(last=False)
            self.used -= size
            self.evictions += 1
            evicted.append(victim)
        self._entries[tensor_id] = nbytes
        self.used += nbytes
        return evicted

    def discard(self, tensor_id: int) -> None:
        size = self._entries.pop(tensor_id, None)
        if size is not None:
            self.used -= size

    def tensors(self) -> tuple[int, ...]:
        return tuple(self._entries)
