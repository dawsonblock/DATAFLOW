"""Energy accounting from resource usage.

Energy = static power x elapsed time + per-MAC, per-memory-byte and per-link-byte
dynamic energy.  A device that declares a sleep power spends idle periods longer
than its gating threshold in that state and pays a wake-up energy; the wake-up
latency is not modelled (the device is assumed to be woken ahead of need), so
gated energy figures are optimistic.  All coefficients come from the topology profile and are
modelling assumptions until calibrated against measurements.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from basp.core.topology import Topology
from basp.simulator.fluid import Resource


@dataclass
class EnergyReport:
    static_j: float = 0.0
    compute_j: float = 0.0
    memory_j: float = 0.0
    link_j: float = 0.0
    per_device_j: dict[str, float] = field(default_factory=dict)
    gated_s: dict[str, float] = field(default_factory=dict)  # time each device spent gated

    @property
    def total_j(self) -> float:
        return self.static_j + self.compute_j + self.memory_j + self.link_j


def energy_report(
    topology: Topology,
    resources: Mapping[tuple, Resource],
    elapsed_s: float,
    gating: Mapping[int, tuple[float, int]] | None = None,
) -> EnergyReport:
    """Energy over ``elapsed_s``.  ``gating`` maps a device to (seconds gated, wake-ups)."""
    report = EnergyReport()
    for dev in topology.devices.values():
        gated_s, wakes = (gating or {}).get(dev.device_id, (0.0, 0))
        gated_s = min(gated_s, elapsed_s) if dev.power.sleep_w is not None else 0.0
        static = dev.power.idle_w * (elapsed_s - gated_s)
        if gated_s:
            static += dev.power.sleep_w * gated_s + dev.power.wake_j * wakes
            report.gated_s[dev.name] = gated_s
        compute = memory = 0.0
        res = resources.get(("compute", dev.device_id))
        if res is not None:
            compute = res.served * dev.power.joules_per_mac
        res = resources.get(("mem", dev.device_id))
        # a shared (unified) memory resource is counted once, against the domain's first device
        if res is not None and topology.memory_owner(dev.device_id) == dev.device_id:
            memory = res.served * dev.power.joules_per_mem_byte
        report.static_j += static
        report.compute_j += compute
        report.memory_j += memory
        report.per_device_j[dev.name] = static + compute + memory
    for i, link in enumerate(topology.links):
        for forward in (True, False):
            res = resources.get(("link", i, forward))
            if res is not None:
                report.link_j += res.served * link.joules_per_byte
    return report
