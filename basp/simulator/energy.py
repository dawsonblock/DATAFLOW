"""Energy accounting from resource usage.

Energy = static power x elapsed time + per-MAC, per-memory-byte and per-link-byte
dynamic energy.  All coefficients come from the topology profile and are
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

    @property
    def total_j(self) -> float:
        return self.static_j + self.compute_j + self.memory_j + self.link_j


def energy_report(topology: Topology, resources: Mapping[tuple, Resource], elapsed_s: float) -> EnergyReport:
    report = EnergyReport()
    for dev in topology.devices.values():
        static = dev.power.idle_w * elapsed_s
        compute = memory = 0.0
        res = resources.get(("compute", dev.device_id))
        if res is not None:
            compute = res.served * dev.power.joules_per_mac
        res = resources.get(("mem", dev.device_id))
        if res is not None:
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
