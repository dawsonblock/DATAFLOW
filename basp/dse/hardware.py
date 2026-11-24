"""Parametric hardware designs: topology, cost and power from a handful of choices.

A :class:`HardwareDesign` is a point in the architectural design space: how
much GPU memory, how many compute-memory modules with how many memory channels
and MAC lanes, how they are connected.  It is turned into a simulator topology
on top of a *base profile* (the host and GPU, ideally calibrated from
measurements) and a table of *estimates* for everything that does not exist yet.

Estimates are (pessimistic, nominal, optimistic) triples.  A design is evaluated
at one **corner**; decisions about hypothetical hardware should hold at the
pessimistic corner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from basp.core.capabilities import DeviceKind
from basp.core.topology import Topology
from basp.simulator.dram import ddr_channel_bandwidth

CORNERS = ("pessimistic", "nominal", "optimistic")
GiB = 1024**3


class DesignError(ValueError):
    """The design is physically or logically impossible."""


def pick(estimate: Any, corner: str) -> float:
    """Value of an estimate at a corner (plain numbers are corner-independent)."""
    if isinstance(estimate, Mapping):
        return float(estimate[corner])
    return float(estimate)


@dataclass(frozen=True)
class HardwareDesign:
    gpu_gib: int = 12  # 0: no GPU
    modules: int = 0
    module_gib: int = 32
    channels: int = 2  # DDR5 channels per module
    ddr_mt: int = 5600
    lanes: int = 1024  # MAC lanes per module
    link_gbps: float = 16.0  # module link bandwidth per direction, GB/s
    fabric: str = "star"  # star: modules on host PCIe; local: GPU and modules on a fast switch
    split_channels: bool = False  # run each memory channel as an independent engine
    persistent: bool = False  # resident kernels on GPU and modules (no launch, no gating)
    sram_mib: int = 16
    host_gib: int = 64

    def __post_init__(self) -> None:
        if self.fabric not in ("star", "local"):
            raise DesignError(f"unknown fabric {self.fabric!r}")
        if min(self.gpu_gib, self.modules) < 0 or self.modules and min(
            self.module_gib, self.channels, self.lanes, self.link_gbps
        ) <= 0:  # fmt: skip
            raise DesignError("design parameters must be positive")
        if self.modules and self.module_gib > 128 * self.channels:
            raise DesignError("at most 128 GiB per memory channel")
        if self.split_channels and self.channels < 2:
            raise DesignError("independent channel engines need at least two channels")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- derived quantities -------------------------------------------------------
    def pcie_lanes(self, estimates: Mapping[str, Any]) -> int:
        """Host PCIe lanes the design occupies."""
        per_lane = float(estimates["pcie_bytes_per_s_per_lane"])
        gpu = 16 if self.gpu_gib else 0
        if self.fabric == "local":
            return 16 if (self.modules or self.gpu_gib) else 0
        return gpu + self.modules * math.ceil(self.link_gbps * 1e9 / per_lane)

    def cost_usd(self, estimates: Mapping[str, Any], corner: str = "nominal") -> float:
        """Estimated system price.  A guess with wide error bars, useful only for ratios."""
        host, module, fabric = estimates["host"], estimates["module"], estimates["fabric"]
        total = pick(host["cost_base_usd"], corner) + self.host_gib * pick(host["cost_per_gib_usd"], corner)
        if self.pcie_lanes(estimates) > int(host["desktop_pcie_lanes"]):
            total += pick(host["workstation_surcharge_usd"], corner)
        if self.gpu_gib:
            table = estimates["gpu"]["cost_usd_by_gib"]
            sizes = sorted(int(k) for k in table)
            size = next((s for s in sizes if s >= self.gpu_gib), sizes[-1])
            total += pick(table[str(size)], corner)
            if self.gpu_gib > sizes[-1]:
                extra = pick(estimates["gpu"]["cost_per_gib_usd_above_24"], corner)
                total += (self.gpu_gib - sizes[-1]) * extra
        total += self.modules * (
            pick(module["cost_base_usd"], corner)
            + self.lanes * pick(module["cost_per_lane_usd"], corner)
            + self.module_gib * pick(module["cost_per_gib_usd"], corner)
            + self.link_gbps * pick(module["cost_per_link_gbps_usd"], corner)
        )
        if self.fabric == "local" and self.modules:
            ports = self.modules + (1 if self.gpu_gib else 0) + 1
            total += pick(fabric["cost_base_usd"], corner) + ports * pick(fabric["cost_per_port_usd"], corner)
        return total

    def module_idle_w(self, estimates: Mapping[str, Any], corner: str = "nominal") -> float:
        module = estimates["module"]
        return (pick(module["idle_w_base"], corner) + self.lanes * pick(module["idle_w_per_lane"], corner)
                + self.module_gib * pick(module["idle_w_per_gib"], corner))  # fmt: skip

    # -- topology -----------------------------------------------------------------
    def topology(self, base: Topology, estimates: Mapping[str, Any], corner: str = "nominal") -> Topology:
        """Simulator topology: ``base`` supplies the host and GPU, the design the rest."""
        if corner not in CORNERS:
            raise DesignError(f"unknown corner {corner!r}")
        data = base.to_dict()
        host_name = data["host"]
        devices = []
        gpu_name = None
        for dev in data["devices"]:
            if dev["kind"] == "CPU" and dev["name"] == host_name:
                devices.append({**dev, "mem_capacity": self.host_gib * GiB})
            elif dev["kind"] == "GPU" and self.gpu_gib and gpu_name is None:
                gpu_name = dev["name"]
                devices.append({**dev, "mem_capacity": self.gpu_gib * GiB,
                                "persistent": self.persistent or dev.get("persistent", False)})  # fmt: skip
        if len(devices) < (2 if self.gpu_gib else 1):
            raise DesignError("the base profile lacks the host or GPU this design needs")
        links = [k for k in data["links"] if {k["a"], k["b"]} == {host_name, gpu_name}]
        gpu_link = links[0] if links else None
        module, fabric = estimates["module"], estimates["fabric"]
        engines = self.channels if self.split_channels else 1
        bandwidth = (self.channels * ddr_channel_bandwidth(self.ddr_mt)
                     * pick(module["mem_efficiency"], corner))  # fmt: skip
        compute = self.lanes * pick(module["clock_hz"], corner)
        idle = self.module_idle_w(estimates, corner)
        hub_port = "fabric" if self.fabric == "local" and self.modules else host_name
        if hub_port == "fabric":
            devices.append({"name": "fabric", "kind": "SWITCH", "mem_capacity": 0,
                            "mem_bandwidth": 1e15, "opcodes": []})  # fmt: skip
            fab = {"bandwidth_ab": pick(fabric["bandwidth_bytes_per_s"], corner),
                   "latency_s": pick(fabric["latency_s"], corner),
                   "joules_per_byte": pick(fabric["joules_per_byte"], corner)}  # fmt: skip
            host_link = dict(gpu_link) if gpu_link else {
                "bandwidth_ab": self.link_gbps * 1e9, "latency_s": pick(module["link_latency_s"], corner)}  # fmt: skip
            links = [{**host_link, "name": "pcie-fabric", "a": host_name, "b": "fabric"}]
            if gpu_name:
                links.append({**fab, "name": "fabric-gpu", "a": "fabric", "b": gpu_name})
        for i in range(self.modules):
            name = f"mod{i}"
            spec = {
                "kind": "NEAR_MEMORY",
                "mem_capacity": self.module_gib * GiB // engines,
                "mem_bandwidth": bandwidth / engines,
                "compute_rate": compute / engines,
                "kernel_startup_s": pick(module["kernel_startup_s"], corner),
                "warm_startup_s": pick(module["warm_startup_s"], corner),
                "persistent": self.persistent,
                "sram_bytes": self.sram_mib * 1024 * 1024 // engines,
                "decode_rate": {"INT4": "inf", "INT8": "inf",
                                "ENTROPY": pick(module["entropy_decode_bytes_per_s"], corner) / engines},
                "stream_formats": ["INT4", "INT8", "ENTROPY"],
                "stream_ingest": True,
                "power": {"idle_w": idle / engines,
                          "joules_per_mac": pick(module["joules_per_mac"], corner),
                          "joules_per_mem_byte": pick(module["joules_per_mem_byte"], corner),
                          "sleep_w": pick(module["sleep_w"], corner) / engines,
                          "gate_after_s": 0.01, "wake_j": 0.05},
            }  # fmt: skip
            if hub_port == "fabric":
                link = {**fab, "name": f"fabric-{name}", "a": "fabric"}
            else:
                link = {"name": f"pcie-{name}", "a": host_name, "bandwidth_ab": self.link_gbps * 1e9,
                        "latency_s": pick(module["link_latency_s"], corner), "joules_per_byte": 6e-11}  # fmt: skip
            if engines == 1:
                devices.append({"name": name, **spec})
                links.append({**link, "b": name})
            else:
                # One link serves the module; each memory channel is its own engine behind it.
                devices.append({"name": name, "kind": "SWITCH", "mem_capacity": 0,
                                "mem_bandwidth": 1e15, "opcodes": []})  # fmt: skip
                links.append({**link, "b": name})
                for c in range(engines):
                    devices.append({"name": f"{name}.c{c}", **spec})
                    links.append({"name": f"{name}.c{c}", "a": name, "b": f"{name}.c{c}",
                                  "bandwidth_ab": 1e15, "latency_s": 0.0})  # fmt: skip
        data.update(devices=devices, links=links,
                    name=f"design:{self.fabric}:g{self.gpu_gib}:m{self.modules}")  # fmt: skip
        return Topology.from_dict(data)


def compute_devices(topology: Topology, kind: DeviceKind) -> list[int]:
    return [d.device_id for d in topology.of_kind(kind) if d.caps.can_compute]


def with_gpu_memory(design: HardwareDesign, gpu_gib: int) -> HardwareDesign:
    return replace(design, gpu_gib=gpu_gib)
