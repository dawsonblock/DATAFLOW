"""Physical topology: devices, bidirectional links and routing.

All rates are sustainable figures (bytes/s, MAC/s), not peak specifications;
see ``configs/`` for the profiles used by the experiments.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from basp.core.capabilities import ALL_OPCODES, NO_COMPUTE, DeviceCapabilities, DeviceKind
from basp.core.graph import OpCode
from basp.core.tensor import CompressionFormat


@dataclass(frozen=True)
class PowerSpec:
    """Energy model parameters; every value is a modelling assumption."""

    idle_w: float = 0.0
    joules_per_mac: float = 0.0
    joules_per_mem_byte: float = 0.0
    #: power drawn while gated; ``None`` means the device cannot be power-gated
    sleep_w: float | None = None
    #: an idle period longer than this is spent gated (beyond the threshold itself)
    gate_after_s: float = 1e-3
    wake_j: float = 0.0  # energy of one wake-up


@dataclass(frozen=True)
class DeviceSpec:
    device_id: int
    name: str
    kind: DeviceKind
    mem_capacity: int
    mem_bandwidth: float
    compute_rate: float = 0.0
    caps: DeviceCapabilities = NO_COMPUTE
    sram_bytes: int = 0
    kernel_startup_s: float = 0.0
    #: start-up of a kernel that begins the instant the previous one ended (batched launch);
    #: ``None`` means every kernel pays the full start-up
    warm_startup_s: float | None = None
    #: kernels stay resident (persistent operators): every start is a warm start
    persistent: bool = False
    queue_depth: int = 64
    power: PowerSpec = field(default_factory=PowerSpec)

    def __post_init__(self) -> None:
        if self.mem_bandwidth <= 0 or self.mem_capacity < 0:
            raise ValueError(f"device {self.name}: invalid memory parameters")
        if self.caps.can_compute and self.compute_rate <= 0:
            raise ValueError(f"device {self.name}: compute device needs a compute rate")


@dataclass(frozen=True)
class LinkSpec:
    """Bidirectional link with independent bandwidth per direction."""

    a: int
    b: int
    bandwidth_ab: float
    bandwidth_ba: float | None = None
    latency_s: float = 0.0
    dma_queue_depth: int = 8
    joules_per_byte: float = 0.0
    name: str = ""

    def bandwidth(self, forward: bool) -> float:
        if forward or self.bandwidth_ba is None:
            return self.bandwidth_ab
        return self.bandwidth_ba


@dataclass(frozen=True)
class Hop:
    link: int  # index into Topology.links
    forward: bool  # True when travelling a -> b


class Topology:
    def __init__(
        self,
        devices: list[DeviceSpec],
        links: list[LinkSpec],
        *,
        host: int | None = None,
        sync_latency_s: float = 0.0,
        name: str = "",
    ) -> None:
        self.devices: dict[int, DeviceSpec] = {}
        for dev in devices:
            if dev.device_id in self.devices:
                raise ValueError(f"duplicate device id {dev.device_id}")
            self.devices[dev.device_id] = dev
        self.links = list(links)
        for link in self.links:
            if link.a not in self.devices or link.b not in self.devices or link.a == link.b:
                raise ValueError(f"link {link.name!r} has invalid endpoints")
            if link.bandwidth_ab <= 0 or (link.bandwidth_ba is not None and link.bandwidth_ba <= 0):
                raise ValueError(f"link {link.name!r} needs positive bandwidth")
        self.host = host if host is not None else min(self.devices)
        self.sync_latency_s = sync_latency_s
        self.name = name
        self._adj: dict[int, list[tuple[int, Hop]]] = {d: [] for d in self.devices}
        for i, link in enumerate(self.links):
            self._adj[link.a].append((link.b, Hop(i, True)))
            self._adj[link.b].append((link.a, Hop(i, False)))
        self._routes: dict[tuple[int, int], tuple[Hop, ...]] = {}
        self._rates: dict[tuple[int, int], float] = {}

    # -- lookup -------------------------------------------------------------------
    def device(self, device_id: int) -> DeviceSpec:
        return self.devices[device_id]

    def by_name(self, name: str) -> DeviceSpec:
        for dev in self.devices.values():
            if dev.name == name:
                return dev
        raise KeyError(name)

    def of_kind(self, kind: DeviceKind) -> list[DeviceSpec]:
        return [d for d in self.devices.values() if d.kind == kind]

    def compute_devices(self) -> list[DeviceSpec]:
        return [d for d in self.devices.values() if d.caps.can_compute]

    # -- routing ------------------------------------------------------------------
    def route(self, src: int, dst: int) -> tuple[Hop, ...]:
        """Lowest-latency path (then fewest hops, then lowest link indices)."""
        if src == dst:
            return ()
        key = (src, dst)
        if key not in self._routes:
            self._routes[key] = self._dijkstra(src, dst)
        return self._routes[key]

    def _dijkstra(self, src: int, dst: int) -> tuple[Hop, ...]:
        heap: list[tuple[float, int, tuple[int, ...], int, tuple[Hop, ...]]] = [(0.0, 0, (), src, ())]
        done: set[int] = set()
        while heap:
            lat, hops, trail, node, path = heapq.heappop(heap)
            if node == dst:
                return path
            if node in done:
                continue
            done.add(node)
            for nxt, hop in self._adj[node]:
                if nxt not in done:
                    link = self.links[hop.link]
                    heapq.heappush(
                        heap,
                        (lat + link.latency_s, hops + 1, trail + (hop.link,), nxt, path + (hop,)),
                    )
        raise ValueError(f"no route from device {src} to device {dst}")

    def path_latency(self, src: int, dst: int) -> float:
        return sum(self.links[h.link].latency_s for h in self.route(src, dst))

    def transfer_rate(self, src: int, dst: int) -> float:
        """Uncontended bytes/s for a copy from ``src`` memory into ``dst`` memory."""
        key = (src, dst)
        if key not in self._rates:
            rate = min(self.devices[src].mem_bandwidth, self.devices[dst].mem_bandwidth)
            for hop in self.route(src, dst):
                rate = min(rate, self.links[hop.link].bandwidth(hop.forward))
            self._rates[key] = rate
        return self._rates[key]

    def validate(self) -> None:
        ids = sorted(self.devices)
        for other in ids[1:]:
            self.route(ids[0], other)

    # -- (de)serialisation --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "host": self.devices[self.host].name,
            "sync_latency_s": self.sync_latency_s,
            "devices": [_device_to_dict(d) for d in self.devices.values()],
            "links": [
                {
                    "name": link.name,
                    "a": self.devices[link.a].name,
                    "b": self.devices[link.b].name,
                    "bandwidth_ab": link.bandwidth_ab,
                    "bandwidth_ba": link.bandwidth_ba,
                    "latency_s": link.latency_s,
                    "dma_queue_depth": link.dma_queue_depth,
                    "joules_per_byte": link.joules_per_byte,
                }
                for link in self.links
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Topology:
        devices = [_device_from_dict(i, d) for i, d in enumerate(data["devices"])]
        ids = {d.name: d.device_id for d in devices}
        if len(ids) != len(devices):
            raise ValueError("device names must be unique")
        links = [
            LinkSpec(
                a=ids[link["a"]],
                b=ids[link["b"]],
                bandwidth_ab=float(link["bandwidth_ab"]),
                bandwidth_ba=None if link.get("bandwidth_ba") is None else float(link["bandwidth_ba"]),
                latency_s=float(link.get("latency_s", 0.0)),
                dma_queue_depth=int(link.get("dma_queue_depth", 8)),
                joules_per_byte=float(link.get("joules_per_byte", 0.0)),
                name=link.get("name", f"{link['a']}-{link['b']}"),
            )
            for link in data.get("links", [])
        ]
        topo = cls(
            devices,
            links,
            host=ids[data["host"]] if "host" in data else None,
            sync_latency_s=float(data.get("sync_latency_s", 0.0)),
            name=data.get("name", ""),
        )
        topo.validate()
        return topo


def _rate(value: Any) -> float:
    return math.inf if value in ("inf", None) else float(value)


def _device_to_dict(dev: DeviceSpec) -> dict[str, Any]:
    caps = dev.caps
    return {
        "name": dev.name,
        "kind": dev.kind.name,
        "mem_capacity": dev.mem_capacity,
        "mem_bandwidth": dev.mem_bandwidth,
        "compute_rate": dev.compute_rate,
        "sram_bytes": dev.sram_bytes,
        "kernel_startup_s": dev.kernel_startup_s,
        "warm_startup_s": dev.warm_startup_s,
        "persistent": dev.persistent,
        "queue_depth": dev.queue_depth,
        "opcodes": "all" if caps.opcodes == ALL_OPCODES else sorted(o.name for o in caps.opcodes),
        "decode_rate": {
            f.name: ("inf" if math.isinf(r) else r) for f, r in caps.decode_rate.items()
        },
        "stream_formats": sorted(f.name for f in caps.stream_formats),
        "expanded_format": caps.expanded_format.name,
        "stream_ingest": caps.stream_ingest,
        "power": {
            "idle_w": dev.power.idle_w,
            "joules_per_mac": dev.power.joules_per_mac,
            "joules_per_mem_byte": dev.power.joules_per_mem_byte,
            "sleep_w": dev.power.sleep_w,
            "gate_after_s": dev.power.gate_after_s,
            "wake_j": dev.power.wake_j,
        },
    }


def _device_from_dict(device_id: int, d: Mapping[str, Any]) -> DeviceSpec:
    opcodes = d.get("opcodes", "all" if float(d.get("compute_rate", 0.0)) > 0 else [])
    caps = DeviceCapabilities(
        opcodes=ALL_OPCODES if opcodes == "all" else frozenset(OpCode[o] for o in opcodes),
        decode_rate={CompressionFormat[k]: _rate(v) for k, v in d.get("decode_rate", {}).items()},
        stream_formats=frozenset(CompressionFormat[f] for f in d.get("stream_formats", [])),
        expanded_format=CompressionFormat[d.get("expanded_format", "INT4")],
        stream_ingest=bool(d.get("stream_ingest", False)),
    )
    power = d.get("power", {})
    return DeviceSpec(
        device_id=device_id,
        name=d["name"],
        kind=DeviceKind[d["kind"]],
        mem_capacity=int(d["mem_capacity"]),
        mem_bandwidth=float(d["mem_bandwidth"]),
        compute_rate=float(d.get("compute_rate", 0.0)),
        caps=caps,
        sram_bytes=int(d.get("sram_bytes", 0)),
        kernel_startup_s=float(d.get("kernel_startup_s", 0.0)),
        warm_startup_s=None if d.get("warm_startup_s") is None else float(d["warm_startup_s"]),
        persistent=bool(d.get("persistent", False)),
        queue_depth=int(d.get("queue_depth", 64)),
        power=PowerSpec(
            idle_w=float(power.get("idle_w", 0.0)),
            joules_per_mac=float(power.get("joules_per_mac", 0.0)),
            joules_per_mem_byte=float(power.get("joules_per_mem_byte", 0.0)),
            sleep_w=None if power.get("sleep_w") is None else float(power["sleep_w"]),
            gate_after_s=float(power.get("gate_after_s", 1e-3)),
            wake_j=float(power.get("wake_j", 0.0)),
        ),
    )
