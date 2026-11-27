"""Shared fixtures: a small illustrative topology and helpers used across the suite."""

from __future__ import annotations

import math

import pytest

from basp.core.capabilities import NO_COMPUTE, DeviceCapabilities, DeviceKind
from basp.core.tensor import CompressionFormat as F
from basp.core.topology import DeviceSpec, LinkSpec, Topology

GB = 1e9
HOST, GPU, MOD0, MOD1, NVME = 0, 1, 2, 3, 4


def make_topology(*, gpu_mem: int = 1 << 30, sram: int = 4 << 20, queue_depth: int = 64,
                  dma_depth: int = 8, with_nvme: bool = False) -> Topology:  # fmt: skip
    """host CPU + GPU + two compute-memory modules (star around the host)."""
    cpu_caps = DeviceCapabilities(
        decode_rate={F.INT4: math.inf, F.ENTROPY: 4e9}, stream_formats=frozenset({F.INT4}),
        stream_ingest=True,
    )  # fmt: skip
    gpu_caps = DeviceCapabilities(
        decode_rate={F.INT4: math.inf, F.ENTROPY: 40e9}, stream_formats=frozenset({F.INT4}),
        stream_ingest=True,
    )  # fmt: skip
    mod_caps = DeviceCapabilities(
        decode_rate={F.INT4: math.inf, F.ENTROPY: 8e9},
        stream_formats=frozenset({F.INT4, F.ENTROPY}), stream_ingest=True,
    )  # fmt: skip
    devices = [
        DeviceSpec(HOST, "host", DeviceKind.CPU, 8 << 30, 54 * GB, 30e9, cpu_caps,
                   kernel_startup_s=0.5e-6, queue_depth=queue_depth),
        DeviceSpec(GPU, "gpu", DeviceKind.GPU, gpu_mem, 500 * GB, 20e12, gpu_caps,
                   kernel_startup_s=3e-6, queue_depth=queue_depth),
        DeviceSpec(MOD0, "mod0", DeviceKind.NEAR_MEMORY, 4 << 30, 54 * GB, 256e9, mod_caps,
                   sram_bytes=sram, kernel_startup_s=1e-6, queue_depth=queue_depth),
        DeviceSpec(MOD1, "mod1", DeviceKind.NEAR_MEMORY, 4 << 30, 54 * GB, 256e9, mod_caps,
                   sram_bytes=sram, kernel_startup_s=1e-6, queue_depth=queue_depth),
    ]  # fmt: skip
    links = [
        LinkSpec(HOST, GPU, 24 * GB, latency_s=5e-6, dma_queue_depth=dma_depth, name="gpu"),
        LinkSpec(HOST, MOD0, 16 * GB, latency_s=2e-6, dma_queue_depth=dma_depth, name="mod0"),
        LinkSpec(HOST, MOD1, 16 * GB, latency_s=2e-6, dma_queue_depth=dma_depth, name="mod1"),
    ]
    if with_nvme:
        devices.append(DeviceSpec(NVME, "nvme", DeviceKind.STORAGE, 1 << 40, 7 * GB, 0.0, NO_COMPUTE))
        links.append(LinkSpec(HOST, NVME, 7 * GB, latency_s=10e-6, name="nvme"))
    return Topology(devices, links, host=HOST, sync_latency_s=2e-6, name="test")


@pytest.fixture
def topology() -> Topology:
    return make_topology()
