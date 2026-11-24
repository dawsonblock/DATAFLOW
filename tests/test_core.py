"""Tensor descriptors, formats, the IR graph and topology routing."""

from __future__ import annotations

import math

import pytest

from basp.core.graph import Graph, GraphError, OpCode, OpNode
from basp.core.tensor import (
    MEMORY_POLICIES,
    AccessFlags,
    CompressionFormat,
    DType,
    FormatTable,
    TensorDescriptor,
    TensorKind,
    TensorSpec,
    activation_bytes,
    contiguous_strides,
    pad_shape,
)
from basp.core.topology import Topology
from basp.experiments.profiles import load_topology
from tests.conftest import GPU, HOST, MOD0, MOD1, make_topology


def spec(tid: int, kind=TensorKind.ACTIVATION, nbytes: int = 16) -> TensorSpec:
    fmt = CompressionFormat.INT4 if kind is TensorKind.EXPERT_WEIGHT else None
    return TensorSpec(tid, f"t{tid}", kind, (nbytes,), DType.FP16, nbytes, fmt, nbytes)


# -- tensors -----------------------------------------------------------------------
def test_descriptor_matches_c_layout_and_round_trips():
    d = TensorDescriptor(2**63 + 5, 7, 99_090_432, 3, int(DType.INT4), int(CompressionFormat.INT4),
                         int(AccessFlags.READ | AccessFlags.IMMUTABLE), (3, 4096, 14336, 0),
                         (1, 2, 3, 0))  # fmt: skip
    raw = d.pack()
    # 3 x u64 + 4 x u32 + 2 x 4 x u64, little endian, no padding
    assert len(raw) == TensorDescriptor.WIRE_SIZE == 104
    assert TensorDescriptor.unpack(raw) == d
    assert d.bumped().version == 8
    with pytest.raises(ValueError):
        TensorDescriptor.unpack(raw[:-1])


def test_effective_bits_per_weight_follow_the_group_scale_rule():
    formats = FormatTable()
    assert formats.bits(CompressionFormat.INT4) == 4.5
    assert formats.bits(CompressionFormat.INT8) == 8.5
    assert formats.bits(CompressionFormat.INT2) == 2.5
    assert formats.bits(CompressionFormat.RAW_FP16) == 16.0
    # the specification's example expert: 3 x 4096 x 14336 weights at 4.5 bits = 94.5 MiB
    assert formats.weight_bytes(3 * 4096 * 14336, CompressionFormat.INT4) == 99_090_432


def test_only_lossless_equivalent_formats_are_interchangeable():
    formats = FormatTable()
    assert formats.interchangeable(CompressionFormat.ENTROPY, CompressionFormat.INT4)
    assert formats.interchangeable(CompressionFormat.INT4, CompressionFormat.INT4)
    assert not formats.interchangeable(CompressionFormat.INT2, CompressionFormat.INT4)
    assert not formats.interchangeable(CompressionFormat.INT8, CompressionFormat.RAW_FP16)


def test_format_overrides_and_validation():
    formats = FormatTable({CompressionFormat.ENTROPY: 3.0})
    assert formats.bits(CompressionFormat.ENTROPY) == 3.0
    assert FormatTable().spec(CompressionFormat.ENTROPY).assumed
    with pytest.raises(ValueError):
        FormatTable({CompressionFormat.ENTROPY: 0})


def test_memory_policies_cover_every_tensor_kind():
    assert set(MEMORY_POLICIES) == set(TensorKind)
    assert MEMORY_POLICIES[TensorKind.WEIGHT].immutable
    assert MEMORY_POLICIES[TensorKind.EXPERT_WEIGHT].replicable
    assert MEMORY_POLICIES[TensorKind.ACTIVATION].temporary
    assert not MEMORY_POLICIES[TensorKind.LEARNING_UPDATE].immutable


def test_shape_helpers():
    assert pad_shape((3, 5)) == (3, 5, 0, 0)
    assert contiguous_strides((3, 5), 2) == (10, 2, 0, 0)
    assert activation_bytes(4096) == 8192
    with pytest.raises(ValueError):
        pad_shape((1, 2, 3, 4, 5))
    d = spec(1, TensorKind.EXPERT_WEIGHT).descriptor(owner_device=2)
    assert d.owner_device == 2 and d.compression_format == int(CompressionFormat.INT4)


# -- graph -------------------------------------------------------------------------
def chain_graph() -> Graph:
    g = Graph()
    for tid in range(1, 6):
        g.add_tensor(spec(tid))
    g.add_tensor(spec(9, TensorKind.EXPERT_WEIGHT))
    g.add_op(OpNode(3, OpCode.ADD, "c", inputs=(2, 3), outputs=(4,)))
    g.add_op(OpNode(1, OpCode.MATVEC, "a", inputs=(1,), weights=(9,), outputs=(2,)))
    g.add_op(OpNode(2, OpCode.MATVEC, "b", inputs=(1,), weights=(9,), outputs=(3,)))
    g.add_op(OpNode(4, OpCode.RMS_NORM, "d", inputs=(4,), outputs=(5,)))
    return g


def test_dependencies_follow_tensor_production():
    g = chain_graph()
    assert g.deps(3) == (1, 2) and g.deps(1) == () and g.deps(4) == (3,)
    assert g.dependents(1) == (3,) and g.producer(4) == 3 and g.consumers(1) == (1, 2)
    assert g.external_inputs() == (1,) and g.sinks() == (4,)
    assert g.topo_order() == [1, 2, 3, 4]
    assert g.topo_order(among=[3, 4]) == [3, 4]  # outside dependencies count as satisfied
    g.validate()


def test_topological_order_is_deterministic_regardless_of_insertion_order():
    g = chain_graph()
    assert g.topo_order() == sorted(g.ops)


def test_graph_rejects_malformed_operations():
    g = chain_graph()
    with pytest.raises(GraphError, match="two producers"):
        g.add_op(OpNode(5, OpCode.ADD, "dup", inputs=(1,), outputs=(2,)))
    with pytest.raises(GraphError, match="unknown tensor"):
        g.add_op(OpNode(6, OpCode.ADD, "ghost", inputs=(77,), outputs=()))
    with pytest.raises(GraphError, match="own outputs"):
        g.add_op(OpNode(7, OpCode.ADD, "loop", inputs=(1,), outputs=(1,)))
    with pytest.raises(GraphError, match="immutable"):
        g.add_op(OpNode(8, OpCode.ADD, "clobber", inputs=(1,), outputs=(9,)))
    with pytest.raises(GraphError, match="not a weight"):
        g.add_op(OpNode(10, OpCode.MATVEC, "w", inputs=(1,), weights=(2,), outputs=()))
    with pytest.raises(GraphError, match="duplicate op"):
        g.add_op(OpNode(1, OpCode.ADD, "again", inputs=(1,), outputs=()))


def test_cycles_through_control_dependencies_are_detected():
    g = chain_graph()
    g.add_tensor(spec(20))
    g.add_op(OpNode(11, OpCode.ADD, "late", inputs=(5,), outputs=(20,), control_deps=(12,)))
    g.add_op(OpNode(12, OpCode.ADD, "early", inputs=(20,), outputs=()))
    with pytest.raises(GraphError, match="cycle"):
        g.topo_order()
    h = chain_graph()
    h.add_op(OpNode(13, OpCode.ADD, "dangling", inputs=(5,), outputs=(), control_deps=(99,)))
    with pytest.raises(GraphError, match="unknown control"):
        h.validate()


# -- topology ----------------------------------------------------------------------
def test_routing_is_shortest_and_rates_are_bottlenecks(topology: Topology):
    route = topology.route(GPU, MOD0)
    assert [(topology.links[h.link].name, h.forward) for h in route] == [("gpu", False), ("mod0", True)]
    assert topology.route(MOD0, MOD0) == ()
    assert topology.path_latency(GPU, MOD0) == pytest.approx(7e-6)
    assert topology.transfer_rate(GPU, MOD0) == 16e9  # slowest hop
    assert topology.transfer_rate(HOST, GPU) == 24e9
    assert topology.transfer_rate(MOD0, MOD1) == 16e9


def test_topology_round_trips_through_its_dictionary_form(topology: Topology):
    clone = Topology.from_dict(topology.to_dict())
    assert clone.to_dict() == topology.to_dict()
    assert clone.devices[MOD0].caps.decode_rate[CompressionFormat.INT4] == math.inf
    assert clone.devices[MOD0].caps.fused(CompressionFormat.ENTROPY)
    assert not clone.devices[GPU].caps.fused(CompressionFormat.ENTROPY)


def test_shipped_hardware_profiles_load_and_validate():
    desktop, basp = load_topology("desktop_gpu12"), load_topology("basp_4module")
    assert {d.name for d in desktop.devices.values()} == {"host", "gpu"}
    assert len(basp.of_kind(basp.by_name("mod0").kind)) == 4
    module = basp.by_name("mod0")
    # 2 x DDR5-5600 at 0.6 sustained efficiency, 1024 lanes x 250 MHz
    assert module.mem_bandwidth == pytest.approx(2 * 5600e6 * 8 * 0.6, rel=0.01)
    assert module.compute_rate == 1024 * 250e6


def test_invalid_topologies_are_rejected():
    good = make_topology()
    with pytest.raises(ValueError, match="duplicate device"):
        Topology([good.devices[HOST], good.devices[HOST]], [])
    with pytest.raises(ValueError, match="no route"):
        Topology(list(good.devices.values()), good.links[:1]).validate()
    with pytest.raises(FileNotFoundError):
        load_topology("does_not_exist")
