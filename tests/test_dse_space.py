"""Discrete design spaces: enumeration, feasibility, encoding and variation operators."""

from __future__ import annotations

import numpy as np
import pytest

from basp.dse import DesignSpace, Param


def space(**kw) -> DesignSpace:
    return DesignSpace(
        [
            Param("lanes", (256, 512, 1024, 2048)),
            Param("modules", (1, 2, 4)),
            Param("kind", ("fpga", "gpu", "cpu"), ordered=False),
            Param("clock", (250,)),
        ],
        **kw,
    )


def no_big_cpu(config) -> bool:
    return not (config["kind"] == "cpu" and config["lanes"] > 512)


def test_parameters_and_spaces_reject_malformed_definitions():
    assert Param("a", [1, 2]).values == (1, 2)  # coerced to a tuple
    with pytest.raises(ValueError, match="no values"):
        Param("a", ())
    with pytest.raises(ValueError, match="twice"):
        Param("a", (1, 2, 1))
    with pytest.raises(ValueError, match="unhashable"):
        Param("a", ([1], [2]))
    with pytest.raises(ValueError, match="unique"):
        DesignSpace([Param("a", (1,)), Param("a", (2,))])
    with pytest.raises(ValueError, match="at least one"):
        DesignSpace([])


def test_shape_identity_and_enumeration_order():
    s = space()
    assert s.names == ("lanes", "modules", "kind", "clock")
    assert s.size == 4 * 3 * 3 * 1 and s.dim == 1 + 1 + 3 + 1
    configs = list(s.enumerate())
    assert len(configs) == s.size and len({s.key(c) for c in configs}) == s.size
    assert configs[0] == {"lanes": 256, "modules": 1, "kind": "fpga", "clock": 250}
    assert configs[1]["kind"] == "gpu" and configs[3]["modules"] == 2  # last parameter varies fastest
    assert configs == sorted(configs, key=lambda c: [p.values.index(c[p.name]) for p in s.params])
    assert s.key({"clock": 250, "kind": "gpu", "modules": 2, "lanes": 512}) == (512, 2, "gpu", 250)
    assert list(s.canonical({"clock": 250, "kind": "gpu", "modules": 2, "lanes": 512})) == list(s.names)


def test_constraints_restrict_enumeration_and_feasibility():
    s = space(constraints=[no_big_cpu])
    configs = list(s.enumerate())
    assert len(configs) == 36 - 2 * 3 and all(no_big_cpu(c) for c in configs)
    assert s.size == 36  # size ignores constraints
    good = {"lanes": 512, "modules": 2, "kind": "cpu", "clock": 250}
    assert s.feasible(good)
    assert not s.feasible({**good, "lanes": 1024})  # violates the constraint
    assert not s.feasible({**good, "lanes": 300})  # not a legal value
    assert not s.feasible({k: v for k, v in good.items() if k != "clock"})  # missing parameter
    assert not s.feasible({**good, "extra": 1})  # not a parameter of this space
    assert not s.feasible({**good, "kind": ["cpu"]})  # unhashable value


def test_encoding_scales_ordered_levels_and_one_hots_categories():
    s = space()
    vec = s.encode({"lanes": 1024, "modules": 1, "kind": "gpu", "clock": 250})
    np.testing.assert_allclose(vec, [2 / 3, 0.0, 0.0, 1.0, 0.0, 0.5])
    assert vec.dtype == np.float64 and len(vec) == s.dim
    low = s.encode({"lanes": 256, "modules": 4, "kind": "cpu", "clock": 250})
    np.testing.assert_allclose(low, [0.0, 1.0, 0.0, 0.0, 1.0, 0.5])


def test_neighbors_step_ordered_levels_and_swap_categories():
    s = space(constraints=[no_big_cpu])
    base = {"lanes": 512, "modules": 1, "kind": "gpu", "clock": 250}
    found = s.neighbors(base)
    changed = sorted((name, n[name]) for n in found for name in s.names if n[name] != base[name])
    assert changed == [("kind", "cpu"), ("kind", "fpga"), ("lanes", 256), ("lanes", 1024), ("modules", 2)]
    assert all(sum(n[k] != base[k] for k in s.names) == 1 and s.feasible(n) for n in found)
    edge = {"lanes": 1024, "modules": 4, "kind": "gpu", "clock": 250}
    kinds = {n["kind"] for n in s.neighbors(edge)}
    assert "cpu" not in kinds  # 1024 lanes on a CPU is infeasible, so it is not a neighbour


def test_sampling_is_deterministic_feasible_and_covers_the_space():
    s = space(constraints=[no_big_cpu])
    first = [s.sample(np.random.default_rng(7)) for _ in range(3)]
    assert first[0] == first[1] == first[2]
    rng = np.random.default_rng(0)
    draws = [s.sample(rng) for _ in range(2000)]
    assert all(s.feasible(c) for c in draws)
    assert {s.key(c) for c in draws} == {s.key(c) for c in s.enumerate()}
    impossible = DesignSpace([Param("a", (1, 2))], constraints=[lambda c: False])
    with pytest.raises(ValueError, match="no feasible"):
        impossible.sample(np.random.default_rng(0))
    assert list(impossible.enumerate()) == []


def test_mutation_returns_a_different_feasible_configuration():
    s = space(constraints=[no_big_cpu])
    rng = np.random.default_rng(1)
    base = {"lanes": 512, "modules": 2, "kind": "cpu", "clock": 250}
    adjacent = total = 0
    for _ in range(500):
        child = s.mutate(base, rng)
        assert s.feasible(child) and child != base
        if child["lanes"] != base["lanes"] and child["kind"] == "cpu":
            total += 1
            adjacent += child["lanes"] == 256  # the only other feasible level for a CPU
    assert total > 20 and adjacent == total
    everything = [s.mutate(base, rng, rate=1.0) for _ in range(200)]
    assert all(c["modules"] != base["modules"] and c["kind"] != base["kind"] for c in everything)
    a = s.mutate(base, np.random.default_rng(5))
    assert a == s.mutate(base, np.random.default_rng(5))  # deterministic for a given generator state


def test_ordered_parameters_mostly_move_to_adjacent_levels():
    s = DesignSpace([Param("x", tuple(range(9)))])
    rng = np.random.default_rng(3)
    steps = [abs(s.mutate({"x": 4}, rng)["x"] - 4) for _ in range(1000)]
    assert 0 not in steps
    assert 0.75 < sum(step == 1 for step in steps) / len(steps) < 0.95  # ~80% steps + adjacent jumps
    assert max(steps) == 4  # occasional jumps reach the ends of the scale


def test_mutation_of_an_isolated_point_returns_it_unchanged():
    lonely = DesignSpace([Param("a", (1, 2, 3))], constraints=[lambda c: c["a"] == 2])
    assert lonely.mutate({"a": 2}, np.random.default_rng(0)) == {"a": 2}
    single = DesignSpace([Param("a", (7,))])
    assert single.mutate({"a": 7}, np.random.default_rng(0)) == {"a": 7}
    assert single.neighbors({"a": 7}) == []


def test_crossover_mixes_parent_genes_and_repairs_infeasible_children():
    s = space()
    rng = np.random.default_rng(2)
    a = {"lanes": 256, "modules": 1, "kind": "fpga", "clock": 250}
    b = {"lanes": 2048, "modules": 4, "kind": "cpu", "clock": 250}
    children = [s.crossover(a, b, rng) for _ in range(200)]
    assert all(c[name] in (a[name], b[name]) for c in children for name in s.names)
    assert len({s.key(c) for c in children}) == 8  # every combination of the three differing genes

    def all_or_nothing(config) -> bool:  # only the two parents themselves are feasible mixes
        return (config["lanes"], config["modules"], config["kind"]) in ((256, 1, "fpga"), (2048, 4, "cpu"))

    strict = space(constraints=[all_or_nothing])
    repaired = [strict.crossover(a, b, rng) for _ in range(100)]
    assert all(c in (a, b) for c in repaired) and a in repaired and b in repaired
