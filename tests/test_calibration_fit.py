"""Fitting the simulator's transfer and roofline models to synthetic timings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from basp.calibration import (
    TransferFit,
    apply_fits,
    apply_link_fit,
    fit_roofline,
    fit_transfer,
)
from basp.core.topology import Topology

PROFILE = json.loads((Path(__file__).resolve().parents[1] / "configs" / "desktop_gpu12.json").read_text())


def noisy(values: np.ndarray, rng, scale: float) -> np.ndarray:
    """One-sided timing noise: interference only ever adds time."""
    return values * (1.0 + np.abs(rng.normal(0.0, scale, values.size)))


def test_transfer_fit_recovers_latency_and_bandwidth():
    rng = np.random.default_rng(1)
    sizes = np.repeat([4e3, 64e3, 1e6, 16e6, 256e6], 7)
    seconds = noisy(5e-6 + sizes / 24e9, rng, 0.05)
    fit = fit_transfer(sizes, seconds)
    # 5% one-sided noise, minimum of 7 repeats per size: both parameters within 5%
    assert fit.latency_s == pytest.approx(5e-6, rel=0.05)
    assert fit.bandwidth_bytes_per_s == pytest.approx(24e9, rel=0.05)
    assert fit.r2 > 0.999
    exact = fit_transfer([1e3, 1e6, 1e9], [2e-6 + 1e3 / 16e9, 2e-6 + 1e6 / 16e9, 2e-6 + 1e9 / 16e9])
    assert exact.latency_s == pytest.approx(2e-6, rel=1e-6)
    assert exact.bandwidth_bytes_per_s == pytest.approx(16e9, rel=1e-6) and exact.r2 == pytest.approx(1)


def test_transfer_fit_clamps_negative_latency_and_rejects_degenerate_data():
    sizes = np.array([1e6, 2e6, 4e6, 8e6])
    through_origin = sizes / 10e9 * np.array([0.97, 1.0, 1.01, 1.02])  # intercept would be < 0
    fit = fit_transfer(sizes, through_origin)
    assert fit.latency_s == 0.0 and fit.bandwidth_bytes_per_s == pytest.approx(10e9, rel=0.05)
    with pytest.raises(ValueError, match="at least two distinct sizes"):
        fit_transfer([1e6, 1e6, 1e6], [1e-3, 1.1e-3, 0.9e-3])
    with pytest.raises(ValueError, match="non-positive slope"):
        fit_transfer([1e3, 1e6, 1e9], [3e-3, 2e-3, 1e-3])
    with pytest.raises(ValueError, match="positive, finite timings"):
        fit_transfer([1e3, 1e6], [1e-3, 0.0])
    with pytest.raises(ValueError, match="equal length"):
        fit_transfer([1e3, 1e6], [1e-3])


def kernel_points(batches, shapes=((2048, 2048), (4096, 4096), (8192, 4096))):
    w = np.array([r * c * 2.0 for r, c in shapes for _ in batches])
    c = np.array([float(r * c * b) for r, c in shapes for b in batches])
    return w, c


def test_roofline_fit_recovers_bandwidth_rate_and_startup():
    rng = np.random.default_rng(2)
    w, c = kernel_points((1, 4, 16, 64, 256))
    seconds = noisy(3e-6 + np.maximum(w / 500e9, c / 20e12), rng, 0.02)
    fit = fit_roofline(w, c, seconds)
    # 2% one-sided noise biases every timing upwards by about 1.6%: parameters within 5%
    assert fit.mem_bandwidth == pytest.approx(500e9, rel=0.05)
    assert fit.compute_rate == pytest.approx(20e12, rel=0.05)
    assert fit.startup_s == pytest.approx(3e-6, rel=0.15)
    assert fit.rel_error < 0.03
    clean = fit_roofline(w, c, 1e-6 + np.maximum(w / 54e9, c / 256e9))
    assert clean.mem_bandwidth == pytest.approx(54e9, rel=2e-3)
    assert clean.compute_rate == pytest.approx(256e9, rel=2e-3)
    assert clean.rel_error < 2e-3


def test_roofline_fit_explains_what_it_cannot_identify():
    w, c = kernel_points((1,))
    with pytest.raises(ValueError, match="same MACs-per-byte ratio"):
        fit_roofline(w, c, 1e-6 + w / 500e9)
    w, c = kernel_points((1, 2))  # still memory-bound everywhere
    with pytest.raises(ValueError, match="compute rate is not identifiable"):
        fit_roofline(w, c, 1e-6 + np.maximum(w / 500e9, c / 20e15))
    w, c = kernel_points((512, 2048))  # compute-bound everywhere
    with pytest.raises(ValueError, match="memory bandwidth is not identifiable"):
        fit_roofline(w, c, 1e-6 + np.maximum(w / 500e15, c / 20e12))
    with pytest.raises(ValueError, match="at least three timings"):
        fit_roofline([1.0, 2.0], [1.0, 4.0], [1e-3, 2e-3])
    with pytest.raises(ValueError, match="positive and finite"):
        fit_roofline([1.0, 2.0, 3.0], [1.0, 4.0, 9.0], [1e-3, -2e-3, 1e-3])


def test_apply_fits_returns_a_new_loadable_profile_and_records_what_is_measured():
    before = json.dumps(PROFILE, sort_keys=True)
    fitted = apply_fits(PROFILE, "gpu", mem_bandwidth=912e9, compute_rate=15e12, kernel_startup_s=9e-6)
    fitted = apply_fits(fitted, "host", mem_bandwidth=38e9)
    assert json.dumps(PROFILE, sort_keys=True) == before  # the template is untouched
    topology = Topology.from_dict(fitted)
    gpu, host = topology.by_name("gpu"), topology.by_name("host")
    assert (gpu.mem_bandwidth, gpu.compute_rate, gpu.kernel_startup_s) == (912e9, 15e12, 9e-6)
    assumed = next(d for d in PROFILE["devices"] if d["name"] == "host")["compute_rate"]
    assert host.mem_bandwidth == 38e9 and host.compute_rate == assumed  # unmeasured: unchanged
    record = fitted["_calibration"]
    assert record["measured"] == ["devices.gpu.compute_rate", "devices.gpu.kernel_startup_s",
                                  "devices.gpu.mem_bandwidth", "devices.host.mem_bandwidth"]  # fmt: skip
    assert record["values"]["devices.host.mem_bandwidth"] == 38e9 and "assumption" in record["note"]
    assert apply_fits(PROFILE, "gpu") == PROFILE  # nothing measured: nothing recorded
    with pytest.raises(KeyError, match="no device named 'fpga'"):
        apply_fits(PROFILE, "fpga", mem_bandwidth=1e9)
    with pytest.raises(ValueError, match="mem_bandwidth must be positive"):
        apply_fits(PROFILE, "gpu", mem_bandwidth=0.0)


def test_apply_link_fit_handles_directions_and_averages_latency():
    down, up = TransferFit(6e-6, 22e9, 0.999), TransferFit(10e-6, 18e9, 0.998)
    both = apply_link_fit(PROFILE, "pcie-gpu", down)
    link = Topology.from_dict(both).links[0]
    assert (link.bandwidth_ab, link.bandwidth_ba, link.latency_s) == (22e9, 22e9, 6e-6)
    split = apply_link_fit(apply_link_fit(PROFILE, "pcie-gpu", down, "ab"), "pcie-gpu", up, "ba")
    link = Topology.from_dict(split).links[0]
    assert (link.bandwidth_ab, link.bandwidth_ba) == (22e9, 18e9)
    assert link.latency_s == pytest.approx(8e-6)  # one latency per link: the mean of both directions
    assert split["_calibration"]["measured"] == ["links.pcie-gpu.bandwidth_ab",
                                                 "links.pcie-gpu.bandwidth_ba",
                                                 "links.pcie-gpu.latency_s"]  # fmt: skip
    one_way = apply_link_fit(PROFILE, "pcie-gpu", up, "ba")
    assert one_way["links"][0]["bandwidth_ab"] == PROFILE["links"][0]["bandwidth_ab"]
    with pytest.raises(ValueError, match="direction must be"):
        apply_link_fit(PROFILE, "pcie-gpu", down, "sideways")
    with pytest.raises(KeyError, match="no link named"):
        apply_link_fit(PROFILE, "cxl", down)
