"""Fitting simulator parameters to micro-benchmark timings (pure functions, no I/O).

Two models are fitted, the same two the simulator uses:

* a transfer takes ``latency + bytes / bandwidth`` (:func:`fit_transfer`);
* a kernel takes ``startup + max(W / B, C / P)`` -- the roofline of weight bytes ``W`` against
  memory bandwidth ``B`` and multiply-accumulates ``C`` against compute rate ``P``
  (:func:`fit_roofline`).

:func:`apply_fits` and :func:`apply_link_fit` write fitted values into a copy of a hardware
profile (the JSON format of ``configs/*.json``) and record which fields are now measured; every
other field of the profile remains an assumption.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TransferFit:
    latency_s: float
    bandwidth_bytes_per_s: float
    r2: float  # coefficient of determination on the per-size minima


@dataclass(frozen=True)
class RooflineFit:
    mem_bandwidth: float  # bytes/s
    compute_rate: float  # MAC/s
    startup_s: float
    rel_error: float  # RMS relative residual of the fitted model


def _per_size_minimum(sizes: Sequence[float], seconds: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(sizes, dtype=np.float64)
    t = np.asarray(seconds, dtype=np.float64)
    if x.shape != t.shape or x.ndim != 1:
        raise ValueError("sizes and seconds must be one-dimensional and of equal length")
    if x.size == 0 or np.any(x < 0) or np.any(t <= 0) or not np.all(np.isfinite(t)):
        raise ValueError("need non-negative sizes and positive, finite timings")
    unique = np.unique(x)
    return unique, np.array([t[x == u].min() for u in unique])


def fit_transfer(sizes_bytes: Sequence[float], seconds: Sequence[float]) -> TransferFit:
    """Fit ``time = latency + bytes / bandwidth`` to transfer timings.

    Repeated samples of one size are reduced to their *minimum*: timing noise on a real machine
    is one-sided (interference only ever adds time), so the minimum is the best estimate of the
    uncontended cost that the simulator's parameters describe.  The line is fitted by least
    squares on relative residuals, so the small transfers that determine the latency are not
    drowned out by the large ones that determine the bandwidth.  A negative intercept is clamped
    to zero and the slope refitted through the origin.
    """
    x, t = _per_size_minimum(sizes_bytes, seconds)
    if x.size < 2:
        raise ValueError("fit_transfer needs timings for at least two distinct sizes")
    w = 1.0 / t  # relative residuals: minimise sum(((a + b x - t) / t)^2)
    design = np.stack([w, x * w], axis=1)
    (latency, slope), *_ = np.linalg.lstsq(design, np.ones_like(t), rcond=None)
    if latency < 0:
        latency = 0.0
        slope = float(np.sum(x * w * w * t) / np.sum((x * w) ** 2))
    if not slope > 0:
        raise ValueError("transfer time does not grow with size (non-positive slope): the sizes "
                         "are too small to resolve the bandwidth")  # fmt: skip
    predicted = latency + slope * x
    total = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - float(np.sum((t - predicted) ** 2)) / total if total > 0 else 1.0
    return TransferFit(float(latency), float(1.0 / slope), r2)


def _roofline_loss(w, c, t, bandwidth: float, rate: float) -> tuple[float, float]:
    """(mean squared relative residual, best non-negative startup) for one (B, P) candidate."""
    core = np.maximum(w / bandwidth, c / rate)
    inv = 1.0 / (t * t)
    startup = max(float(np.sum((t - core) * inv) / np.sum(inv)), 0.0)
    return float(np.mean(((startup + core - t) / t) ** 2)), startup


def fit_roofline(weight_bytes: Sequence[float], macs: Sequence[float],
                 seconds: Sequence[float]) -> RooflineFit:  # fmt: skip
    """Fit ``T = startup + max(W / B, C / P)`` to kernel timings.

    The timings must include shapes that are memory-bound (few MACs per weight byte, e.g. a
    single-vector product) and shapes that are compute-bound (many MACs per weight byte, e.g. a
    large batch); otherwise one of the two rates is not identifiable and a ``ValueError`` explains
    which.  The search is a coarse logarithmic grid over (B, P) refined by repeated zooming; for
    each candidate the optimal start-up time has a closed form.
    """
    w = np.asarray(weight_bytes, dtype=np.float64)
    c = np.asarray(macs, dtype=np.float64)
    t = np.asarray(seconds, dtype=np.float64)
    if not (w.shape == c.shape == t.shape) or w.ndim != 1:
        raise ValueError("weight_bytes, macs and seconds must be one-dimensional and equal length")
    if w.size < 3:
        raise ValueError("fit_roofline needs at least three timings")
    if np.any(w <= 0) or np.any(c <= 0) or np.any(t <= 0) or not np.all(np.isfinite(t)):
        raise ValueError("weight bytes, MACs and timings must be positive and finite")
    intensity = c / w  # MACs per weight byte: what separates the two regimes
    if intensity.max() / intensity.min() < 1.0 + 1e-9:
        raise ValueError("every timing has the same MACs-per-byte ratio, so memory bandwidth and "
                         "compute rate cannot be separated: vary the batch size")  # fmt: skip

    # W / t and C / t are lower bounds of B and P (a kernel is never faster than its roofline).
    lo_b, lo_p = float((w / t).max()), float((c / t).max())
    centre = (math.log(lo_b) + math.log(30.0) / 2, math.log(lo_p) + math.log(30.0) / 2)
    span = math.log(30.0) * 1.2  # half-width: covers 0.6x .. 1600x of the lower bounds
    best = (math.inf, lo_b, lo_p, 0.0)
    for _ in range(7):
        grid_b = np.exp(np.linspace(centre[0] - span, centre[0] + span, 33))
        grid_p = np.exp(np.linspace(centre[1] - span, centre[1] + span, 33))
        for b in grid_b:
            for p in grid_p:
                loss, startup = _roofline_loss(w, c, t, float(b), float(p))
                if loss < best[0]:
                    best = (loss, float(b), float(p), startup)
        centre = (math.log(best[1]), math.log(best[2]))
        span /= 4.0
    loss, bandwidth, rate, startup = best

    # A rate is identified only if the fit gets clearly worse without its regime: compare with
    # the same model when that rate is made effectively infinite.
    rms = math.sqrt(loss)
    margin = max(0.01, 0.5 * rms)
    memory_bound = (w / bandwidth) >= (c / rate)
    n_mem, n_cmp = int(memory_bound.sum()), int((~memory_bound).sum())
    if math.sqrt(_roofline_loss(w, c, t, bandwidth * 1e9, rate)[0]) - rms < margin:
        n_mem = 0
    if math.sqrt(_roofline_loss(w, c, t, bandwidth, rate * 1e9)[0]) - rms < margin:
        n_cmp = 0
    if n_mem == 0:
        raise ValueError("no timing is memory-bound at the best fit, so the memory bandwidth is "
                         "not identifiable: add shapes with fewer MACs per weight byte "
                         "(single-vector products on large matrices)")  # fmt: skip
    if n_cmp == 0:
        raise ValueError("no timing is compute-bound at the best fit, so the compute rate is not "
                         "identifiable: add shapes with more MACs per weight byte "
                         "(larger batches)")  # fmt: skip
    return RooflineFit(bandwidth, rate, startup, rms)


# -- writing fits into hardware profiles -------------------------------------------------
_NOTE = ("Fields listed under 'measured' were fitted from measurements on this machine; every "
         "other value in the profile remains an assumption.")  # fmt: skip


def _record(profile: dict[str, Any], path: str, value: float) -> None:
    record = profile.setdefault("_calibration", {})
    record.setdefault("note", _NOTE)
    measured = set(record.get("measured", []))
    measured.add(path)
    record["measured"] = sorted(measured)
    record.setdefault("values", {})[path] = value


def _named(items: list[dict[str, Any]], name: str, what: str) -> dict[str, Any]:
    for item in items:
        if item.get("name") == name:
            return item
    known = [item.get("name") for item in items]
    raise KeyError(f"profile has no {what} named {name!r}; known: {known}")


def apply_fits(profile: dict[str, Any], device: str, *, mem_bandwidth: float | None = None,
               compute_rate: float | None = None,
               kernel_startup_s: float | None = None) -> dict[str, Any]:  # fmt: skip
    """New profile with measured values written into ``device`` (the input is not modified)."""
    out = copy.deepcopy(profile)
    target = _named(out.get("devices", []), device, "device")
    updates = {"mem_bandwidth": mem_bandwidth, "compute_rate": compute_rate,
               "kernel_startup_s": kernel_startup_s}  # fmt: skip
    for key, value in updates.items():
        if value is None:
            continue
        if not (math.isfinite(value) and value >= 0) or (key != "kernel_startup_s" and value <= 0):
            raise ValueError(f"{key} must be positive and finite, got {value!r}")
        target[key] = float(value)
        _record(out, f"devices.{device}.{key}", float(value))
    return out


def apply_link_fit(profile: dict[str, Any], link: str, fit: TransferFit,
                   direction: str = "both") -> dict[str, Any]:  # fmt: skip
    """New profile with a measured transfer fit written into ``link``.

    ``direction`` is ``"ab"``, ``"ba"`` or ``"both"`` relative to the link's endpoints.  A link
    has one latency: when the two directions were fitted separately it is the mean of the two.
    """
    if direction not in ("ab", "ba", "both"):
        raise ValueError(f"direction must be 'ab', 'ba' or 'both', got {direction!r}")
    out = copy.deepcopy(profile)
    target = _named(out.get("links", []), link, "link")
    directions = ("ab", "ba") if direction == "both" else (direction,)
    for d in directions:
        target[f"bandwidth_{d}"] = float(fit.bandwidth_bytes_per_s)
        _record(out, f"links.{link}.bandwidth_{d}", float(fit.bandwidth_bytes_per_s))
        out["_calibration"].setdefault("link_latency_s", {}).setdefault(link, {})[d] = fit.latency_s
    latencies = out["_calibration"]["link_latency_s"][link].values()
    target["latency_s"] = float(sum(latencies) / len(latencies))
    _record(out, f"links.{link}.latency_s", target["latency_s"])
    return out
