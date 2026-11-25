"""From a measured benchmark to a simulator prediction, and from both to a calibration verdict.

A :class:`~basp.calibration.Measurement` records what a real system did
(typically llama.cpp with ``-ngl N`` layers on the GPU).  ``predict`` simulates
the same model, quantisation, layer split, context and batch on the hardware
profile of the machine that produced the measurement.  ``fit_profile`` tunes a
machine profile on the **calibration** split; ``calibration_status`` scores
predictions against the **validation** split, which the fit never sees.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from basp.calibration import (
    CalibrationReport,
    Measurement,
    MeasurementSet,
    apply_fits,
    evaluate_calibration,
)
from basp.core.capabilities import DeviceKind
from basp.core.tensor import FormatTable
from basp.core.topology import Topology
from basp.dataflow.base import DataflowMode, ExecutionPolicy
from basp.experiments.harness import build
from basp.experiments.profiles import load_topology
from basp.scheduler import placement
from basp.workloads.trace import synth_trace
from basp.workloads.zoo import MODELS

#: hardware profile for each machine name used in measurement records
MACHINE_PROFILES: dict[str, str] = {
    "i9-11900kf-3080ti": "i9_11900kf_rtx3080ti",
    "macbook-m2": "macbook_m2",
}


def predict(measurement: Measurement, profiles: Mapping[str, str | Topology] | None = None, *,
            tokens: int = 6, warmup_tokens: int = 2) -> float | None:  # fmt: skip
    """Simulated tokens per second for the benchmark ``measurement`` describes.

    Returns ``None`` when the machine has no profile, the model shape is unknown
    or the model does not fit the profile -- such measurements cannot be scored.
    """
    profiles = {**MACHINE_PROFILES, **(profiles or {})}
    profile = profiles.get(measurement.machine)
    spec = MODELS.get(measurement.model)
    if profile is None or spec is None:
        return None
    topology = load_topology(profile) if isinstance(profile, str) else profile
    gpus = topology.of_kind(DeviceKind.GPU)
    host = topology.host
    if not gpus:
        return None
    gpu = gpus[0].device_id
    # the measured file's bits per weight replace the nominal size of the stored format
    formats = FormatTable({spec.weight_format: measurement.bits_per_weight})
    layers = spec.n_layers if measurement.n_gpu_layers < 0 else min(measurement.n_gpu_layers, spec.n_layers)
    prefill = measurement.kind == "prefill"
    batch = max(measurement.batch, 1)
    if prefill:
        # prompt tokens are processed together: llama-bench records the physical batch
        batch = measurement.batch if measurement.batch > 1 else 512
        tokens, warmup_tokens = 1, 0
        context = measurement.context_len  # depth before the prompt; the prompt is one sequence
    else:
        context = measurement.context_len or 256  # average KV length over the generated tokens
    trace = synth_trace(tokens, spec.n_layers, spec.n_experts, spec.top_k, batch=batch, seed=0)
    policy = ExecutionPolicy("static_split", frozenset({DataflowMode.WEIGHT_STATIONARY}),
                             frozenset({host, gpu}), joint_rounds=0)  # fmt: skip
    try:
        setup = build(topology, spec, policy,
                      lambda p, r: placement.place_static_split(p, r, gpu, host, layers,
                                                                measurement.n_cpu_moe),
                      batch=batch, context_len=context, max_new_tokens=tokens, formats=formats,
                      prompt=prefill)  # fmt: skip
    except placement.PlacementError:
        return None
    result = setup.run(trace, tokens=tokens, warmup_tokens=warmup_tokens)
    return result.tokens_per_s if result.ok else None


def load_measurements(path: str | Path | None) -> MeasurementSet:
    """One measurement set from a file or from every ``*/measurements.json`` under a directory."""
    combined = MeasurementSet()
    if path is None or not Path(path).exists():
        return combined
    files = sorted(Path(path).glob("*/measurements.json")) if Path(path).is_dir() else [Path(path)]
    for file in files:
        for m in MeasurementSet.load(file):
            combined.add(m)
    return combined


def calibration_status(path: str | Path | None, profiles: Mapping[str, str | Topology] | None = None,
                       *, median_max: float = 0.10, p95_max: float = 0.20,
                       min_validation: int = 8) -> CalibrationReport:  # fmt: skip
    """Calibration verdict from measurements: a file, or a directory whose ``*/measurements.json``
    files (one per machine, as ``benchmarks.run_calibration`` writes them) are combined.
    ``None`` or a missing path: nothing measured."""
    measurements = load_measurements(path)
    return score(measurements, profiles, median_max=median_max, p95_max=p95_max,
                 min_validation=min_validation)  # fmt: skip


def score(measurements: MeasurementSet, profiles: Mapping[str, str | Topology] | None = None, *,
          median_max: float = 0.10, p95_max: float = 0.20, min_validation: int = 8) -> CalibrationReport:  # fmt: skip
    """Predict every measurement and judge the validation split against the thresholds."""
    return evaluate_calibration(lambda m: predict(m, profiles), measurements, median_max=median_max,
                                p95_max=p95_max, min_validation=min_validation)  # fmt: skip


# -- fitting a machine profile to the calibration split ----------------------------------
#: profile fields a fit may scale, as (role, field); the role is resolved to a device name
FIT_PARAMETERS: tuple[str, ...] = ("host.compute_rate", "host.mem_bandwidth", "gpu.mem_bandwidth",
                                   "gpu.compute_rate")  # fmt: skip


@dataclass
class ProfileFit:
    """A machine profile tuned on calibration measurements."""

    profile: dict[str, Any]  # the fitted profile, in its JSON form
    scales: dict[str, float]  # fitted value / prior value, per parameter
    used: list[str]  # ids of the calibration measurements the fit saw
    median_error_before: float  # in-sample, with the prior profile
    median_error_after: float  # in-sample, with the fitted profile
    evaluations: int  # simulator runs spent (one per measurement and parameter setting)


def _device_names(profile: Mapping[str, Any]) -> dict[str, str]:
    roles = {"host": profile["host"]}
    gpus = [d["name"] for d in profile["devices"] if d["kind"] == "GPU"]
    if gpus:
        roles["gpu"] = gpus[0]
    return roles


def _scaled(profile: Mapping[str, Any], scales: Mapping[str, float]) -> dict[str, Any]:
    out = copy.deepcopy(dict(profile))
    roles = _device_names(profile)
    for path, scale in scales.items():
        role, field = path.split(".")
        device = next(d for d in out["devices"] if d["name"] == roles[role])
        device[field] = device[field] * scale
    return out


def fit_profile(measurements: Iterable[Measurement], machine: str, profile: Mapping[str, Any], *,
                parameters: Sequence[str] = FIT_PARAMETERS, span: float = 16.0, sweeps: int = 2,
                regularisation: float = 1e-3, tokens: int = 1, warmup_tokens: int = 0) -> ProfileFit:  # fmt: skip
    """Tune ``profile`` so the simulator reproduces ``machine``'s calibration measurements.

    Only measurements of ``machine`` in the **calibration** split are used;
    validation measurements are never looked at, whatever is passed in.  Each
    parameter is a multiplicative correction of one prior value, searched within
    ``[1/span, span]`` by coordinate descent on the mean squared log error.  A
    small penalty keeps a parameter at its prior when the measurements do not
    constrain it (for example the GPU's rates when nothing was offloaded).
    One simulated token per measurement is enough: a static layer split has no
    warm-up, and its first token is within 0.02% of the steady state.
    """
    used = [m for m in measurements if m.machine == machine and m.split == "calibration"]
    if not used:
        raise ValueError(f"no calibration-split measurements for machine {machine!r}")
    roles = _device_names(profile)
    names = [p for p in parameters if p.split(".")[0] in roles]
    unknown = [p for p in parameters if p not in FIT_PARAMETERS]
    if unknown:
        raise ValueError(f"cannot fit {unknown}; choose from {list(FIT_PARAMETERS)}")
    limit = math.log(span)
    simulated: dict[tuple, float] = {}

    def prediction(m: Measurement, logs: Sequence[float]) -> float:
        key = (m.id, *(round(v, 6) for v in logs))
        if key not in simulated:
            topology = Topology.from_dict(_scaled(profile, {n: math.exp(v) for n, v in zip(names, logs)}))
            value = predict(m, {machine: topology}, tokens=tokens, warmup_tokens=warmup_tokens)
            if value is None or value <= 0:
                raise ValueError(f"measurement {m.id!r} cannot be simulated on this profile")
            simulated[key] = value
        return simulated[key]

    # Which parameters can change each prediction (probed at both ends of the search range):
    # a measurement is re-simulated only when one of *its* parameters moves.
    zero = [0.0] * len(names)
    depends: dict[str, list[int]] = {}
    for m in used:
        base = prediction(m, zero)
        depends[m.id] = [i for i in range(len(names)) if any(
            abs(prediction(m, [*zero[:i], end, *zero[i + 1:]]) - base) > 1e-9 * base for end in (-limit, limit))]

    def relevant(m: Measurement, logs: Sequence[float]) -> list[float]:
        return [v if i in depends[m.id] else 0.0 for i, v in enumerate(logs)]

    def loss(logs: Sequence[float]) -> tuple[float, list[float]]:
        errors, total = [], 0.0
        for m in used:
            predicted = prediction(m, relevant(m, logs))
            errors.append(abs(predicted - m.tokens_per_s) / m.tokens_per_s)
            total += math.log(predicted / m.tokens_per_s) ** 2
        return total / len(used) + regularisation * sum(v * v for v in logs), errors

    logs = [0.0] * len(names)
    before = sorted(loss(logs)[1])
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for sweep in range(sweeps):
        for i in range(len(names)):
            def along(value: float, i: int = i) -> float:
                return loss([*logs[:i], value, *logs[i + 1:]])[0]

            if sweep == 0:
                # coarse grid first (the error need not be unimodal), then golden-section refinement
                grid = [limit * (k - 4) / 4 for k in range(9)]
                best = min([*grid, logs[i]], key=lambda v: (along(v), abs(v)))
                lo, hi = max(best - limit / 4, -limit), min(best + limit / 4, limit)
            else:  # later sweeps only refine around the current value (the others have moved)
                best = logs[i]
                lo, hi = max(best - limit / 8, -limit), min(best + limit / 8, limit)
            for _step in range(9):
                a, b = hi - golden * (hi - lo), lo + golden * (hi - lo)
                if along(a) <= along(b):
                    hi = b
                else:
                    lo = a
            logs[i] = min([best, (lo + hi) / 2], key=lambda v: (along(v), abs(v)))
    after = sorted(loss(logs)[1])
    scales = {n: math.exp(v) for n, v in zip(names, logs)}
    fitted = copy.deepcopy(dict(profile))
    for path, scale in scales.items():
        role, field = path.split(".")
        prior = next(d for d in profile["devices"] if d["name"] == roles[role])[field]
        fitted = apply_fits(fitted, roles[role], **{field: prior * scale})
    record = fitted["_calibration"]
    record["fitted_from"] = sorted(m.id for m in used)
    record["scales_vs_prior"] = {k: round(v, 4) for k, v in scales.items()}
    record["split"] = "calibration"

    def median(values: list[float]) -> float:
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    return ProfileFit(fitted, scales, record["fitted_from"], median(before), median(after), len(simulated))
