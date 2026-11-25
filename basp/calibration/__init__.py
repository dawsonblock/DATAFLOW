"""Calibration: measurements, parameter fitting and predictive-accuracy checks.

Roadmap phase 1 replaces assumed simulator parameters with measured ones.  This package holds
the parts that do not depend on any machine: measurement records with a declared
calibration/validation split, fits of the simulator's transfer and roofline models to
micro-benchmark timings, the error statistics the hardware gate uses, and helpers for recorded
router decisions.  The scripts that produce the measurements live in ``benchmarks/``.
"""

from basp.calibration.coverage import REGIMES, coverage, regime_covered, regimes_of
from basp.calibration.curves import PerformanceCurve, validate_curve
from basp.calibration.fit import (
    RooflineFit,
    TransferFit,
    apply_fits,
    apply_link_fit,
    fit_roofline,
    fit_transfer,
)
from basp.calibration.measurements import (
    KINDS,
    SPLITS,
    Measurement,
    MeasurementSet,
    find_problems,
)
from basp.calibration.routing import build_trace, logits_to_routing, trace_statistics
from basp.calibration.validate import (
    CalibrationReport,
    ErrorStats,
    error_stats,
    evaluate_by_machine,
    evaluate_calibration,
)

__all__ = [
    "KINDS",
    "REGIMES",
    "SPLITS",
    "CalibrationReport",
    "ErrorStats",
    "Measurement",
    "MeasurementSet",
    "PerformanceCurve",
    "RooflineFit",
    "TransferFit",
    "apply_fits",
    "apply_link_fit",
    "build_trace",
    "coverage",
    "error_stats",
    "evaluate_by_machine",
    "evaluate_calibration",
    "find_problems",
    "fit_roofline",
    "fit_transfer",
    "logits_to_routing",
    "regime_covered",
    "regimes_of",
    "trace_statistics",
    "validate_curve",
]
