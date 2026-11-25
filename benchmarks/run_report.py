"""Run the experiments and write a simulation report.

    python -m benchmarks.run_report                  # v0.3 report -> reports/v0.3/
    python -m benchmarks.run_report --quick          # smoke run   -> reports/quick/
    python -m benchmarks.run_report --release v0.1   # the v0.1 experiment set

The report reads the design-space exploration results written by
``python -m benchmarks.explore`` (reports/v0.3/dse.json) and, if present, a
measurement file produced on the calibration machines (measurements.json).
After calibrating a machine, pass its profile with ``--base-profile`` (to this
script and to ``benchmarks.explore``) so the comparison runs on measured
host and GPU parameters.
Report files are deterministic; wall-clock timing is printed here only.
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="few tokens and fewer sweep points")
    parser.add_argument("--release", default="v0.3", choices=["v0.1", "v0.3"],
                        help="v0.1: the original experiment set (its report is kept for the record)")  # fmt: skip
    parser.add_argument("--workload", default="workload_qwen3_decode",
                        help="v0.1 only: name in configs/ or a JSON path")  # fmt: skip
    parser.add_argument("--dse", default="reports/v0.3/dse.json", help="exploration results")
    parser.add_argument("--trace", help="recorded routing trace (benchmarks.record_routing_trace *.decode.json)")
    parser.add_argument("--compression", help="compression profile of real weights (benchmarks.profile_compression)")
    parser.add_argument("--measurements", default="evidence",
                        help="measurement set from the calibration machines (may be absent)")  # fmt: skip
    parser.add_argument("--base-profile", default=None,
                        help="host/GPU profile the comparison is built on (default desktop_gpu12)")  # fmt: skip
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args()
    out = args.out or ("reports/quick" if args.quick else f"reports/{args.release}")
    t0 = time.perf_counter()

    def progress(stage: str) -> None:
        print(f"[{time.perf_counter() - t0:6.1f} s] {stage}", flush=True)

    if args.release == "v0.1":
        from basp.experiments.report import write_report

        md, results = write_report(out, args.workload, quick=args.quick, progress=progress)
    else:
        from basp.experiments.report_v3 import write_report

        md, results = write_report(out, quick=args.quick, progress=progress, dse_path=args.dse,
                                   measurements_path=args.measurements, base=args.base_profile,
                                   trace_path=args.trace, compression_path=args.compression)  # fmt: skip
    print(f"wrote {md} and {results} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
