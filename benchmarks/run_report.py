"""Run every experiment and write the simulation report.

    python -m benchmarks.run_report            # full report  -> reports/v0.1/
    python -m benchmarks.run_report --quick    # smoke run    -> reports/quick/

The report files are deterministic; wall-clock timing is printed here only.
"""

from __future__ import annotations

import argparse
import time

from basp.experiments.report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="few tokens and fewer sweep points")
    parser.add_argument("--workload", default="workload_qwen3_decode", help="name in configs/ or a JSON path")
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args()
    out = args.out or ("reports/quick" if args.quick else "reports/v0.1")
    t0 = time.perf_counter()

    def progress(stage: str) -> None:
        print(f"[{time.perf_counter() - t0:6.1f} s] {stage}", flush=True)

    md, results = write_report(out, args.workload, quick=args.quick, progress=progress)
    print(f"wrote {md} and {results} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
