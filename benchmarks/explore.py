"""Search the architectural design space with the simulator in the loop.

    python -m benchmarks.explore --budget 2000 --workers 4      # full exploration
    python -m benchmarks.explore --budget 60 --method bayesian  # small, sequential

Writes every evaluated design, the Pareto front (throughput, tokens per joule,
estimated cost) and the front re-evaluated at the pessimistic and optimistic
corners to a JSON file that the report generator reads.  All module parameters
and all prices are estimates; see configs/design_estimates.json.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from basp.dse.search import evolutionary, search
from basp.experiments.explore import (
    DEFAULT_WORKLOAD,
    OBJECTIVES,
    design_space,
    evaluate_corners,
    evaluate_design,
    seed_designs,
)
from basp.workloads.zoo import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=int, default=2000, help="designs to evaluate")
    parser.add_argument("--method", default="evolutionary",
                        choices=["auto", "exhaustive", "evolutionary", "bayesian", "random"])  # fmt: skip
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_WORKLOAD["model"], choices=sorted(MODELS))
    parser.add_argument("--base-profile", default="desktop_gpu12",
                        help="host/GPU profile every design is built on (a calibrated machine profile)")  # fmt: skip
    parser.add_argument("--out", default="reports/v0.2/dse.json")
    args = parser.parse_args()

    workload = dict(DEFAULT_WORKLOAD, model=args.model)
    spec = MODELS[args.model]
    space = design_space(spec.bytes_at() / 2**30)
    evaluate = functools.partial(evaluate_design, workload=workload, base_profile=args.base_profile)
    corners_of = functools.partial(evaluate_corners, workload=workload, base_profile=args.base_profile)
    t0 = time.perf_counter()

    def run(map_fn):
        if args.method == "evolutionary":
            # seeded so that the conventional baseline and every architecture are always judged
            population = max(8, min(64, round(args.budget**0.5)))
            return evolutionary(space, evaluate, OBJECTIVES, population=population,
                                generations=max(1, args.budget // population - 1), seed=args.seed,
                                initial=seed_designs(space), map_fn=map_fn)  # fmt: skip
        return search(space, evaluate, OBJECTIVES, method=args.method, budget=args.budget,
                      seed=args.seed, **({} if args.method == "bayesian" else {"map_fn": map_fn}))  # fmt: skip

    if args.workers > 1 and args.method != "bayesian":
        with ProcessPoolExecutor(args.workers) as pool:
            result = run(functools.partial(pool.map, chunksize=1))
            corners = list(pool.map(corners_of, [e.config for e in result.front]))
    else:
        result = run(map)
        corners = [corners_of(e.config) for e in result.front]
    feasible = sum(1 for _ in space.enumerate())
    data = {
        "_note": "Simulated under configs/design_estimates.json; prices and module parameters "
                 "are estimates, not measurements.",
        "workload": workload,
        "base_profile": args.base_profile,
        "method": result.method,
        "seed": args.seed,
        "space_size": space.size,
        "feasible_designs": feasible,
        "evaluated": result.n_evaluated,
        "objectives": [{"name": o.name, "maximize": o.maximize} for o in OBJECTIVES],
        "evaluations": [{"config": e.config, "metrics": e.metrics, "feasible": e.feasible}
                        for e in result.evaluations],
        "front": [{"config": e.config, "metrics": e.metrics, "corners": c}
                  for e, c in zip(result.front, corners)],
    }  # fmt: skip
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"{result.n_evaluated} designs evaluated of {feasible} feasible ({space.size} raw) with "
          f"{result.method}; front has {len(result.front)} designs; {time.perf_counter() - t0:.0f} s; "
          f"wrote {out}")  # fmt: skip


if __name__ == "__main__":
    main()
