"""Run the pre-registered calibration plan on this machine, then fit and validate its profile.

    python -m benchmarks.run_calibration --machine i9-11900kf-3080ti --models-dir ~/models \
        --llama-bench ~/llama.cpp/build/bin/llama-bench
    python -m benchmarks.run_calibration --machine macbook-m2 --models-dir ~/models --dry-run

One command for the whole procedure in ``docs/calibration.md``:

1. reads ``configs/calibration_plan.json`` -- which benchmark belongs to which split was fixed
   there before any result existed;
2. checks every model file: GGUF header readable, shape equal to the simulator's model,
   exact bits per weight;
3. runs every planned llama-bench configuration, appending to
   ``evidence/<machine>/measurements.json`` after each one.  A configuration that fails (for
   example: does not fit GPU memory) is recorded in ``runs.json`` and the plan continues.
   Re-running resumes: configurations already recorded with the same repetitions, the same
   llama-bench executable and the same model weights (full-content SHA-256) are skipped; a
   rebuilt executable or changed weights re-runs them, and the older measurements are then
   excluded from fitting and scoring (``identity.json`` records the current identities);
4. records the machine (CPU, GPU, driver, memory, llama.cpp build) in ``environment.json``;
5. fits the machine's profile on the calibration split only and scores the validation split
   (``calibration.json``, ``profile_fitted.json``).  With ``--update-profile`` the fitted
   profile replaces the prior in ``configs/``.

Commit the ``evidence/<machine>/`` directory (and the updated profile).  Nothing here can
make a validation failure look like a pass: the split comes from the plan, and the verdict
from held-out measurements only.

NOT exercised in the development container (no GPU, llama.cpp or model files there); the
plan handling, resumption and fitting are tested with a fake llama-bench.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from basp.calibration import MeasurementSet
from basp.calibration.plan import plan_runs
from basp.core.topology import Topology
from basp.evidence import (
    CONFIG_DIR,
    IN_CHECKOUT,
    REPO_DIR,
    environment,
    file_identity,
    git_state,
    profile_identity,
    sha256_file,
    work_dir,
)
from basp.experiments.calibrate import fit_profile, score, staleness
from basp.experiments.profiles import load_json
from benchmarks.run_llama_bench import BenchModel, resolve_model, run_one

PLAN = CONFIG_DIR / "calibration_plan.json"


def run_key(run: dict[str, Any], identity: dict[str, Any] | None = None) -> tuple:
    """What makes two runs the same run: the planned configuration, its repetitions, and the
    exact executable and model weights that execute it."""
    identity = identity or {}
    return (run["split"], run["model"], run["ngl"], run["depth"], run["n_cpu_moe"], run["n_prompt"], run["n_gen"],
            run["repetitions"], identity.get("executable_sha256"), (identity.get("models") or {}).get(run["model"]))  # fmt: skip


def _command(args: list[str]) -> str | None:
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return (done.stdout.strip() or None) if done.returncode == 0 else None


def machine_facts(llama_bench: str | None) -> dict[str, Any]:
    """What this machine is: recorded with the measurements, never used to compute anything."""
    facts: dict[str, Any] = {"platform": platform.platform(), "processor": platform.processor(),
                             "software": environment(), "repository": git_state()}  # fmt: skip
    query = "--query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max"
    facts["gpu"] = _command(["nvidia-smi", query, "--format=csv,noheader"])
    if sys.platform == "darwin":
        facts["cpu"] = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
        facts["memory_bytes"] = _command(["sysctl", "-n", "hw.memsize"])
    elif Path("/proc/cpuinfo").exists():
        names = [line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")]  # fmt: skip
        facts["cpu"] = names[0] if names else None
        facts["logical_cpus"] = len(names)
        meminfo = Path("/proc/meminfo").read_text().splitlines() if Path("/proc/meminfo").exists() else []
        facts["memory_kib"] = next((line.split()[1] for line in meminfo if line.startswith("MemTotal")), None)
    if llama_bench:
        facts["llama_bench"] = {"path": llama_bench, "version": _command([llama_bench, "--version"])}
    return facts


def find_models(plan: dict[str, Any], runs: list[dict[str, Any]], models_dir: Path | None,
                overrides: dict[str, str], *, check_shape: bool) -> dict[str, BenchModel]:  # fmt: skip
    needed = sorted({r["model"] for r in runs})
    found: dict[str, BenchModel] = {}
    missing = []
    for name in needed:
        path = overrides.get(name)
        if path is None and models_dir is not None:
            candidate = models_dir / plan["models"][name]["file"]
            path = str(candidate) if candidate.exists() else None
        if path is None:
            missing.append(name)
            continue
        found[name] = resolve_model(name, path, check_shape=check_shape)
        fp = found[name].fingerprint
        print(f"{name}: {Path(path).name}, {fp.get('parameters', 0) / 1e9:.2f} B parameters, "
              f"{found[name].bits_per_weight:.3f} bits per weight, {fp.get('file_type')}")  # fmt: skip
    if missing:
        lines = [f"  {name}: {plan['models'][name]['file']} (e.g. from the {plan['models'][name]['repository']} "
                 "repository on Hugging Face)" for name in missing]  # fmt: skip
        raise SystemExit("model files not found:\n" + "\n".join(lines) + "\nPut them in --models-dir or pass "
                         "--model NAME=PATH. Any GGUF of the same model works: its shape is checked from the "
                         "header, and its exact bits per weight are used.")  # fmt: skip
    return found


def fit_and_score(evidence: Path, machine: str, profile_name: str) -> dict[str, Any]:
    """Fit on the calibration split, score the validation split; ``profile_name`` is a configs/
    name or a path (e.g. a prior seeded by ``benchmarks.profile_hardware``)."""
    measurements = MeasurementSet.load(evidence / "measurements.json")
    identity_file = evidence / "identity.json"
    stale = staleness({machine: json.loads(identity_file.read_text())} if identity_file.exists() else {})
    own = [m for m in measurements if m.machine == machine]
    current = [m for m in own if stale(m) is None]
    thresholds = load_json("hardware_gate")
    prior = load_json(profile_name)
    limits = {"median_max": thresholds["calibration_median_max"], "p95_max": thresholds["calibration_p95_max"],
              "min_validation": thresholds["min_validation_measurements"], "exclude": stale}  # fmt: skip
    before = score(MeasurementSet(own), {machine: Topology.from_dict(prior)}, **limits)
    summary: dict[str, Any] = {"prior_profile": str(profile_name), "prior_profile_sha": profile_identity(prior),
                               "prior": before.to_json(), "excluded_stale": len(own) - len(current)}  # fmt: skip
    if any(m.split == "calibration" for m in current):
        fit = fit_profile(current, machine, prior)
        after = score(MeasurementSet(own), {machine: Topology.from_dict(fit.profile)}, **limits)
        (evidence / "profile_fitted.json").write_text(json.dumps(fit.profile, indent=1) + "\n")
        summary.update(fitted=after.to_json(), scales=fit.scales, fitted_from=fit.used,
                       fitted_profile_sha=profile_identity(fit.profile))  # fmt: skip
    (evidence / "calibration.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", required=True, help="machine label in the plan")
    parser.add_argument("--models-dir", type=Path, help="directory holding the plan's GGUF files")
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                        help="explicit model file for a plan model (repeatable)")  # fmt: skip
    parser.add_argument("--llama-bench", default="llama-bench", help="path of the llama-bench binary")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--prior", help="profile to start the fit from: a configs/ name or a path (default: the "
                                        "plan's profile; e.g. one seeded by benchmarks.profile_hardware)")
    parser.add_argument("--evidence", type=Path, help="output directory (default: evidence/<machine>)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per llama-bench run")
    parser.add_argument("--update-profile", action="store_true",
                        help="write the fitted profile over the machine's prior in configs/")  # fmt: skip
    parser.add_argument("--no-shape-check", action="store_true")
    parser.add_argument("--fit-only", action="store_true", help="skip benchmarking; fit and score what exists")
    parser.add_argument("--dry-run", action="store_true", help="list the planned runs and exit")
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text())
    runs = plan_runs(plan, args.machine)
    profile_name = args.prior or plan["machines"][args.machine]["profile"]
    evidence = args.evidence or work_dir() / "evidence" / args.machine
    n_cal = sum(r["split"] == "calibration" for r in runs)
    print(f"plan v{plan['version']} ({sha256_file(args.plan)[:12]}): {len(runs)} configurations for "
          f"{args.machine} ({n_cal} calibration, {len(runs) - n_cal} validation)")  # fmt: skip
    if args.dry_run:
        for r in runs:
            extra = (f" -d {r['depth']}" if r["depth"] else "") + (f" -ncmoe {r['n_cpu_moe']}" if r["n_cpu_moe"] else "")
            print(f"  {r['split']:11s} {r['model']:14s} -ngl {r['ngl']}{extra}")
        return 0
    evidence.mkdir(parents=True, exist_ok=True)
    if not args.fit_only:
        binary = shutil.which(args.llama_bench)
        if binary is None:
            raise SystemExit(f"cannot find llama-bench ({args.llama_bench!r}); build llama.cpp and pass --llama-bench")
        overrides = dict(item.split("=", 1) for item in args.model)
        models = find_models(plan, runs, args.models_dir, overrides, check_shape=not args.no_shape_check)
        (evidence / "environment.json").write_text(json.dumps(machine_facts(binary), indent=1) + "\n")
        identity = {"executable": file_identity(binary), "executable_sha256": file_identity(binary)["sha256"],
                    "models": {name: m.fingerprint.get("sha256") for name, m in sorted(models.items())},
                    "plan_sha256": sha256_file(args.plan)}  # fmt: skip
        (evidence / "identity.json").write_text(json.dumps(identity, indent=1) + "\n")
        out = evidence / "measurements.json"
        log_path = evidence / "runs.json"
        target = MeasurementSet.load(out) if out.exists() else MeasurementSet()
        log = json.loads(log_path.read_text()) if log_path.exists() else []
        done = {tuple(entry["key"]) for entry in log}
        provenance = {"plan_version": plan["version"], "plan_sha256": sha256_file(args.plan),
                      "executable_sha256": identity["executable_sha256"]}  # fmt: skip
        for i, run in enumerate(runs, 1):
            key = run_key(run, identity)
            if key in done:
                continue
            print(f"[{i}/{len(runs)}] ", end="")
            record = run_one(binary, models[run["model"]], machine=args.machine, split=run["split"],
                             ngl=run["ngl"], target=target, repetitions=run["repetitions"],
                             n_prompt=run["n_prompt"], n_gen=run["n_gen"], depth=run["depth"] or None,
                             n_cpu_moe=run["n_cpu_moe"] or None, provenance=provenance,
                             timeout_s=args.timeout)  # fmt: skip
            log.append({"key": list(key), **record})
            target.save(out)
            log_path.write_text(json.dumps(log, indent=1) + "\n")
        current_keys = {run_key(r, identity) for r in runs}
        log_now = [e for e in log if tuple(e["key"]) in current_keys]
        failed = [e for e in log_now if e["status"] != "ok"]
        print(f"{len(log_now) - len(failed)} of {len(log_now)} configurations ran; {len(failed)} failed (see runs.json)")
    summary = fit_and_score(evidence, args.machine, profile_name)
    verdict = summary.get("fitted", summary["prior"])
    v = verdict["validation"]
    print(f"validation after fitting: {v['n']} measurements, median {v['median']:.1%}, p95 {v['p95']:.1%} -> "
          f"{verdict['status'].replace('_', ' ').upper()}")  # fmt: skip
    for reason in verdict.get("reasons", []):
        print(f"  {reason}")
    if args.update_profile and "fitted" in summary:
        if verdict["status"] != "pass":
            raise SystemExit("validation did not pass: the fitted profile is not written to configs/ "
                             "(it is in the evidence folder for diagnosis)")
        if not IN_CHECKOUT:
            raise SystemExit("--update-profile writes configs/ of a source checkout; run it from a checkout "
                             f"(the fitted profile is in {evidence})")  # fmt: skip
        target_profile = CONFIG_DIR / f"{plan['machines'][args.machine]['profile']}.json"
        shutil.copyfile(evidence / "profile_fitted.json", target_profile)
        print(f"updated {target_profile.relative_to(REPO_DIR)}")
    print(f"wrote {evidence.relative_to(REPO_DIR) if evidence.is_relative_to(REPO_DIR) else evidence}/; commit it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
