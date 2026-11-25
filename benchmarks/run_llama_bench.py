"""Run llama.cpp's ``llama-bench`` over a matrix of models and offload settings and record
the results as BASP measurements.

    python -m benchmarks.run_llama_bench --machine i9-11900kf-3080ti --split calibration \
        --model-key qwen3-30b-a3b=/models/Qwen3-30B-A3B-Q4_K_M.gguf \
        --ngl 0 --ngl 99 --n-cpu-moe 28 --out measurements.json

Every run must declare ``--split`` (there is no default): a benchmark belongs either to the
calibration split, which may be used to tune simulator parameters, or to the validation split,
which is used only to judge predictive accuracy.  The output file refuses to hold the same
benchmark (machine, model, bits per weight, ``-ngl``, threads, kind) in both splits, so plan the
matrix before running: for example calibrate on ``-ngl 0`` and ``-ngl 99`` and validate on the
partial offloads, or calibrate on one model and validate on another.

The effective bits per weight of each file (tensor data in bits divided by the parameter count)
is read from its GGUF header and recorded with each measurement, so the simulator models the same
weight traffic; the header also fingerprints the file and is checked against the simulator's shape
for the model name.  A configuration that fails (typically: does not fit GPU memory) is recorded
and the matrix continues.

NOT exercised in the development container: running ``llama-bench`` itself (no llama.cpp, no
GPU, no model files there).  The JSON format was checked against the llama-bench source vendored
in llama-cpp-python 0.3.38 (``tools/llama-bench/llama-bench.cpp``): one array of objects whose
numeric fields are JSON numbers, prompt processing and generation as separate records, fields
including ``n_gpu_layers``, ``n_cpu_moe``, ``n_threads``, ``n_ubatch``, ``n_prompt``, ``n_gen``,
``n_depth``, ``avg_ts`` and ``stddev_ts``.  A newer llama.cpp may differ; the parser fails loudly,
printing the keys it received, if a required one is missing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from basp.calibration.measurements import Measurement, MeasurementSet
from basp.evidence import file_identity
from basp.models.gguf import GGUFError, check_against, read_gguf
from basp.workloads.zoo import MODELS

REQUIRED_KEYS = ("n_gpu_layers", "n_prompt", "n_gen", "avg_ts")


def extract_json(text: str) -> Any:
    """The JSON document in ``llama-bench`` output, tolerating log lines around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("llama-bench output contains no JSON array; first 200 characters: "
                             f"{text[:200]!r}") from None  # fmt: skip
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"llama-bench output is not valid JSON ({exc})") from None


def parse_llama_bench(json_text: str, *, machine: str, split: str, model: str,
                      bits_per_weight: float, skipped: list[str] | None = None,
                      provenance: dict | None = None) -> list[Measurement]:  # fmt: skip
    """Convert ``llama-bench -o json`` output into measurements.

    A record with ``n_gen == 0`` is prompt processing (``prefill``: ``avg_ts`` is prompt tokens
    per second, processed ``min(n_prompt, n_batch)`` at a time); one with ``n_prompt == 0`` is
    generation (``decode``).  Records with both set (llama-bench ``-pg``) mix the two phases and
    are skipped; their descriptions are appended to ``skipped`` when a list is given.
    ``n_gpu_layers`` is recorded as reported, so "all layers" is whatever large number was passed
    to ``-ngl``.
    """
    records = extract_json(json_text)
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("expected a JSON array of result objects from llama-bench")
    out: list[Measurement] = []
    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        missing = [key for key in REQUIRED_KEYS if key not in record]
        if missing:
            raise ValueError(
                f"llama-bench record {index} lacks the expected key(s) {missing}; this parser "
                f"assumes the format described in its docstring. Keys received: {sorted(record)}"
            )
        n_prompt, n_gen = int(record["n_prompt"]), int(record["n_gen"])
        ngl, threads = int(record["n_gpu_layers"]), int(record.get("n_threads", 0) or 0)
        if n_prompt > 0 and n_gen > 0:
            if skipped is not None:
                skipped.append(f"record {index}: mixed prompt+generation test "
                               f"(n_prompt={n_prompt}, n_gen={n_gen})")  # fmt: skip
            continue
        if n_prompt == 0 and n_gen == 0:
            raise ValueError(f"llama-bench record {index} has n_prompt == n_gen == 0")
        kind = "prefill" if n_gen == 0 else "decode"
        depth = int(record.get("n_depth", 0) or 0)
        cpu_moe = int(record.get("n_cpu_moe", 0) or 0)
        batch = 1
        if kind == "prefill":
            # one forward pass processes the physical batch: n_ubatch, never more than n_batch
            sizes = [int(record[k]) for k in ("n_ubatch", "n_batch") if record.get(k)]
            batch = min([n_prompt, *sizes])
            context = depth  # the prompt's own tokens are added by the simulator
        else:
            context = depth + n_gen // 2  # average KV length over the generated tokens
        base = (f"{machine}/{model}/b{bits_per_weight:g}/ngl{ngl}/cmoe{cpu_moe}/t{threads}/"
                f"d{depth}/{kind}-p{n_prompt}-g{n_gen}")  # fmt: skip
        seen[base] = seen.get(base, 0) + 1
        commit = record.get("build_commit")
        out.append(Measurement(
            id=base if seen[base] == 1 else f"{base}#{seen[base]}",
            machine=machine,
            split=split,
            kind=kind,
            model=model,
            tokens_per_s=float(record["avg_ts"]),
            bits_per_weight=float(bits_per_weight),
            n_gpu_layers=ngl,
            n_cpu_moe=cpu_moe,
            context_len=context,
            batch=batch,
            threads=threads,
            stddev_tokens_per_s=float(record.get("stddev_ts", 0.0) or 0.0),
            source=f"llama-bench {commit}" if commit else "llama-bench",
            notes=f"n_prompt={n_prompt} n_gen={n_gen} n_depth={depth} file={record.get('model_filename', '')}",
            provenance={**(provenance or {}), "llama_bench_build": commit,
                        "flash_attn": record.get("flash_attn"), "cpu_info": record.get("cpu_info"),
                        "gpu_info": record.get("gpu_info"), "backends": record.get("backends")},
        ))  # fmt: skip
    return out


def bench_command(binary: str, model_path: str, ngl: int, *, threads: int | None, repetitions: int,
                  n_prompt: int | None, n_gen: int | None, extra: list[str], depth: int | None = None,
                  n_cpu_moe: int | None = None) -> list[str]:  # fmt: skip
    command = [binary, "-m", model_path, "-ngl", str(ngl), "-o", "json", "-r", str(repetitions)]
    if threads:
        command += ["-t", str(threads)]
    if n_prompt is not None:
        command += ["-p", str(n_prompt)]
    if n_gen is not None:
        command += ["-n", str(n_gen)]
    if depth:
        command += ["-d", str(depth)]
    if n_cpu_moe:
        command += ["-ncmoe", str(n_cpu_moe)]
    return command + extra


def add_unique(target: MeasurementSet, measurement: Measurement) -> Measurement:
    """Add a measurement, numbering its id if the same benchmark was already recorded (a repeat)."""
    candidate, n = measurement, 1
    while candidate.id in target:
        n += 1
        candidate = Measurement(**{**measurement.to_json(), "id": f"{measurement.id}@run{n}"})
    return target.add(candidate)


@dataclass
class BenchModel:
    """A model file to benchmark: its measurement name, path, size per weight and fingerprint."""

    name: str
    path: str
    bits_per_weight: float
    fingerprint: dict[str, Any]
    shape_check: dict[str, Any] | None = None


def resolve_model(name: str, path: str, *, bits_per_weight: float | None = None,
                  check_shape: bool = True) -> BenchModel:  # fmt: skip
    """Read the GGUF header: exact bits per weight, fingerprint, and agreement with the simulator's shape.

    A file whose shape does not match the simulator's shape for ``name`` would calibrate the
    simulator against a different model; that is refused unless ``check_shape`` is false.
    """
    if not Path(path).exists():
        raise SystemExit(f"model file {path} does not exist")
    try:
        gguf = read_gguf(path)
    except GGUFError as exc:
        if bits_per_weight is None:
            raise SystemExit(f"{path}: {exc}; pass --bits-per-weight to benchmark it anyway") from None
        return BenchModel(name, path, bits_per_weight, {"file": Path(path).name, "error": str(exc)})
    measured_bits = round(gguf.bits_per_weight, 4)
    check = None
    if name in MODELS:
        check = check_against(gguf, MODELS[name])
        if check_shape and not check["ok"]:
            raise SystemExit(f"{path} does not look like {name}: " + "; ".join(check["problems"])
                             + ". Fix the NAME=PATH pairing, or pass --no-shape-check.")  # fmt: skip
    elif check_shape:
        raise SystemExit(f"{name!r} is not a model shape the simulator knows ({', '.join(sorted(MODELS))})")
    if bits_per_weight is not None and abs(bits_per_weight - measured_bits) > 0.01:
        print(f"note: {Path(path).name} has {measured_bits} bits per weight; using the given {bits_per_weight}")
    # the header identifies the shape; only the full content identifies the weights
    fingerprint = {**gguf.fingerprint(), "sha256": file_identity(path)["sha256"]}
    return BenchModel(name, path, bits_per_weight if bits_per_weight is not None else measured_bits,
                      fingerprint, check)  # fmt: skip


def run_one(binary: str, model: BenchModel, *, machine: str, split: str, ngl: int, target: MeasurementSet,
            threads: int | None = None, repetitions: int = 3, n_prompt: int | None = None,
            n_gen: int | None = None, depth: int | None = None, n_cpu_moe: int | None = None,
            extra: list[str] | None = None, provenance: dict[str, Any] | None = None,
            timeout_s: float | None = None) -> dict[str, Any]:  # fmt: skip
    """Run one llama-bench configuration and add its measurements to ``target``.

    Returns a record of what happened.  A configuration that fails (most often: the model does
    not fit GPU memory at this offload) is recorded as a failure, not an error, so a matrix
    keeps going; llama-bench itself stops at the first model that fails to load.
    """
    command = bench_command(binary, model.path, ngl, threads=threads, repetitions=repetitions,
                            n_prompt=n_prompt, n_gen=n_gen, extra=list(extra or []), depth=depth,
                            n_cpu_moe=n_cpu_moe)  # fmt: skip
    record: dict[str, Any] = {"model": model.name, "split": split, "ngl": ngl, "n_cpu_moe": n_cpu_moe or 0,
                              "depth": depth or 0, "threads": threads, "command": command}  # fmt: skip
    print("running:", " ".join(command), flush=True)
    try:
        done = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {**record, "status": "timeout"}
    if done.returncode != 0:
        tail = done.stderr[-1500:]
        print(f"  failed ({done.returncode}); recorded and continuing. Last output:\n{tail}")
        return {**record, "status": "failed", "returncode": done.returncode, "stderr_tail": tail}
    skipped: list[str] = []
    added = []
    for measurement in parse_llama_bench(done.stdout, machine=machine, split=split, model=model.name,
                                         bits_per_weight=model.bits_per_weight, skipped=skipped,
                                         provenance={**(provenance or {}), "model_file": model.fingerprint}):  # fmt: skip
        stored = add_unique(target, measurement)
        added.append(stored.id)
        print(f"  {stored.id}: {stored.tokens_per_s:.2f} tokens/s ({stored.kind})")
    for note in skipped:
        print(f"  skipped {note}")
    return {**record, "status": "ok", "measurements": added, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--machine", required=True, help="label of this machine")
    parser.add_argument("--split", required=True, choices=["calibration", "validation"],
                        help="declared before running; never chosen after seeing results")  # fmt: skip
    parser.add_argument("--model-key", action="append", required=True, metavar="NAME=PATH",
                        help="measurement model name and GGUF path (repeatable)")  # fmt: skip
    parser.add_argument("--ngl", action="append", type=int, required=True,
                        help="number of layers to offload (repeatable)")  # fmt: skip
    parser.add_argument("--bits-per-weight", type=float,
                        help="override the bits per weight read from the GGUF header")  # fmt: skip
    parser.add_argument("--out", required=True, help="measurement set to create or append to")
    parser.add_argument("--threads", action="append", type=int, help="thread count (repeatable)")
    parser.add_argument("--depth", action="append", type=int, help="llama-bench -d: KV depth (repeatable)")
    parser.add_argument("--n-cpu-moe", action="append", type=int,
                        help="llama-bench -ncmoe: expert weights of the first N layers on the CPU (repeatable)")  # fmt: skip
    parser.add_argument("--repetitions", type=int, default=3, help="llama-bench -r")
    parser.add_argument("--n-prompt", type=int, help="llama-bench -p (default: its own default)")
    parser.add_argument("--n-gen", type=int, help="llama-bench -n (default: its own default)")
    parser.add_argument("--llama-bench", default="llama-bench", help="path of the llama-bench binary")
    parser.add_argument("--extra", action="append", default=[], help="extra argument passed through")
    parser.add_argument("--no-shape-check", action="store_true",
                        help="benchmark a file whose shape differs from the simulator's model")  # fmt: skip
    parser.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = parser.parse_args(argv)

    pairs: list[tuple[str, str]] = []
    for item in args.model_key:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"--model-key expects NAME=PATH, got {item!r}")
        pairs.append((name, path))
    grid = [(ngl, threads, depth, cmoe) for ngl in args.ngl for threads in (args.threads or [None])
            for depth in (args.depth or [None]) for cmoe in (args.n_cpu_moe or [None])]  # fmt: skip
    if args.dry_run:
        for _, path in pairs:
            for ngl, threads, depth, cmoe in grid:
                print(" ".join(bench_command(args.llama_bench, path, ngl, threads=threads,
                                             repetitions=args.repetitions, n_prompt=args.n_prompt,
                                             n_gen=args.n_gen, extra=args.extra, depth=depth, n_cpu_moe=cmoe)))  # fmt: skip
        return 0
    binary = shutil.which(args.llama_bench)
    if binary is None:
        raise SystemExit(f"cannot find the llama.cpp benchmark binary {args.llama_bench!r}: build "
                         "llama.cpp and pass --llama-bench /path/to/llama-bench")  # fmt: skip
    models = [resolve_model(name, path, bits_per_weight=args.bits_per_weight,
                            check_shape=not args.no_shape_check) for name, path in pairs]  # fmt: skip
    out = Path(args.out)
    target = MeasurementSet.load(out) if out.exists() else MeasurementSet()
    records = []
    for model in models:
        for ngl, threads, depth, cmoe in grid:
            records.append(run_one(binary, model, machine=args.machine, split=args.split, ngl=ngl,
                                   target=target, threads=threads, repetitions=args.repetitions,
                                   n_prompt=args.n_prompt, n_gen=args.n_gen, depth=depth, n_cpu_moe=cmoe,
                                   extra=args.extra))  # fmt: skip
            target.save(out)  # after every run, so an interrupted matrix keeps its results
    failed = [r for r in records if r["status"] != "ok"]
    print(f"{len(records) - len(failed)} configuration(s) ran, {len(failed)} failed; {out} now holds "
          f"{len(target)} ({len(target.calibration)} calibration, {len(target.validation)} validation)")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
