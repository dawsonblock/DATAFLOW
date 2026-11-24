"""Run llama.cpp's ``llama-bench`` over a matrix of models and offload settings and record
the results as BASP measurements.

    python -m benchmarks.run_llama_bench --machine i9-11900kf-3080ti --split calibration \
        --model-key qwen3-30b-a3b=/models/Qwen3-30B-A3B-Q4_K_M.gguf \
        --ngl 0 --ngl 16 --ngl 99 --bits-per-weight 4.85 --out measurements.json

Every run must declare ``--split`` (there is no default): a benchmark belongs either to the
calibration split, which may be used to tune simulator parameters, or to the validation split,
which is used only to judge predictive accuracy.  The output file refuses to hold the same
benchmark (machine, model, bits per weight, ``-ngl``, threads, kind) in both splits, so plan the
matrix before running: for example calibrate on ``-ngl 0`` and ``-ngl 99`` and validate on the
partial offloads, or calibrate on one model and validate on another.

``--bits-per-weight`` is the effective size of the GGUF quantisation (file size in bits divided
by the parameter count); it is recorded with each measurement so the simulator models the same
weight traffic.

NOT exercised in the development container: running ``llama-bench`` itself (no llama.cpp, no
GPU, no model files there).  The JSON format is *assumed* from the field names ``llama-bench -o
json`` is believed to emit (``n_gpu_layers``, ``n_threads``, ``n_prompt``, ``n_gen``, ``avg_ts``,
``stddev_ts``, ...), with prompt processing and generation reported as separate records.  The
parser fails loudly, printing the keys it actually received, if that assumption is wrong.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from basp.calibration.measurements import Measurement, MeasurementSet

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
                      bits_per_weight: float,
                      skipped: list[str] | None = None) -> list[Measurement]:  # fmt: skip
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
        batch = 1
        if kind == "prefill":
            batch = min(n_prompt, int(record.get("n_batch", n_prompt) or n_prompt))
        base = (f"{machine}/{model}/b{bits_per_weight:g}/ngl{ngl}/t{threads}/"
                f"{kind}-p{n_prompt}-g{n_gen}")  # fmt: skip
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
            context_len=0,
            batch=batch,
            threads=threads,
            stddev_tokens_per_s=float(record.get("stddev_ts", 0.0) or 0.0),
            source=f"llama-bench {commit}" if commit else "llama-bench",
            notes=f"n_prompt={n_prompt} n_gen={n_gen} file={record.get('model_filename', '')}",
        ))  # fmt: skip
    return out


def bench_command(binary: str, model_path: str, ngl: int, *, threads: int | None, repetitions: int,
                  n_prompt: int | None, n_gen: int | None, extra: list[str]) -> list[str]:  # fmt: skip
    command = [binary, "-m", model_path, "-ngl", str(ngl), "-o", "json", "-r", str(repetitions)]
    if threads:
        command += ["-t", str(threads)]
    if n_prompt is not None:
        command += ["-p", str(n_prompt)]
    if n_gen is not None:
        command += ["-n", str(n_gen)]
    return command + extra


def add_unique(target: MeasurementSet, measurement: Measurement) -> Measurement:
    """Add a measurement, numbering its id if the same benchmark was already recorded (a repeat)."""
    candidate, n = measurement, 1
    while candidate.id in target:
        n += 1
        candidate = Measurement(**{**measurement.to_json(), "id": f"{measurement.id}@run{n}"})
    return target.add(candidate)


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
    parser.add_argument("--bits-per-weight", type=float, required=True,
                        help="effective bits per weight of the GGUF quantisation")  # fmt: skip
    parser.add_argument("--out", required=True, help="measurement set to create or append to")
    parser.add_argument("--threads", action="append", type=int, help="thread count (repeatable)")
    parser.add_argument("--repetitions", type=int, default=3, help="llama-bench -r")
    parser.add_argument("--n-prompt", type=int, help="llama-bench -p (default: its own default)")
    parser.add_argument("--n-gen", type=int, help="llama-bench -n (default: its own default)")
    parser.add_argument("--llama-bench", default="llama-bench", help="path of the llama-bench binary")
    parser.add_argument("--extra", action="append", default=[], help="extra argument passed through")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = parser.parse_args(argv)

    models: dict[str, str] = {}
    for item in args.model_key:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"--model-key expects NAME=PATH, got {item!r}")
        models[name] = path
    commands = [
        (name, bench_command(args.llama_bench, path, ngl, threads=threads,
                             repetitions=args.repetitions, n_prompt=args.n_prompt,
                             n_gen=args.n_gen, extra=args.extra))
        for name, path in models.items() for ngl in args.ngl for threads in (args.threads or [None])
    ]  # fmt: skip
    if args.dry_run:
        for _, command in commands:
            print(" ".join(command))
        return 0
    binary = shutil.which(args.llama_bench)
    if binary is None:
        raise SystemExit(f"cannot find the llama.cpp benchmark binary {args.llama_bench!r}: build "
                         "llama.cpp and pass --llama-bench /path/to/llama-bench")  # fmt: skip
    for path in models.values():
        if not Path(path).exists():
            raise SystemExit(f"model file {path} does not exist")

    out = Path(args.out)
    target = MeasurementSet.load(out) if out.exists() else MeasurementSet()
    added = 0
    for name, command in commands:
        command[0] = binary
        print("running:", " ".join(command), flush=True)
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise SystemExit(f"llama-bench failed ({done.returncode}):\n{done.stderr[-2000:]}")
        skipped: list[str] = []
        for measurement in parse_llama_bench(done.stdout, machine=args.machine, split=args.split,
                                             model=name, bits_per_weight=args.bits_per_weight,
                                             skipped=skipped):  # fmt: skip
            stored = add_unique(target, measurement)
            added += 1
            print(f"  {stored.id}: {stored.tokens_per_s:.2f} tokens/s ({stored.kind})")
        for note in skipped:
            print(f"  skipped {note}")
        target.save(out)  # after every run, so an interrupted matrix keeps its results
    print(f"added {added} measurement(s); {out} now holds {len(target)} "
          f"({len(target.calibration)} calibration, {len(target.validation)} validation)")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
