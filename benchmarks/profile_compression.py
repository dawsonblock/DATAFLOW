"""Measure what lossless coding of INT4 weight codes actually achieves.

    python -m benchmarks.profile_compression --synthetic gaussian --quick
    python -m benchmarks.profile_compression weights.npz --out compression.json
    python -m benchmarks.profile_compression model.safetensors --max-tensors 12

Float weight matrices are quantised to the BASP INT4 format (signed 4-bit codes with one FP16
scale per 32 weights, 4.5 bits per weight).  For the nibble-packed codes the script reports

* the order-0 entropy of the codes -- the bound for any coder that models symbols independently;
* for zlib, bz2 and lzma (Python standard library), applied to independently addressable blocks:
  the achieved bits per weight *including* the FP16 scales, and the decode throughput in bytes
  of compressed input per second on this machine.

These replace the assumed ratio of the ENTROPY stream format once run on real model weights.
Software decode rates on a CPU say nothing about an FPGA or GPU decoder; they bound what a
host-side software decoder could feed.

Exercised in the development container: everything, in ``--synthetic`` mode and on ``.npy`` /
``.npz`` files.  Not exercised: ``.safetensors`` input (needs the optional ``safetensors``
package and real weights).
"""

from __future__ import annotations

import argparse
import bz2
import json
import lzma
import statistics
import sys
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from basp.kernels import quantize_int4, symbol_entropy_bits

CODECS: dict[str, tuple[Callable[[bytes], bytes], Callable[[bytes], bytes]]] = {
    "zlib": (lambda b: zlib.compress(b, 9), zlib.decompress),
    "bz2": (lambda b: bz2.compress(b, 9), bz2.decompress),
    "lzma": (lambda b: lzma.compress(b, preset=6), lzma.decompress),
}

FPGA_CAVEAT = ("Decode rates are for single-threaded software on this CPU. They say nothing "
               "about an FPGA or GPU decoder.")  # fmt: skip


def synthetic_weights(kind: str, rows: int, cols: int, seed: int = 0) -> np.ndarray:
    """Random weight matrix: ``gaussian``, ``laplace`` or heavy-tailed ``student-t`` (3 dof)."""
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.standard_normal((rows, cols)).astype(np.float32)
    if kind == "laplace":
        return rng.laplace(size=(rows, cols)).astype(np.float32)
    if kind == "student-t":
        return np.clip(rng.standard_t(3, size=(rows, cols)), -50, 50).astype(np.float32)
    raise ValueError(f"unknown synthetic distribution {kind!r}")


def load_weights(path: str | Path) -> dict[str, np.ndarray]:
    """Float matrices from ``.npy``, ``.npz`` or ``.safetensors`` (tensors of rank < 2 are skipped)."""
    path = Path(path)
    if path.suffix == ".npy":
        raw = {path.stem: np.load(path)}
    elif path.suffix == ".npz":
        with np.load(path) as archive:
            raw = {name: archive[name] for name in archive.files}
    elif path.suffix == ".safetensors":
        try:
            from safetensors.numpy import load_file
        except ImportError:
            raise SystemExit("reading .safetensors needs the 'safetensors' package: "
                             "pip install safetensors") from None  # fmt: skip
        raw = load_file(str(path))
    else:
        raise SystemExit(f"unsupported weight file {path.name!r}: use .npy, .npz or .safetensors")
    out: dict[str, np.ndarray] = {}
    for name, array in raw.items():
        if array.ndim < 2 or not np.issubdtype(array.dtype, np.floating):
            continue
        out[name] = np.asarray(array, dtype=np.float32).reshape(array.shape[0], -1)
    return out


def split_blocks(data: bytes, block_bytes: int) -> list[bytes]:
    if block_bytes < 1:
        raise ValueError("block size must be positive")
    return [data[i:i + block_bytes] for i in range(0, len(data), block_bytes)] or [b""]


def measure_codec(name: str, blocks: list[bytes], repeats: int = 3) -> dict[str, float]:
    """Compress every block independently; verify the round trip; time decoding."""
    compress, decompress = CODECS[name]
    packed = [compress(block) for block in blocks]
    if [decompress(p) for p in packed] != blocks:
        raise RuntimeError(f"{name}: decoded data differs from the input")
    compressed_bytes = sum(len(p) for p in packed)
    timings = []
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        for p in packed:
            decompress(p)
        timings.append(time.perf_counter() - t0)
    seconds = statistics.median(timings)
    raw_bytes = sum(len(b) for b in blocks)
    return {
        "compressed_bytes": float(compressed_bytes),
        "ratio": compressed_bytes / raw_bytes if raw_bytes else 1.0,
        "decode_compressed_bytes_per_s": compressed_bytes / seconds if seconds > 0 else float("inf"),
        "decode_output_bytes_per_s": raw_bytes / seconds if seconds > 0 else float("inf"),
    }


def profile_tensor(weights: np.ndarray, *, block_kib: int = 64, repeats: int = 3) -> dict[str, Any]:
    """Compression profile of one weight matrix after INT4 quantisation."""
    tensor = quantize_int4(weights)
    n_weights = tensor.n_weights
    scale_bits = tensor.scales.nbytes * 8 / n_weights
    code_bytes = np.ascontiguousarray(tensor.packed).tobytes()
    blocks = split_blocks(code_bytes, block_kib * 1024)
    result: dict[str, Any] = {
        "shape": [int(tensor.rows), int(tensor.cols)],
        "n_weights": int(n_weights),
        "raw_bits_per_weight": tensor.nbytes * 8 / n_weights,
        "scale_bits_per_weight": scale_bits,
        "order0_entropy_bits_per_code": symbol_entropy_bits(tensor),
        "blocks": len(blocks),
        "codecs": {},
    }
    result["order0_bound_bits_per_weight"] = result["order0_entropy_bits_per_code"] + scale_bits
    for name in CODECS:
        stats = measure_codec(name, blocks, repeats)
        stats["bits_per_weight"] = stats["compressed_bytes"] * 8 / n_weights + scale_bits
        result["codecs"][name] = stats
    return result


def summarise(tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Weight-count-weighted totals over all profiled tensors."""
    total = sum(t["n_weights"] for t in tensors.values())
    if total == 0:
        raise ValueError("no tensors were profiled")

    def weighted(key: str) -> float:
        return sum(t[key] * t["n_weights"] for t in tensors.values()) / total

    codecs: dict[str, dict[str, float]] = {}
    for name in CODECS:
        compressed = sum(t["codecs"][name]["compressed_bytes"] for t in tensors.values())
        seconds = sum(t["codecs"][name]["compressed_bytes"]
                      / t["codecs"][name]["decode_compressed_bytes_per_s"] for t in tensors.values())  # fmt: skip
        codecs[name] = {
            "bits_per_weight": compressed * 8 / total + weighted("scale_bits_per_weight"),
            "decode_compressed_bytes_per_s": compressed / seconds if seconds > 0 else float("inf"),
        }
    return {
        "n_weights": total,
        "raw_bits_per_weight": weighted("raw_bits_per_weight"),
        "order0_bound_bits_per_weight": weighted("order0_bound_bits_per_weight"),
        "codecs": codecs,
    }


def render(summary: dict[str, Any], source: str) -> str:
    lines = [
        f"INT4 compression profile: {source}",
        f"  weights profiled        {summary['n_weights']:,}",
        f"  stored INT4 format      {summary['raw_bits_per_weight']:.3f} bits/weight",
        (f"  order-0 entropy bound   {summary['order0_bound_bits_per_weight']:.3f} bits/weight"
         " (codes + FP16 scales)"),
    ]
    for name, stats in summary["codecs"].items():
        lines.append(f"  {name:<5} {stats['bits_per_weight']:.3f} bits/weight, decode "
                     f"{stats['decode_compressed_bytes_per_s'] / 1e6:,.1f} MB/s of compressed input")  # fmt: skip
    lines.append("  " + FPGA_CAVEAT)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("weights", nargs="*", help=".npy, .npz or .safetensors files")
    parser.add_argument("--synthetic", choices=["gaussian", "laplace", "student-t"],
                        help="profile a random matrix instead of real weights")  # fmt: skip
    parser.add_argument("--rows", type=int, default=2048, help="rows of the synthetic matrix")
    parser.add_argument("--cols", type=int, default=4096, help="columns of the synthetic matrix")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-kib", type=int, default=64,
                        help="size of the independently addressable blocks, KiB of packed codes")  # fmt: skip
    parser.add_argument("--repeats", type=int, default=5, help="decode timing repeats (median)")
    parser.add_argument("--max-tensors", type=int, default=0, help="profile at most N tensors per file")
    parser.add_argument("--max-rows", type=int, default=0, help="use at most N rows of each tensor")
    parser.add_argument("--quick", action="store_true", help="small synthetic matrix, fewer repeats")
    parser.add_argument("--out", help="write the full result as JSON to this path")
    args = parser.parse_args(argv)
    if not args.weights and not args.synthetic:
        parser.error("give weight files or --synthetic")
    if args.quick:
        args.rows, args.cols, args.repeats = min(args.rows, 256), min(args.cols, 4096), 3

    tensors: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    if args.synthetic:
        sources.append(f"synthetic {args.synthetic} {args.rows}x{args.cols} (seed {args.seed}; "
                       "NOT model weights)")  # fmt: skip
        weights = synthetic_weights(args.synthetic, args.rows, args.cols, args.seed)
        tensors[f"synthetic.{args.synthetic}"] = profile_tensor(
            weights, block_kib=args.block_kib, repeats=args.repeats)  # fmt: skip
    for path in args.weights:
        loaded = load_weights(path)
        if not loaded:
            print(f"{path}: no floating-point matrices found", file=sys.stderr)
            continue
        sources.append(str(path))
        for i, (name, weights) in enumerate(loaded.items()):
            if args.max_tensors and i >= args.max_tensors:
                break
            if args.max_rows:
                weights = weights[: args.max_rows]
            tensors[f"{Path(path).name}:{name}"] = profile_tensor(
                weights, block_kib=args.block_kib, repeats=args.repeats)  # fmt: skip
    if not tensors:
        raise SystemExit("nothing to profile")

    summary = summarise(tensors)
    source = "; ".join(sources)
    print(render(summary, source))
    if args.out:
        payload = {"format": "basp.compression_profile", "source": source,
                   "block_kib": args.block_kib, "summary": summary, "tensors": tensors,
                   "caveat": FPGA_CAVEAT}  # fmt: skip
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
