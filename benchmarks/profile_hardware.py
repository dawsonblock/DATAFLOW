"""Measure the hardware parameters the simulator needs, on the machine this runs on.

    python -m benchmarks.profile_hardware --quick
    python -m benchmarks.profile_hardware --machine i9-11900kf-3080ti --out hw_i9.json \
        --template desktop_gpu12 --host-device host --device gpu --link pcie-gpu

What it measures:

1. **Sustained memory bandwidth** with NumPy: copying and summing arrays far larger than the
   caches, in one process and in N processes at once.  A copy of ``n`` bytes reads ``n`` and
   writes ``n``, so the bus traffic is at least twice the payload rate (more if the C library
   does not use non-temporal stores); both figures are reported.
2. **A NumPy INT4 streaming-kernel rate** using ``basp.kernels``.  This is the *reference*
   kernel: it is far below an optimised AVX/NEON kernel.  Do not use it for ``compute_rate`` --
   calibrate that from ``llama-bench`` results (``benchmarks/run_llama_bench.py``).
3. **GPU figures**, if PyTorch with CUDA or MPS is installed: device name and memory,
   host->device and device->host transfer times over a size sweep (fitted to
   ``latency + bytes / bandwidth``), kernel launch overhead (a tiny synchronised operation),
   and FP16 matrix-vector/matrix-matrix times on resident weights (fitted to the roofline
   ``startup + max(W / B, C / P)`` when both regimes are covered).

With ``--template`` the fitted values are written into a copy of a hardware profile; every field
not listed under ``_calibration.measured`` in the result remains an assumption.

Exercised in the development container: parts 1 and 2 and the profile writing.  NOT exercised:
part 3 on real hardware -- the container has neither a GPU nor PyTorch.  The GPU functions take
the ``torch`` module as an argument and were run only against a small fake in the tests.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from basp.calibration.fit import (
    TransferFit,
    apply_fits,
    apply_link_fit,
    fit_roofline,
    fit_transfer,
)
from basp.kernels import matvec_int4, quantize_int4

MiB = 1024 * 1024
REFERENCE_KERNEL_NOTE = ("reference NumPy kernel, far below an optimised AVX/NEON kernel -- use "
                         "llama-bench results to calibrate compute_rate, not this")  # fmt: skip
Clock = Callable[[], float]


# -- CPU and memory ----------------------------------------------------------------------
def machine_info() -> dict[str, Any]:
    return {
        "node": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def measure_copy(nbytes: int, repeats: int = 5) -> dict[str, float]:
    """Single-process copy of ``nbytes``: payload rate and the implied bus traffic (>= 2x)."""
    src = np.ones(nbytes, dtype=np.uint8)
    dst = np.empty_like(src)
    np.copyto(dst, src)  # touch every page before timing
    best = float("inf")
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        best = min(best, time.perf_counter() - t0)
    payload = nbytes / best
    return {"payload_bytes_per_s": payload, "traffic_bytes_per_s": 2.0 * payload}


def measure_sum(nbytes: int, repeats: int = 5) -> dict[str, float]:
    """Single-process read-only pass (a sum over float64): a lower bound of one core's read rate."""
    data = np.ones(max(nbytes // 8, 1), dtype=np.float64)
    data.sum()
    best = float("inf")
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        data.sum()
        best = min(best, time.perf_counter() - t0)
    return {"read_bytes_per_s": data.nbytes / best}


def _parallel_worker(mode: str, nbytes: int, duration: float, barrier, results) -> None:
    """Run copies or sums for ``duration`` seconds after all workers are ready; report bytes/s."""
    if mode == "copy":
        src = np.ones(nbytes, dtype=np.uint8)
        dst = np.empty_like(src)
        np.copyto(dst, src)
    else:
        src = np.ones(max(nbytes // 8, 1), dtype=np.float64)
        src.sum()
    barrier.wait()
    moved = 0
    t0 = time.perf_counter()
    while True:
        if mode == "copy":
            np.copyto(dst, src)
        else:
            src.sum()
        moved += src.nbytes
        elapsed = time.perf_counter() - t0
        if elapsed >= duration:
            break
    results.put(moved / elapsed)


def measure_parallel(mode: str, nbytes: int, processes: int, duration: float = 1.0) -> dict[str, Any]:
    """Aggregate rate of ``processes`` simultaneous copy or read loops (``mode``: copy | read)."""
    if mode not in ("copy", "read"):
        raise ValueError("mode must be 'copy' or 'read'")
    ctx = multiprocessing.get_context("spawn")  # same behaviour on Linux, macOS and Windows
    barrier, results = ctx.Barrier(processes), ctx.Queue()
    workers = [ctx.Process(target=_parallel_worker, args=(mode, nbytes, duration, barrier, results))
               for _ in range(processes)]  # fmt: skip
    for worker in workers:
        worker.start()
    rates = [results.get(timeout=120 + 10 * duration) for _ in workers]
    for worker in workers:
        worker.join()
    total = float(sum(rates))
    out: dict[str, Any] = {"processes": processes, "per_process_bytes_per_s": sorted(rates)}
    if mode == "copy":
        out["payload_bytes_per_s"] = total
        out["traffic_bytes_per_s"] = 2.0 * total
    else:
        out["read_bytes_per_s"] = total
    return out


def measure_int4_kernel(rows: int = 512, cols: int = 4096, repeats: int = 3) -> dict[str, Any]:
    """MAC rate of the NumPy reference INT4 kernel (see the module docstring for the caveat)."""
    rng = np.random.default_rng(0)
    tensor = quantize_int4(rng.standard_normal((rows, cols)).astype(np.float32))
    x = rng.standard_normal(cols).astype(np.float16)
    matvec_int4(tensor, x)
    best = float("inf")
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        matvec_int4(tensor, x)
        best = min(best, time.perf_counter() - t0)
    return {"rows": rows, "cols": cols, "macs_per_s": rows * cols / best,
            "weight_bytes_per_s": tensor.nbytes / best, "note": REFERENCE_KERNEL_NOTE}  # fmt: skip


def profile_cpu(*, quick: bool, processes: int | None = None) -> dict[str, Any]:
    nbytes = (64 if quick else 512) * MiB
    repeats = 3 if quick else 7
    duration = 0.2 if quick else 1.5
    n = processes or (min(os.cpu_count() or 1, 4) if quick else (os.cpu_count() or 1))
    copy_one, read_one = measure_copy(nbytes, repeats), measure_sum(nbytes, repeats)
    per_process = max(nbytes // 2, 16 * MiB)
    copy_all = measure_parallel("copy", per_process, n, duration)
    read_all = measure_parallel("read", per_process, n, duration)
    sustained = max(copy_all["traffic_bytes_per_s"], read_all["read_bytes_per_s"],
                    copy_one["traffic_bytes_per_s"])  # fmt: skip
    return {
        "array_bytes": nbytes,
        "copy_single": copy_one,
        "read_single": read_one,
        "copy_parallel": copy_all,
        "read_parallel": read_all,
        "sustained_bytes_per_s": sustained,
        "sustained_note": "largest observed bus traffic: max(parallel read, 2 x parallel copy "
                          "payload, 2 x single copy payload)",
        "int4_reference_kernel": measure_int4_kernel(256 if quick else 1024, 2048 if quick else 4096,
                                                     2 if quick else 5),
    }  # fmt: skip


# -- GPU (PyTorch) -------------------------------------------------------------------------
def pick_device(torch) -> str | None:
    """``"cuda"``, ``"mps"`` or ``None`` when PyTorch sees no accelerator."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return None


def synchronise(torch, device: str) -> None:
    """Wait for queued device work so a timing covers the whole operation."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def device_info(torch, device: str) -> dict[str, Any]:
    if device == "cuda":
        return {"device": "cuda", "name": torch.cuda.get_device_name(0),
                "memory_bytes": int(torch.cuda.get_device_properties(0).total_memory)}  # fmt: skip
    recommended = getattr(getattr(torch, "mps", None), "recommended_max_memory", None)
    return {"device": device, "name": f"Apple {platform.machine()} (MPS, unified memory)",
            "memory_bytes": int(recommended()) if callable(recommended) else None}  # fmt: skip


def measure_transfers(torch, device: str, sizes: Sequence[int], repeats: int = 5,
                      clock: Clock = time.perf_counter) -> dict[str, list[tuple[int, float]]]:  # fmt: skip
    """Host->device and device->host copy timings as (bytes, seconds) samples."""
    samples: dict[str, list[tuple[int, float]]] = {"h2d": [], "d2h": []}
    for size in sizes:
        host = torch.empty(int(size), dtype=torch.uint8)
        if device == "cuda":
            try:
                host = host.pin_memory()  # pageable memory would measure the page-locking cost
            except RuntimeError:
                pass
        on_device = host.to(device)  # warm-up: allocator and page faults
        synchronise(torch, device)
        for _ in range(max(repeats, 1)):
            t0 = clock()
            on_device = host.to(device)
            synchronise(torch, device)
            samples["h2d"].append((int(size), clock() - t0))
            t0 = clock()
            on_device.cpu()
            samples["d2h"].append((int(size), clock() - t0))
    return samples


def measure_launch_overhead(torch, device: str, repeats: int = 200,
                            clock: Clock = time.perf_counter) -> float:  # fmt: skip
    """Median seconds of a tiny synchronised operation: the fixed cost of launching a kernel."""
    x = torch.zeros(1, device=device)
    y = x + 1
    synchronise(torch, device)
    timings = []
    for _ in range(max(repeats, 1)):
        t0 = clock()
        y = x + 1
        synchronise(torch, device)
        timings.append(clock() - t0)
    del y
    return float(statistics.median(timings))


def measure_matvec(torch, device: str, shapes: Sequence[tuple[int, int, int]], repeats: int = 5,
                   clock: Clock = time.perf_counter) -> list[dict[str, float]]:  # fmt: skip
    """FP16 products ``W[rows, cols] @ X[cols, batch]`` on resident weights (minimum of repeats).

    Batch 1 is memory-bound (every weight is read for one MAC); large batches are compute-bound.
    """
    points: list[dict[str, float]] = []
    for rows, cols, batch in shapes:
        w = torch.randn((rows, cols), dtype=torch.float16, device=device)
        x = torch.randn((cols, batch), dtype=torch.float16, device=device)
        y = w @ x
        synchronise(torch, device)
        best = float("inf")
        for _ in range(max(repeats, 1)):
            t0 = clock()
            y = w @ x
            synchronise(torch, device)
            best = min(best, clock() - t0)
        del y
        points.append({"rows": rows, "cols": cols, "batch": batch, "weight_bytes": rows * cols * 2.0,
                       "macs": float(rows) * cols * batch, "seconds": best})  # fmt: skip
    return points


def profile_gpu(torch, *, quick: bool, clock: Clock = time.perf_counter) -> dict[str, Any] | None:
    """All GPU measurements and their fits, or ``None`` when PyTorch sees no accelerator."""
    device = pick_device(torch)
    if device is None:
        return None
    sizes = [4096, 65536, MiB, 16 * MiB, 64 * MiB] + ([] if quick else [256 * MiB, 1024 * MiB])
    shapes = [(r, c, b) for r, c in ([(2048, 2048), (4096, 4096)] if quick else
                                     [(2048, 2048), (4096, 4096), (8192, 8192), (16384, 8192)])
              for b in ((1, 16, 256) if quick else (1, 4, 16, 64, 256, 1024))]  # fmt: skip
    repeats = 3 if quick else 7
    result: dict[str, Any] = device_info(torch, device)
    transfers = measure_transfers(torch, device, sizes, repeats, clock)
    result["transfers"] = transfers
    for direction, samples in transfers.items():
        try:
            fit = fit_transfer([s for s, _ in samples], [t for _, t in samples])
            result[f"{direction}_fit"] = {"latency_s": fit.latency_s, "r2": fit.r2,
                                          "bandwidth_bytes_per_s": fit.bandwidth_bytes_per_s}  # fmt: skip
        except ValueError as exc:
            result[f"{direction}_fit_error"] = str(exc)
    result["launch_overhead_s"] = measure_launch_overhead(torch, device, 50 if quick else 300, clock)
    points = measure_matvec(torch, device, shapes, repeats, clock)
    result["matvec"] = points
    try:
        fit = fit_roofline([p["weight_bytes"] for p in points], [p["macs"] for p in points],
                           [p["seconds"] for p in points])  # fmt: skip
        result["roofline_fit"] = {"mem_bandwidth": fit.mem_bandwidth, "compute_rate": fit.compute_rate,
                                  "startup_s": fit.startup_s, "rel_error": fit.rel_error}  # fmt: skip
    except ValueError as exc:
        result["roofline_fit_error"] = str(exc)
    return result


# -- profile writing -----------------------------------------------------------------------
def load_template(name: str) -> dict[str, Any]:
    """A hardware profile by path, or by name from ``configs/``."""
    path = Path(name)
    if path.suffix != ".json":
        path = Path(__file__).resolve().parents[1] / "configs" / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"hardware profile template {path} not found")
    return json.loads(path.read_text())


def fitted_profile(template: dict[str, Any], measurements: dict[str, Any], *,
                   host_device: str | None = None, device: str | None = None,
                   link: str | None = None) -> dict[str, Any]:  # fmt: skip
    """Copy of ``template`` with every available measured value applied."""
    profile = template
    if host_device:
        profile = apply_fits(profile, host_device,
                             mem_bandwidth=measurements["cpu"]["sustained_bytes_per_s"])  # fmt: skip
    gpu = measurements.get("gpu")
    if gpu and device:
        roofline = gpu.get("roofline_fit", {})
        profile = apply_fits(profile, device, mem_bandwidth=roofline.get("mem_bandwidth"),
                             compute_rate=roofline.get("compute_rate"),
                             kernel_startup_s=gpu.get("launch_overhead_s"))  # fmt: skip
    if gpu and link:
        spec = next((k for k in profile.get("links", []) if k.get("name") == link), None)
        if spec is None:
            raise SystemExit(f"template has no link named {link!r}")
        towards_device = "ab" if spec.get("b") == device else "ba"
        away = "ba" if towards_device == "ab" else "ab"
        for key, direction in (("h2d_fit", towards_device), ("d2h_fit", away)):
            if key in gpu:
                fit = TransferFit(gpu[key]["latency_s"], gpu[key]["bandwidth_bytes_per_s"],
                                  gpu[key]["r2"])  # fmt: skip
                profile = apply_link_fit(profile, link, fit, direction)
    if "_calibration" in profile:
        profile = dict(profile)
        profile["name"] = f"{template.get('name', 'profile')}_measured"
        profile["_note"] = ("PARTLY MEASURED PROFILE. Fields listed in _calibration.measured were "
                            "fitted on the named machine; every other value is still an assumption.")  # fmt: skip
        profile["_calibration"]["machine"] = measurements["machine"]
    return profile


def render(measurements: dict[str, Any]) -> str:
    cpu = measurements["cpu"]
    gb = 1e9
    copy_one, copy_all, read_all = cpu["copy_single"], cpu["copy_parallel"], cpu["read_parallel"]
    kernel = cpu["int4_reference_kernel"]
    lines = [
        f"Hardware profile of {measurements['machine']['node'] or 'this machine'}",
        (f"  copy, 1 process      {copy_one['payload_bytes_per_s'] / gb:7.1f} GB/s payload "
         f"({copy_one['traffic_bytes_per_s'] / gb:.1f} GB/s traffic)"),
        f"  read, 1 process      {cpu['read_single']['read_bytes_per_s'] / gb:7.1f} GB/s",
        (f"  copy, {copy_all['processes']:>2} processes   "
         f"{copy_all['payload_bytes_per_s'] / gb:7.1f} GB/s payload "
         f"({copy_all['traffic_bytes_per_s'] / gb:.1f} GB/s traffic)"),
        f"  read, {read_all['processes']:>2} processes   {read_all['read_bytes_per_s'] / gb:7.1f} GB/s",
        f"  sustained bandwidth  {cpu['sustained_bytes_per_s'] / gb:7.1f} GB/s",
        (f"  INT4 reference kernel {kernel['macs_per_s'] / 1e9:6.3f} GMAC/s "
         "(NumPy reference; not a compute_rate calibration)"),
    ]
    gpu = measurements.get("gpu")
    if gpu is None:
        lines.append(f"  GPU: not measured ({measurements.get('gpu_skipped', 'no accelerator')})")
    else:
        lines.append(f"  GPU: {gpu['name']}")
        for key, label in (("h2d_fit", "host->device"), ("d2h_fit", "device->host")):
            if key in gpu:
                lines.append(f"    {label}  {gpu[key]['bandwidth_bytes_per_s'] / gb:6.1f} GB/s, "
                             f"latency {gpu[key]['latency_s'] * 1e6:.1f} us")  # fmt: skip
        lines.append(f"    launch overhead {gpu['launch_overhead_s'] * 1e6:.1f} us")
        if "roofline_fit" in gpu:
            fit = gpu["roofline_fit"]
            lines.append(f"    FP16 kernels    {fit['mem_bandwidth'] / gb:.0f} GB/s memory, "
                         f"{fit['compute_rate'] / 1e12:.2f} TMAC/s (residual {fit['rel_error']:.1%})")  # fmt: skip
        else:
            lines.append(f"    roofline not fitted: {gpu.get('roofline_fit_error')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    parser.add_argument("--quick", action="store_true", help="short run with small arrays")
    parser.add_argument("--machine", help="label for this machine (default: its host name)")
    parser.add_argument("--processes", type=int, help="processes for the parallel memory tests")
    parser.add_argument("--no-gpu", action="store_true", help="skip the GPU measurements")
    parser.add_argument("--require-gpu", action="store_true",
                        help="fail instead of skipping when GPU measurements are impossible")  # fmt: skip
    parser.add_argument("--out", help="where to write the raw measurements (JSON)")
    parser.add_argument("--template", help="hardware profile to update: a name in configs/ or a path")
    parser.add_argument("--host-device", help="device of the template that is this machine's CPU")
    parser.add_argument("--device", help="device of the template that is this machine's GPU")
    parser.add_argument("--link", help="link of the template between the CPU and the GPU")
    parser.add_argument("--profile-out", help="where to write the fitted profile (JSON)")
    args = parser.parse_args(argv)

    info = machine_info()
    if args.machine:
        info["node"] = args.machine
    measurements: dict[str, Any] = {"format": "basp.hardware_measurements", "machine": info,
                                    "quick": args.quick, "gpu": None}  # fmt: skip
    measurements["cpu"] = profile_cpu(quick=args.quick, processes=args.processes)
    if args.no_gpu:
        measurements["gpu_skipped"] = "--no-gpu"
    else:
        try:
            import torch
        except ImportError:
            measurements["gpu_skipped"] = "PyTorch is not installed (pip install torch)"
        else:
            measurements["gpu"] = profile_gpu(torch, quick=args.quick)
            if measurements["gpu"] is None:
                measurements["gpu_skipped"] = "PyTorch found neither CUDA nor MPS"
    if args.require_gpu and measurements["gpu"] is None:
        raise SystemExit(f"GPU measurements required but impossible: {measurements['gpu_skipped']}")

    print(render(measurements))
    out = Path(args.out or f"hardware_{info['node'] or 'machine'}.json")
    out.write_text(json.dumps(measurements, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    if args.template:
        template = load_template(args.template)
        profile = fitted_profile(template, measurements, host_device=args.host_device,
                                 device=args.device, link=args.link)  # fmt: skip
        target = Path(args.profile_out or f"{profile.get('name', 'profile')}.json")
        target.write_text(json.dumps(profile, indent=2) + "\n")
        measured = profile.get("_calibration", {}).get("measured", [])
        print(f"wrote {target} ({len(measured)} measured field(s): {', '.join(measured) or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
