"""Closed-form dataflow comparison for one expert (specification section 6).

    python -m benchmarks.dataflow_model
    python -m benchmarks.dataflow_model --hidden 2048 --intermediate 768 --bits 4.5 --link 16

Compares moving the weights to a remote processor with executing next to the
memory that holds them.  Analytical illustration, not a measurement.
"""

from __future__ import annotations

import argparse

from basp.experiments.analytic import compare_paths, size_arithmetic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, default=4096, help="hidden dimension")
    parser.add_argument("--intermediate", type=int, default=14336, help="intermediate dimension")
    parser.add_argument("--bits", type=float, default=4.5, help="effective bits per weight")
    parser.add_argument("--link", type=float, default=24.0, help="interconnect bandwidth, GB/s")
    parser.add_argument("--memory", type=float, default=54.0, help="local memory bandwidth, GB/s")
    parser.add_argument("--compute", type=float, default=128.0, help="local compute, GMAC/s")
    parser.add_argument("--decode", type=float, default=None, help="decode throughput, GB/s (default: unlimited)")
    args = parser.parse_args()
    result = compare_paths(
        args.hidden, args.intermediate, args.bits, args.link * 1e9, args.memory * 1e9,
        args.compute * 1e9, decode_rate=args.decode * 1e9 if args.decode else float("inf"),
    )  # fmt: skip
    width = max(len(name) for name, _ in result.as_rows())
    for name, value in result.as_rows():
        print(f"{name:<{width}}  {value}")
    sizing = size_arithmetic(bits_per_weight=args.bits)
    print(f"\nTwo DDR5-5600 channels: {sizing.memory_bandwidth / 1e9:.1f} GB/s theoretical; "
          f"{sizing.required_mac_rate / 1e9:.0f} GMAC/s needed to consume the stream; "
          f"1024 lanes at 250 MHz give {sizing.lane_mac_rate / 1e9:.0f} GMAC/s.")  # fmt: skip


if __name__ == "__main__":
    main()
