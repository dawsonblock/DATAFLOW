"""Measured timings of the physical runtime (wall clock on this machine)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Telemetry:
    op_seconds: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    op_count: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    prefill_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_tokens: int = 0
    decode_seconds: float = 0.0
    # Expert demand and what it cost.  These are different quantities and are reported apart:
    # *logical* demand (what the computation asked for), cache behaviour, the loader's work on
    # misses, and -- only where a measurement source observed it -- physical traffic.
    expert_requests: int = 0  # logical: unique experts a pass needed, summed over layers and passes
    expert_logical_bytes: int = 0  # logical: stored (quantised) size of those experts; NOT bytes moved
    expert_hits: int = 0  # requests satisfied by resident experts
    expert_misses: int = 0  # requests that had to be loaded
    expert_loads: int = 0  # experts dequantised (successful loads)
    expert_decoded_stored_bytes: int = 0  # quantised bytes the loader processed on misses (may come from the OS page cache)
    expert_decoded_bytes: int = 0  # float32 bytes the loader produced on misses
    expert_failed_acquisitions: int = 0  # acquisitions rolled back
    peak_load_workspace_bytes: int = 0  # largest dequantisation scratch reservation
    expert_token_pairs: int = 0  # token-expert pairs computed (rows x top-k)
    forward_calls: int = 0
    forward_rows: int = 0

    def record(self, backend: str, op: str, seconds: float) -> None:
        self.op_seconds[(backend, op)] += seconds
        self.op_count[(backend, op)] += 1

    @property
    def decode_tokens_per_s(self) -> float:
        return self.decode_tokens / self.decode_seconds if self.decode_seconds else 0.0

    @property
    def prefill_tokens_per_s(self) -> float:
        return self.prefill_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "prefill": {"tokens": self.prefill_tokens, "seconds": self.prefill_seconds,
                        "tokens_per_s": self.prefill_tokens_per_s},
            "decode": {"tokens": self.decode_tokens, "seconds": self.decode_seconds,
                       "tokens_per_s": self.decode_tokens_per_s},
            "experts": {"logical_requests": self.expert_requests, "logical_stored_bytes": self.expert_logical_bytes,
                        "cache_hits": self.expert_hits, "cache_misses": self.expert_misses,
                        "loads": self.expert_loads, "decoded_stored_bytes": self.expert_decoded_stored_bytes,
                        "decoded_float32_bytes": self.expert_decoded_bytes,
                        "failed_acquisitions": self.expert_failed_acquisitions,
                        "peak_load_workspace_bytes": self.peak_load_workspace_bytes,
                        "token_expert_pairs": self.expert_token_pairs},
            "forward": {"calls": self.forward_calls, "rows": self.forward_rows},
            "ops": [{"backend": b, "op": o, "count": self.op_count[(b, o)], "seconds": s}
                    for (b, o), s in sorted(self.op_seconds.items())],
        }  # fmt: skip


def io_snapshot() -> dict[str, Any]:
    """Physical I/O the operating system attributes to this process, with the source of each
    number.  ``storage_read_bytes`` counts bytes actually fetched from the storage layer: data
    served from the OS page cache (a memory-mapped model that is already cached) is not in it."""
    out: dict[str, Any] = {"storage_read_bytes": None, "major_faults": None, "source": []}
    try:
        with open("/proc/self/io") as f:
            fields = dict(line.strip().split(": ", 1) for line in f if ": " in line)
        out["storage_read_bytes"] = int(fields["read_bytes"])
        out["source"].append("/proc/self/io read_bytes")
    except (OSError, KeyError, ValueError):
        pass
    try:
        import resource

        out["major_faults"] = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
        out["source"].append("getrusage ru_majflt")
    except (ImportError, OSError):
        pass
    return out


def io_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Physical I/O between two snapshots (``None`` where the platform does not report it)."""
    out: dict[str, Any] = {"source": after["source"]}
    for key in ("storage_read_bytes", "major_faults"):
        out[key] = None if before[key] is None or after[key] is None else after[key] - before[key]
    return out
