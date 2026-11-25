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
    expert_loads: int = 0  # experts dequantised from the file (cache misses)
    expert_hits: int = 0  # experts found already dequantised in the cache
    expert_requests: int = 0  # unique experts a forward pass needed, summed over layers and calls
    expert_bytes: int = 0  # stored (quantised) bytes of those experts: what must be read from memory
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
            "experts": {"loads": self.expert_loads, "cache_hits": self.expert_hits,
                        "requests": self.expert_requests, "stored_bytes": self.expert_bytes,
                        "token_expert_pairs": self.expert_token_pairs},
            "forward": {"calls": self.forward_calls, "rows": self.forward_rows},
            "ops": [{"backend": b, "op": o, "count": self.op_count[(b, o)], "seconds": s}
                    for (b, o), s in sorted(self.op_seconds.items())],
        }  # fmt: skip
