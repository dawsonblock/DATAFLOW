"""Admission control with pressure feedback.

A request (a prompt plus the tokens it may generate) is admitted only if the memory it can
come to need fits in its domain *now*: KV cache for every token it may hold, activations and
kernel workspace.  Its memory is reserved at admission, so a running request can never be
starved of memory later by another one being admitted.  When the domain is short, unused
cached weights (expert caches) are evicted first; if that cannot make room the request is
refused with a reason -- the controller never over-commits.

Pressure feedback: ``state()`` reports *normal*, *elevated* or *critical* from the fraction of
the pool in use; under elevated pressure new requests are admitted only if they leave the
critical margin free, so caches and prefetching shrink before requests are refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .allocator import OutOfMemory
from .domains import AdmissionRefused, Block, Category, MemoryDomain


@dataclass(frozen=True)
class RequestShape:
    """What one request may need."""

    prompt_tokens: int
    max_new_tokens: int
    kv_bytes_per_token: int  # all layers, K and V
    activation_bytes: int = 0
    workspace_bytes: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.max_new_tokens

    @property
    def kv_bytes(self) -> int:
        return self.tokens * self.kv_bytes_per_token

    @property
    def total(self) -> int:
        return self.kv_bytes + self.activation_bytes + self.workspace_bytes


@dataclass
class Admission:
    request_id: str
    admitted: bool
    reason: str = ""
    blocks: list[Block] = field(default_factory=list)
    evicted: list[str] = field(default_factory=list)


class AdmissionController:
    def __init__(self, domain: MemoryDomain, *, elevated: float = 0.80, critical: float = 0.95) -> None:
        if not 0 < elevated < critical <= 1:
            raise ValueError("need 0 < elevated < critical <= 1")
        self.domain = domain
        self.elevated, self.critical = elevated, critical
        self.active: dict[str, Admission] = {}

    def state(self) -> str:
        p = self.domain.pressure
        return "critical" if p >= self.critical else "elevated" if p >= self.elevated else "normal"

    def admit(self, request_id: str, shape: RequestShape) -> Admission:
        if request_id in self.active:
            raise ValueError(f"request {request_id!r} is already admitted")
        if shape.prompt_tokens < 0 or shape.max_new_tokens < 0 or shape.kv_bytes_per_token < 0:
            raise ValueError("request sizes must be non-negative")
        need = {Category.KV: shape.kv_bytes, Category.ACTIVATIONS: shape.activation_bytes,
                Category.WORKSPACE: shape.workspace_bytes}  # fmt: skip
        need = {c: n for c, n in need.items() if n > 0}
        total = sum(self.domain._rounded(n) for n in need.values())
        if total > self.domain.usable:
            return Admission(request_id, False, f"needs {total} bytes; the domain has {self.domain.usable} usable")
        for category, n in need.items():
            limit = self.domain.limits.get(category)
            if limit is not None and self.domain.used_by(category) + self.domain._rounded(n) > limit:
                return Admission(request_id, False, f"{category.value} limit reached")
        # under pressure, leave the critical margin free after admission
        margin = 0 if self.state() == "normal" else int(self.domain.usable * (1 - self.critical))
        target = total + margin
        evicted = []
        if self.domain.free < target:
            evicted = [b.owner for b in self.domain.evict(target, category=Category.WEIGHTS)]
        if self.domain.free < target:
            return Admission(request_id, False, f"insufficient memory: {total} bytes (+{margin} margin) needed, "
                             f"{self.domain.free} free after evicting every unreferenced cached weight",
                             evicted=evicted)  # fmt: skip
        blocks: list[Block] = []
        try:
            for category, n in need.items():
                blocks.append(self.domain.allocate(n, category, f"{request_id}.{category.value}", refs=1))
        except (OutOfMemory, AdmissionRefused) as exc:  # fragmentation: undo, refuse
            for b in blocks:
                self.domain.release(b, free_when_unused=True)
            return Admission(request_id, False, f"fragmented: {exc}", evicted=evicted)
        admission = Admission(request_id, True, "", blocks, evicted)
        self.active[request_id] = admission
        return admission

    def finish(self, request_id: str) -> None:
        admission = self.active.pop(request_id)
        for block in admission.blocks:
            self.domain.release(block, free_when_unused=True)
