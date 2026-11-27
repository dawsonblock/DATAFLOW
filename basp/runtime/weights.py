"""Model weights in host memory: dense weights resident, experts loaded on demand.

Every byte is charged to the host's memory domain (``basp.memory``): dense weights as pinned
``weights`` blocks; experts as unpinned blocks in a least-recently-used cache, so an expert can
be dropped and re-read from the memory-mapped file.

**The expert-memory contract.**  ``expert_cache_bytes`` is a hard ceiling on the float32 arrays
of resident experts -- cached *and* in use.  A layer that needs more experts than fit at once
is computed in bounded groups (``groups``); a ceiling that cannot hold one expert is refused
when the store is built.  ``None`` is the unrestricted mode: only the memory domain limits how
many experts are resident.  The ceiling covers expert arrays only; dequantisation scratch
(reserved as workspace while experts load), dense weights, KV caches and the process's other
memory are accounted separately (``memory_report``).

**Acquisition is a transaction.**  ``fetch`` reserves memory for every missing expert *before*
any is loaded, loads them on a bounded number of worker threads, and on any failure --
allocation, file read, dequantisation -- releases every reference it took and frees every
block it created, so a failed call leaves no trace and can be retried.

**Ownership.**  Each expert block registers the store as its owner: when the domain reclaims
it (the store's own eviction, or admission control making room for a request), the cache entry
is dropped first.  The store never keeps arrays for memory the domain reports as free.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import numpy as np

from basp.memory import Block, Category, MemoryDomain
from basp.memory.allocator import OutOfMemory
from basp.models.gguf_tensors import GGUFTensors
from basp.models.quants import transient_bytes

from .config import ModelConfig, UnsupportedModel
from .telemetry import Telemetry

_ATTENTION = ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm", "ffn_norm")
_DENSE_FFN = ("ffn_gate", "ffn_up", "ffn_down")
_MOE = ("ffn_gate_inp",)
_EXPERTS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def expected_tensors(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor the runtime reads, with its NumPy shape."""
    h, hd, ff = config.hidden, config.head_dim, config.ffn
    shapes = {"token_embd.weight": (config.vocab, h), "output_norm.weight": (h,)}
    if not config.tied_embeddings:
        shapes["output.weight"] = (config.vocab, h)
    for i in range(config.n_layers):
        p = f"blk.{i}."
        shapes.update({
            p + "attn_norm.weight": (h,), p + "ffn_norm.weight": (h,),
            p + "attn_q.weight": (config.n_head * hd, h), p + "attn_k.weight": (config.n_kv_head * hd, h),
            p + "attn_v.weight": (config.n_kv_head * hd, h), p + "attn_output.weight": (h, config.n_head * hd),
            p + "attn_q_norm.weight": (hd,), p + "attn_k_norm.weight": (hd,),
        })  # fmt: skip
        if config.moe:
            e, ef = config.n_experts, config.expert_ffn
            shapes.update({p + "ffn_gate_inp.weight": (e, h), p + "ffn_gate_exps.weight": (e, ef, h),
                           p + "ffn_up_exps.weight": (e, ef, h), p + "ffn_down_exps.weight": (e, h, ef)})  # fmt: skip
        else:
            shapes.update({p + "ffn_gate.weight": (ff, h), p + "ffn_up.weight": (ff, h),
                           p + "ffn_down.weight": (h, ff)})  # fmt: skip
    return shapes


class ExpertBudgetExceeded(MemoryError):
    """The expert-memory ceiling (or the memory domain) cannot hold what one step needs."""


@dataclass
class _Entry:
    mats: tuple[np.ndarray, np.ndarray, np.ndarray] | None  # None only while its transaction loads it
    block: Block


class WeightStore:
    """``embedding="rows"`` reads token embeddings row by row instead of keeping the table (only
    when the output projection does not share it); ``workers`` threads dequantise experts in
    parallel (NumPy releases the interpreter lock while it works)."""

    def __init__(self, tensors: GGUFTensors, config: ModelConfig, domain: MemoryDomain, telemetry: Telemetry,
                 *, expert_cache_bytes: int | None = None, embedding: str = "resident",
                 workers: int | None = None) -> None:  # fmt: skip
        self.tensors, self.config, self.domain, self.telemetry = tensors, config, domain, telemetry
        if embedding not in ("resident", "rows"):
            raise ValueError("embedding must be 'resident' or 'rows'")
        expected = expected_tensors(config)
        present = {t.name for t in tensors.file.tensors}
        missing = sorted(set(expected) - present)
        unknown = sorted(present - set(expected))
        if missing:
            raise UnsupportedModel(f"missing tensors: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unknown:  # never silently ignore part of a model (biases, extra norms, ...)
            raise UnsupportedModel(f"tensors this runtime does not implement: {unknown[:5]}")
        for name, shape in expected.items():
            if tensors.shape(name) != shape:
                raise UnsupportedModel(f"{name}: shape {tensors.shape(name)}, expected {shape}")
        #: float32 bytes of one resident expert (gate, up and down matrices)
        self.expert_nbytes = 3 * config.hidden * config.expert_ffn * 4 if config.moe else 0
        if expert_cache_bytes is not None:
            if expert_cache_bytes < 0:
                raise ValueError("expert_cache_bytes must be non-negative")
            if config.moe and expert_cache_bytes < self.expert_nbytes:
                raise ExpertBudgetExceeded(
                    f"expert_cache_bytes={expert_cache_bytes} cannot hold one expert of this model "
                    f"({self.expert_nbytes} bytes as float32)")  # fmt: skip
        self.expert_cache_bytes = expert_cache_bytes
        self.rows_embedding = embedding == "rows" and not config.tied_embeddings
        self._workers = workers or min(4, os.cpu_count() or 1)
        self._pool = ThreadPoolExecutor(max_workers=self._workers)
        names = [n for n in expected if not any(f".{kind}." in n for kind in _EXPERTS)
                 and not (self.rows_embedding and n == "token_embd.weight")]  # fmt: skip
        self._dense: dict[str, np.ndarray] = {}
        self.dense_nbytes = 0
        for name, array in zip(names, self._pool.map(tensors.load, names)):
            domain.allocate(array.nbytes, Category.WEIGHTS, name, pinned=True)
            self._dense[name] = array
            self.dense_nbytes += array.nbytes
        self._experts: OrderedDict[tuple[int, int], _Entry] = OrderedDict()
        self.peak_resident_bytes = 0
        # stored (quantised) bytes of one expert, per layer: formats can differ between layers
        self._expert_stored = [sum(tensors.slice_nbytes(f"blk.{i}.{k}.weight") for k in _EXPERTS)
                               for i in range(config.n_layers)] if config.moe else []  # fmt: skip
        # scratch one in-flight load needs: its matrices are dequantised one after another
        self._scratch = [max(transient_bytes(tensors.info(f"blk.{i}.{k}.weight").ggml_type,
                                             config.hidden * config.expert_ffn) for k in _EXPERTS)
                         for i in range(config.n_layers)] if config.moe else []  # fmt: skip

    def __getitem__(self, name: str) -> np.ndarray:
        return self._dense[name]

    @property
    def output(self) -> np.ndarray:
        return self._dense["token_embd.weight" if self.config.tied_embeddings else "output.weight"]

    def embed(self, ids: np.ndarray) -> np.ndarray:
        if self.rows_embedding:
            return self.tensors.load_rows("token_embd.weight", ids)
        return self._dense["token_embd.weight"][ids]

    # -- experts: accounting --------------------------------------------------------------------
    @property
    def resident_bytes(self) -> int:
        """Float32 bytes of every resident expert, cached or in use (what the ceiling bounds)."""
        return len(self._experts) * self.expert_nbytes

    def cached(self, layer: int, expert: int) -> bool:
        entry = self._experts.get((layer, expert))
        return entry is not None and entry.mats is not None

    def _forget(self, key: tuple[int, int], _block: Block) -> None:
        """The owner's reclamation hook: drop the arrays before the domain frees the block."""
        self._experts.pop(key, None)

    def _reclaimable(self, keep: set[tuple[int, int]]) -> int:
        return sum(1 for key, entry in self._experts.items() if key not in keep and entry.block.evictable)

    def group_capacity(self, layer: int, wanted: list[int]) -> int:
        """How many of ``wanted`` can be resident at once: bounded by the ceiling and by what the
        domain can provide (free memory plus the unreferenced experts it could reclaim)."""
        keys = {(layer, e) for e in wanted}
        held = sum(1 for key, entry in self._experts.items() if key not in keys and not entry.block.evictable)
        by_limit = len(wanted)
        if self.expert_cache_bytes is not None:
            by_limit = self.expert_cache_bytes // self.expert_nbytes - held
        charged = self.domain._rounded(self.expert_nbytes)
        present = sum(1 for key in keys if key in self._experts)
        available = self.domain.headroom(Category.WEIGHTS) + self._reclaimable(keys) * charged
        # one in-flight load also needs its dequantisation scratch
        by_domain = present + max(0, available - self.domain._rounded(self._scratch[layer])) // charged
        return max(0, min(len(wanted), by_limit, by_domain))

    def groups(self, layer: int, experts) -> list[list[int]]:
        """``experts`` in ascending order, split into groups that can each be resident at once.
        One group when everything fits.  Raises before anything is acquired if not even one
        expert can be made resident."""
        wanted = sorted({int(e) for e in experts})
        if not wanted:
            return []
        capacity = self.group_capacity(layer, wanted)
        if capacity < 1:
            raise ExpertBudgetExceeded(
                f"layer {layer} needs an expert of {self.expert_nbytes} bytes but none can be made resident "
                f"(ceiling {self.expert_cache_bytes}, {self.resident_bytes} bytes resident and in use, "
                f"{self.domain.free} bytes free in the domain)")  # fmt: skip
        return [wanted[i:i + capacity] for i in range(0, len(wanted), capacity)]

    # -- experts: transactional acquisition ---------------------------------------------------
    def fetch(self, layer: int, experts) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Dequantised weights of ``experts`` of one layer.  Each returned expert is referenced
        (cannot be evicted) until ``release``.  All or nothing: on any failure every reference
        taken and every block created by this call is undone before the error propagates."""
        wanted = sorted({int(e) for e in experts})
        if any(not 0 <= e < self.config.n_experts for e in wanted) or not 0 <= layer < self.config.n_layers:
            raise IndexError(f"layer {layer}, experts {wanted}: outside the model")
        limit = self.expert_cache_bytes
        if limit is not None and len(wanted) * self.expert_nbytes > limit:
            raise ExpertBudgetExceeded(
                f"{len(wanted)} experts ({len(wanted) * self.expert_nbytes} bytes) exceed the ceiling of {limit} "
                "bytes for one acquisition; split the layer with groups()")  # fmt: skip
        tel = self.telemetry
        tel.expert_requests += len(wanted)
        tel.expert_logical_bytes += len(wanted) * self._expert_stored[layer]
        keep = {(layer, e) for e in wanted}
        retained: list[Block] = []
        created: list[tuple[int, int]] = []
        missing: list[int] = []
        try:
            for e in wanted:
                entry = self._experts.get((layer, e))
                if entry is None:
                    missing.append(e)
                    continue
                if entry.mats is None:
                    raise RuntimeError(f"expert {(layer, e)} is being loaded by another acquisition")
                self.domain.retain(entry.block)
                retained.append(entry.block)
                self._experts.move_to_end((layer, e))
            # reserve every missing expert before any work that allocates memory
            for e in missing:
                self._make_room(self.expert_nbytes, keep)
                key = (layer, e)
                block = self.domain.allocate(self.expert_nbytes, Category.WEIGHTS, f"blk.{layer}.expert{e}",
                                             refs=1, on_reclaim=partial(self._forget, key))  # fmt: skip
                self._experts[key] = _Entry(None, block)
                created.append(key)
            self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)
            self._load(layer, missing)
        except BaseException:
            for key in created:
                entry = self._experts.get(key)
                if entry is not None:
                    self.domain.release(entry.block)
                    self.domain.free_block(entry.block)  # the hook drops the entry
            for block in retained:
                self.domain.release(block)
            tel.expert_failed_acquisitions += 1
            raise
        tel.expert_hits += len(wanted) - len(missing)
        tel.expert_misses += len(missing)
        return {e: self._experts[(layer, e)].mats for e in wanted}

    def _make_room(self, nbytes: int, keep: set[tuple[int, int]]) -> None:
        """Reclaim unreferenced experts, least recently used first, until ``nbytes`` more fit under
        the ceiling and in the domain."""
        limit = self.expert_cache_bytes

        def short() -> bool:
            over = limit is not None and self.resident_bytes + nbytes > limit
            return over or not self.domain.can_allocate(nbytes, Category.WEIGHTS)

        for key in list(self._experts):
            if not short():
                return
            entry = self._experts.get(key)
            if entry is not None and key not in keep and entry.block.evictable:
                self.domain.free_block(entry.block)  # single reclamation path; the hook drops the entry
        if limit is not None and self.resident_bytes + nbytes > limit:
            raise ExpertBudgetExceeded(f"{self.resident_bytes} bytes of experts are in use; {nbytes} more would "
                                       f"exceed the ceiling of {limit} bytes")  # fmt: skip

    def _load(self, layer: int, missing: list[int]) -> None:
        """Dequantise the reserved experts.  At most ``workers`` loads are in flight, and the
        scratch memory they need is reserved (as workspace) before they start."""
        if not missing:
            return
        domain, tel = self.domain, self.telemetry
        width = min(self._workers, len(missing))
        scratch = None
        while scratch is None:
            try:
                scratch = domain.allocate(width * self._scratch[layer], Category.WORKSPACE, "expert loading", refs=1)
            except OutOfMemory:
                if width == 1:
                    raise
                width = max(1, width // 2)
        tel.peak_load_workspace_bytes = max(tel.peak_load_workspace_bytes, scratch.requested)

        def load(e: int):
            return tuple(self.tensors.load_slice(f"blk.{layer}.{kind}.weight", e) for kind in _EXPERTS)

        try:
            for i in range(0, len(missing), width):
                chunk = missing[i:i + width]
                futures = [self._pool.submit(load, e) for e in chunk]
                try:
                    results = [f.result() for f in futures]
                except BaseException:
                    for f in futures:
                        f.cancel()
                    wait(futures)  # no worker may still be writing when the transaction rolls back
                    raise
                for e, mats in zip(chunk, results):
                    if sum(m.nbytes for m in mats) != self.expert_nbytes:
                        raise RuntimeError(f"expert {(layer, e)}: loaded size differs from its reservation")
                    self._experts[(layer, e)].mats = mats
                    tel.expert_loads += 1
                    tel.expert_decoded_stored_bytes += self._expert_stored[layer]
                    tel.expert_decoded_bytes += self.expert_nbytes
        finally:
            domain.release(scratch, free_when_unused=True)

    def release(self, layer: int, experts) -> None:
        for e in sorted({int(e) for e in experts}):
            self.domain.release(self._experts[(layer, e)].block)

    @contextmanager
    def acquired(self, layer: int, experts):
        """``with store.acquired(layer, experts) as mats:`` -- fetch, then always release."""
        mats = self.fetch(layer, experts)
        try:
            yield mats
        finally:
            self.release(layer, mats)

    def expert(self, layer: int, e: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One expert (fetched and released at once; for callers outside a forward pass)."""
        with self.acquired(layer, [e]) as mats:
            return mats[e]

    def clear_cache(self) -> int:
        """Drop every expert that is not in use (an explicit cache state for a measurement);
        returns how many were dropped."""
        dropped = 0
        for key in list(self._experts):
            entry = self._experts.get(key)
            if entry is not None and entry.block.evictable:
                self.domain.free_block(entry.block)
                dropped += 1
        return dropped

    def check_invariant(self) -> None:
        """Every entry owns a live block; no entry is left half-loaded; the ceiling holds."""
        for key, entry in self._experts.items():
            assert entry.block.live, f"expert {key} keeps arrays for a freed block"
            assert entry.mats is not None, f"expert {key} was left half-loaded"
        if self.expert_cache_bytes is not None:
            assert self.resident_bytes <= self.expert_cache_bytes, "resident experts exceed the ceiling"

    def close(self) -> None:
        self._pool.shutdown(wait=True)
