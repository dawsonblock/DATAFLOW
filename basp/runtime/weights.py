"""Model weights in host memory: dense weights resident, experts loaded on demand.

Every byte is charged to the host's memory domain (``basp.memory``): dense weights as pinned
``weights`` blocks; experts as unpinned blocks in a bounded least-recently-used cache, so an
expert can be dropped and re-read from the memory-mapped file.  An expert in use by the
current operation is referenced and is never evicted.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from basp.memory import Block, Category, MemoryDomain
from basp.models.gguf_tensors import GGUFTensors

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
        self.rows_embedding = embedding == "rows" and not config.tied_embeddings
        self._pool = ThreadPoolExecutor(max_workers=workers or min(4, os.cpu_count() or 1))
        names = [n for n in expected if not any(f".{kind}." in n for kind in _EXPERTS)
                 and not (self.rows_embedding and n == "token_embd.weight")]  # fmt: skip
        self._dense: dict[str, np.ndarray] = {}
        for name, array in zip(names, self._pool.map(tensors.load, names)):
            domain.allocate(array.nbytes, Category.WEIGHTS, name, pinned=True)
            self._dense[name] = array
        self.expert_cache_bytes = expert_cache_bytes
        self._experts: OrderedDict[tuple[int, int], tuple[tuple[np.ndarray, ...], Block]] = OrderedDict()
        # stored (quantised) bytes of one expert, per layer: formats can differ between layers
        self._expert_stored = [sum(tensors.slice_nbytes(f"blk.{i}.{k}.weight") for k in _EXPERTS)
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

    # -- experts ------------------------------------------------------------------------------
    def fetch(self, layer: int, experts) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Dequantised weights of ``experts`` of one layer, loading missing ones in parallel.
        Each returned expert is referenced (cannot be evicted) until ``release``."""
        wanted = sorted({int(e) for e in experts})
        self.telemetry.expert_requests += len(wanted)
        self.telemetry.expert_bytes += len(wanted) * self._expert_stored[layer]
        out = {}
        missing = []
        for e in wanted:
            entry = self._experts.get((layer, e))
            if entry is None:
                missing.append(e)
                continue
            self._experts.move_to_end((layer, e))
            self.domain.retain(entry[1])
            self.telemetry.expert_hits += 1
            out[e] = entry[0]

        def load(e: int):
            return tuple(self.tensors.load_slice(f"blk.{layer}.{kind}.weight", e) for kind in _EXPERTS)

        for e, mats in zip(missing, self._pool.map(load, missing)):
            nbytes = sum(m.nbytes for m in mats)
            self._make_room(nbytes)
            block = self.domain.allocate(nbytes, Category.WEIGHTS, f"blk.{layer}.expert{e}", refs=1)
            self._experts[(layer, e)] = (mats, block)
            self.telemetry.expert_loads += 1
            out[e] = mats
        return out

    def release(self, layer: int, experts) -> None:
        for e in {int(e) for e in experts}:
            self.domain.release(self._experts[(layer, e)][1])

    def expert(self, layer: int, e: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One expert (fetched and released at once; for callers outside a forward pass)."""
        mats = self.fetch(layer, [e])[e]
        self.release(layer, [e])
        return mats

    def _make_room(self, nbytes: int) -> None:
        limit = self.expert_cache_bytes
        for key in list(self._experts):
            over_limit = limit is not None and self._cached_bytes() + nbytes > limit
            if not over_limit and self.domain.can_allocate(nbytes, Category.WEIGHTS):
                return
            block = self._experts[key][1]
            if block.evictable:  # an expert in use by the current call is never evicted
                del self._experts[key]
                self.domain.free_block(block)

    def _cached_bytes(self) -> int:
        return sum(block.size for _, block in self._experts.values())

    def close(self) -> None:
        self._pool.shutdown(wait=True)
