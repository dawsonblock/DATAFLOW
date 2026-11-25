"""Model weights in host memory: dense weights resident, experts loaded on demand.

Every byte is charged to the host's memory domain (``basp.memory``): dense weights as pinned
``weights`` blocks; experts as unpinned blocks in a bounded least-recently-used cache, so an
expert can be dropped and re-read from the memory-mapped file.  An expert in use by the
current operation is referenced and is never evicted.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from basp.memory import Category, MemoryDomain
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
    def __init__(self, tensors: GGUFTensors, config: ModelConfig, domain: MemoryDomain, telemetry: Telemetry,
                 *, expert_cache_bytes: int | None = None) -> None:  # fmt: skip
        self.tensors, self.config, self.domain, self.telemetry = tensors, config, domain, telemetry
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
        self._dense: dict[str, np.ndarray] = {}
        for name in expected:
            if any(f".{kind}." in name for kind in _EXPERTS):
                continue
            array = tensors.load(name)
            domain.allocate(array.nbytes, Category.WEIGHTS, name, pinned=True)
            self._dense[name] = array
        self.expert_cache_bytes = expert_cache_bytes
        self._experts: OrderedDict[tuple[int, int], tuple[tuple[np.ndarray, ...], object]] = OrderedDict()

    def __getitem__(self, name: str) -> np.ndarray:
        return self._dense[name]

    @property
    def output(self) -> np.ndarray:
        return self._dense["token_embd.weight" if self.config.tied_embeddings else "output.weight"]

    def expert(self, layer: int, e: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (layer, e)
        if key in self._experts:
            self._experts.move_to_end(key)
            self.telemetry.expert_hits += 1
            return self._experts[key][0]
        mats = tuple(self.tensors.load_slice(f"blk.{layer}.{kind}.weight", e) for kind in _EXPERTS)
        nbytes = sum(m.nbytes for m in mats)
        limit = self.expert_cache_bytes
        while self._experts and (limit is not None and self._cached_bytes() + nbytes > limit
                                 or not self.domain.can_allocate(nbytes, Category.WEIGHTS)):  # fmt: skip
            _, (_, block) = self._experts.popitem(last=False)
            self.domain.free_block(block)
        block = self.domain.allocate(nbytes, Category.WEIGHTS, f"blk.{layer}.expert{e}")
        self._experts[key] = (mats, block)
        self.telemetry.expert_loads += 1
        return mats

    def _cached_bytes(self) -> int:
        return sum(block.size for _, block in self._experts.values())
