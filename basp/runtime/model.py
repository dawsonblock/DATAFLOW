"""The Qwen3 (dense and MoE) forward pass, issued operation by operation to the dispatcher."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ModelConfig
from .dispatcher import Dispatcher
from .weights import WeightStore


class ContextOverflow(RuntimeError):
    """More tokens than the session's KV cache holds."""


class KVCache:
    def __init__(self, config: ModelConfig, max_tokens: int) -> None:
        shape = (config.n_layers, max_tokens, config.n_kv_head, config.head_dim)
        self.k = np.zeros(shape, np.float32)
        self.v = np.zeros(shape, np.float32)
        self.max_tokens = max_tokens

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes

    def write(self, layer: int, positions: np.ndarray, k: np.ndarray, v: np.ndarray) -> None:
        if positions.max() >= self.max_tokens:
            raise ContextOverflow(f"position {int(positions.max())} exceeds the cache of {self.max_tokens} tokens")
        self.k[layer, positions] = k
        self.v[layer, positions] = v


@dataclass
class ForwardOutput:
    logits: np.ndarray  # (rows, vocab)
    experts: list[np.ndarray] = field(default_factory=list)  # per layer (T, top_k), MoE only
    router_logits: list[np.ndarray] = field(default_factory=list)
    expert_weights: list[np.ndarray] = field(default_factory=list)  # per layer (T, top_k), MoE only
    hidden: list[np.ndarray] = field(default_factory=list)  # per layer output, if captured


class Qwen3Model:
    def __init__(self, config: ModelConfig, weights: WeightStore, dispatcher: Dispatcher) -> None:
        self.config, self.w, self.d = config, weights, dispatcher

    def forward(self, ids: np.ndarray, positions: np.ndarray, kv: KVCache, *, all_logits: bool = False,
                capture: bool = False) -> ForwardOutput:  # fmt: skip
        c, w, run = self.config, self.w, self.d.run
        ids = np.asarray(ids, dtype=np.int64)
        if ids.min() < 0 or ids.max() >= c.vocab:
            raise ValueError(f"token id outside the vocabulary of {c.vocab}")
        t = len(ids)
        out = ForwardOutput(np.empty(0))
        self.d.telemetry.forward_calls += 1
        self.d.telemetry.forward_rows += t
        x = run("embedding", "embedding", w.embed, ids)
        for i in range(c.n_layers):
            p = f"blk.{i}."
            h = run("attention", "rms_norm", x, w[p + "attn_norm.weight"], eps=c.eps)
            q = run("attention", "matmul", h, w[p + "attn_q.weight"]).reshape(t, c.n_head, c.head_dim)
            k = run("attention", "matmul", h, w[p + "attn_k.weight"]).reshape(t, c.n_kv_head, c.head_dim)
            v = run("attention", "matmul", h, w[p + "attn_v.weight"]).reshape(t, c.n_kv_head, c.head_dim)
            q = run("attention", "head_rms_norm", q, w[p + "attn_q_norm.weight"], eps=c.eps)
            k = run("attention", "head_rms_norm", k, w[p + "attn_k_norm.weight"], eps=c.eps)
            q = run("attention", "rope", q, positions, theta=c.rope_theta)
            k = run("attention", "rope", k, positions, theta=c.rope_theta)
            kv.write(i, positions, k, v)
            a = run("attention", "attention", q, kv.k[i], kv.v[i], positions)
            x = run("attention", "add", x, run("attention", "matmul", a, w[p + "attn_output.weight"]))
            if c.moe:
                h = run("router", "rms_norm", x, w[p + "ffn_norm.weight"], eps=c.eps)
                experts, weights, router_logits = run("router", "moe_router", h, w[p + "ffn_gate_inp.weight"],
                                                      top_k=c.top_k, normalize=c.normalize_topk)  # fmt: skip
                # Grouped execution: each selected expert is fetched once and serves every row that
                # chose it.  If the layer's union does not fit the expert ceiling, it is computed in
                # bounded groups; groups are ascending and accumulate into one buffer, so the
                # additions happen in exactly the order of a single pass.
                y = None
                for group in self.w.groups(i, np.unique(experts)):
                    with self.w.acquired(i, group) as mats:
                        y = run("experts", "moe_experts", h, experts, weights, mats.__getitem__, only=group, out=y)
                self.d.telemetry.expert_token_pairs += experts.size
                x = run("experts", "add", x, y)
                out.experts.append(experts)
                out.router_logits.append(router_logits)
                out.expert_weights.append(weights)
            else:
                h = run("ffn", "rms_norm", x, w[p + "ffn_norm.weight"], eps=c.eps)
                y = run("ffn", "swiglu", h, w[p + "ffn_gate.weight"], w[p + "ffn_up.weight"], w[p + "ffn_down.weight"])
                x = run("ffn", "add", x, y)
            if capture:
                out.hidden.append(x.copy())
        rows = x if all_logits else x[-1:]
        h = run("output", "rms_norm", rows, w["output_norm.weight"], eps=c.eps)
        out.logits = run("output", "matmul", h, w.output)
        return out
