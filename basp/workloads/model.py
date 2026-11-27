"""Mixture-of-experts model shapes used as workloads.

Only shapes matter to the dataflow engine: parameter counts fix weight bytes and
multiply-accumulate counts.  ``QWEN3_30B_A3B`` uses the published configuration
of Qwen/Qwen3-30B-A3B (hidden 2048, 48 layers, 128 experts of intermediate size
768, 8 selected per token, 32 heads / 4 KV heads of dimension 128, vocabulary
151936, untied embeddings), which reproduces the published totals of roughly
30.5 billion parameters and 3.3 billion activated per token.
"""

from __future__ import annotations

from dataclasses import dataclass

from basp.core.tensor import CompressionFormat


@dataclass(frozen=True)
class MoEModelSpec:
    name: str
    hidden: int
    n_layers: int
    n_experts: int
    top_k: int
    expert_inter: int
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    vocab: int = 0
    tied_embeddings: bool = False
    weight_format: CompressionFormat = CompressionFormat.INT4
    #: experts that every token uses in addition to the routed ones (part of the trunk block)
    n_shared_experts: int = 0
    #: per-head RMSNorm of queries and keys (Qwen3): one head_dim weight vector each per layer
    qk_norm: bool = False
    #: RMSNorm before the output projection (every real model here; the small numeric model has none)
    final_norm: bool = True
    #: attention parameters per layer when the architecture is not plain QKVO projections
    attn_params_override: int | None = None
    #: provenance of the shape: "published config", "representative" or "hypothetical"
    source: str = "representative"

    def __post_init__(self) -> None:
        if not 0 < self.top_k <= self.n_experts:
            raise ValueError("top_k must be in (0, n_experts]")

    @property
    def dense(self) -> bool:
        """A dense transformer: one feed-forward block per layer, always used."""
        return self.n_experts == 1

    @property
    def expert_params(self) -> int:
        """Gate, up and down projections of one SwiGLU expert."""
        return 3 * self.hidden * self.expert_inter

    @property
    def attn_params(self) -> int:
        """Query, key, value and output projections of one layer."""
        if self.attn_params_override is not None:
            return self.attn_params_override
        proj = self.hidden * self.head_dim * (self.n_heads + 2 * self.n_kv_heads)
        return proj + self.n_heads * self.head_dim * self.hidden

    @property
    def router_params(self) -> int:
        return 0 if self.dense else self.n_experts * self.hidden

    @property
    def shared_params(self) -> int:
        """Always-active experts of one layer."""
        return self.n_shared_experts * self.expert_params

    @property
    def norm_params(self) -> int:
        # two RMSNorm weight vectors per layer, plus the query and key norms where the model has them
        return 2 * self.hidden + (2 * self.head_dim if self.qk_norm else 0)

    @property
    def trunk_params(self) -> int:
        """Parameters of one layer that every token uses (everything but routed experts)."""
        return self.attn_params + self.router_params + self.norm_params + self.shared_params

    @property
    def embed_params(self) -> int:
        return self.vocab * self.hidden

    @property
    def head_params(self) -> int:
        return self.vocab * self.hidden

    @property
    def final_norm_params(self) -> int:
        return self.hidden if self.final_norm and self.vocab else 0

    @property
    def embedding_storage_params(self) -> int:
        """Unique storage of the embedding table and the output projection (tied: one table)."""
        return self.embed_params * (1 if self.tied_embeddings else 2)

    @property
    def total_params(self) -> int:
        per_layer = self.n_experts * self.expert_params + self.trunk_params
        return self.n_layers * per_layer + self.embedding_storage_params + self.final_norm_params

    @property
    def active_params(self) -> int:
        """"Activated" parameters as model cards count them: everything but the routed experts a
        token does not select (both embedding tables included)."""
        per_layer = self.top_k * self.expert_params + self.trunk_params
        return self.n_layers * per_layer + self.embedding_storage_params + self.final_norm_params

    @property
    def params_read_per_token(self) -> int:
        """Parameters actually read to produce one token: every layer's trunk block and ``top_k``
        routed experts, one embedding row, the final norm and the whole output projection."""
        per_layer = self.top_k * self.expert_params + self.trunk_params
        embedding_row = self.hidden if self.vocab else 0
        return self.n_layers * per_layer + embedding_row + self.final_norm_params + self.head_params

    def layer_accounting(self, *, batch: int = 1, mixed_context: int = 0,
                         selected: int | None = None, unique: int | None = None) -> dict[str, int]:  # fmt: skip
        """Arithmetic and parameter reads of one layer for ``batch`` rows, from the model's
        dimensions alone (independent of how the compiler lowers it).

        ``selected``: token-expert pairs (default ``batch * top_k``); ``unique``: distinct routed
        experts those pairs use (default ``min(selected, n_experts)``).
        """
        selected = batch * self.top_k if selected is None else selected
        unique = min(selected, self.n_experts) if unique is None else unique
        norm_macs = batch * self.hidden * (2 if self.n_heads else 1)
        attention_macs = batch * self.attention_macs(mixed_context) if self.n_heads else 0
        return {
            "trunk_macs": norm_macs + attention_macs + batch * self.router_params + batch * self.shared_params,
            "trunk_params_read": self.trunk_params,  # norms, attention, router and shared experts, once
            "expert_macs": selected * (self.expert_params + self.expert_inter),  # three projections and the gate
            "expert_params_read": unique * self.expert_params,
            "reduce_macs": batch * self.hidden * (min(self.top_k, unique) + self.n_shared_experts + 1),
        }  # fmt: skip

    def step_accounting(self, *, batch: int = 1, head_rows: int | None = None) -> dict[str, int]:
        """Arithmetic and parameter reads of a step outside the layers: the embedding lookup, the
        final norm and the output projection."""
        rows = batch if head_rows is None else head_rows
        has_vocab = bool(self.vocab)
        return {
            "embed_macs": batch * self.hidden if has_vocab else 0,  # one row copied per token
            "embed_params_read": batch * self.hidden if has_vocab else 0,
            "final_norm_macs": rows * self.hidden if self.final_norm_params else 0,
            "final_norm_params_read": self.final_norm_params,
            "head_macs": rows * self.head_params,
            "head_params_read": self.head_params,
        }

    @property
    def kv_bytes_per_token_layer(self) -> int:
        """FP16 keys and values appended per token in one layer."""
        return 2 * self.n_kv_heads * self.head_dim * 2

    def attention_macs(self, context_len: int) -> int:
        """Projection MACs plus score and value mixing over the cached context."""
        return self.attn_params + 2 * self.n_heads * self.head_dim * context_len

    def bytes_at(self, bits_per_weight: float = 4.5) -> float:
        return self.total_params * bits_per_weight / 8.0


QWEN3_30B_A3B = MoEModelSpec(
    name="Qwen3-30B-A3B",
    hidden=2048,
    n_layers=48,
    n_experts=128,
    top_k=8,
    expert_inter=768,
    n_heads=32,
    n_kv_heads=4,
    head_dim=128,
    vocab=151936,
    qk_norm=True,
    source="published config",
)


def tiny_spec(**overrides) -> MoEModelSpec:
    """A small attention-free MoE stack for tests and numerical co-simulation."""
    base = {
        "name": "tiny-moe", "hidden": 64, "n_layers": 3, "n_experts": 8, "top_k": 2,
        "expert_inter": 96, "vocab": 256, "final_norm": False,
    }  # fmt: skip
    base.update(overrides)
    return MoEModelSpec(**base)
