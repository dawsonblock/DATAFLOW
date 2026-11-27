"""Model hyper-parameters read from GGUF metadata (Qwen3 dense and Qwen3-MoE)."""

from __future__ import annotations

from dataclasses import dataclass

from basp.models.gguf import GGUFFile

SUPPORTED_ARCHITECTURES = ("qwen3", "qwen3moe")


class UnsupportedModel(ValueError):
    """A model architecture or feature the runtime does not implement."""


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    n_layers: int
    hidden: int
    n_head: int
    n_kv_head: int
    head_dim: int
    ffn: int  # dense FFN width (0 for MoE models)
    vocab: int
    rope_theta: float
    eps: float
    context_length: int
    n_experts: int = 0
    top_k: int = 0
    expert_ffn: int = 0
    normalize_topk: bool = True
    tied_embeddings: bool = False

    @property
    def moe(self) -> bool:
        return self.n_experts > 0

    @classmethod
    def from_gguf(cls, gguf: GGUFFile) -> ModelConfig:
        arch = gguf.architecture
        if arch not in SUPPORTED_ARCHITECTURES:
            raise UnsupportedModel(f"architecture {arch!r} is not supported (supported: {SUPPORTED_ARCHITECTURES})")

        def need(key: str):
            value = gguf.metadata.get(f"{arch}.{key}")
            if value is None:
                raise UnsupportedModel(f"GGUF metadata lacks {arch}.{key}")
            return value

        n_head = int(need("attention.head_count"))
        hidden = int(need("embedding_length"))
        head_dim = int(gguf.metadata.get(f"{arch}.attention.key_length", hidden // n_head))
        if int(gguf.metadata.get(f"{arch}.attention.value_length", head_dim)) != head_dim:
            raise UnsupportedModel("key and value head sizes differ")
        if gguf.metadata.get(f"{arch}.rope.scaling.type", "none") not in ("none", None):
            raise UnsupportedModel("RoPE scaling is not implemented")
        embd = gguf.tensor("token_embd.weight")
        if embd is None:
            raise UnsupportedModel("no token_embd.weight tensor")
        n_experts = int(gguf.metadata.get(f"{arch}.expert_count", 0) or 0)
        return cls(
            architecture=arch, n_layers=int(need("block_count")), hidden=hidden, n_head=n_head,
            n_kv_head=int(gguf.metadata.get(f"{arch}.attention.head_count_kv", n_head)), head_dim=head_dim,
            ffn=0 if n_experts else int(need("feed_forward_length")), vocab=int(embd.shape[1]),
            rope_theta=float(gguf.metadata.get(f"{arch}.rope.freq_base", 10000.0)),
            eps=float(need("attention.layer_norm_rms_epsilon")),
            context_length=int(gguf.metadata.get(f"{arch}.context_length", 4096)),
            n_experts=n_experts, top_k=int(gguf.metadata.get(f"{arch}.expert_used_count", 0) or 0),
            expert_ffn=int(gguf.metadata.get(f"{arch}.expert_feed_forward_length", 0) or 0),
            normalize_topk=bool(gguf.metadata.get(f"{arch}.expert_weights_norm", True)),
            tied_embeddings=gguf.tensor("output.weight") is None,
        )  # fmt: skip
