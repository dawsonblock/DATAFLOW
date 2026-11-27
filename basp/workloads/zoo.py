"""Model shapes for the benchmark suite, from 4 billion to 3 trillion parameters.

Dense transformers are represented as a mixture of one expert per layer that is
always selected (the feed-forward block), so the same dataflow machinery
applies.  Shapes marked "published config" were read from the model's public
``config.json``; "hypothetical" shapes exist only to probe scaling.  Only the
shapes are used -- no weights, and no claim about any model's quality.
"""

from __future__ import annotations

from basp.workloads.model import QWEN3_30B_A3B, MoEModelSpec, tiny_spec


def _dense(name: str, hidden: int, layers: int, heads: int, kv_heads: int, intermediate: int,
           vocab: int, *, tied: bool = False, qk_norm: bool = False,
           source: str = "published config") -> MoEModelSpec:  # fmt: skip
    return MoEModelSpec(name, hidden=hidden, n_layers=layers, n_experts=1, top_k=1,
                        expert_inter=intermediate, n_heads=heads, n_kv_heads=kv_heads,
                        head_dim=128, vocab=vocab, tied_embeddings=tied, qk_norm=qk_norm,
                        source=source)  # fmt: skip


QWEN3_4B = _dense("Qwen3-4B", 2560, 36, 32, 8, 9728, 151936, tied=True, qk_norm=True)
QWEN3_8B = _dense("Qwen3-8B", 4096, 36, 32, 8, 12288, 151936, qk_norm=True)
QWEN3_14B = _dense("Qwen3-14B", 5120, 40, 40, 8, 17408, 151936, qk_norm=True)
LLAMA_70B = _dense("Llama-3.3-70B", 8192, 80, 64, 8, 28672, 128256)

# DeepSeek-V3: 61 layers, 256 routed experts (8 selected) plus one shared expert, hidden 7168,
# expert intermediate size 2048.  Two simplifications keep the uniform-layer shape: its
# multi-head latent attention is entered as a parameter count (about 187 M per layer), and its
# three dense layers are folded into the trunk blocks of the 58 mixture layers.
_DSV3_ATTENTION = 7168 * 1536 + 1536 * 128 * 192 + 7168 * 576 + 512 * 128 * 256 + 128 * 128 * 7168
_DSV3_DENSE_FFN = 3 * 7168 * 18432
DEEPSEEK_V3 = MoEModelSpec(
    "DeepSeek-V3", hidden=7168, n_layers=58, n_experts=256, top_k=8, expert_inter=2048,
    n_heads=128, n_kv_heads=2, head_dim=128, vocab=129280, n_shared_experts=1,
    attn_params_override=(61 * _DSV3_ATTENTION + 3 * _DSV3_DENSE_FFN) // 58,
    source="published config, approximated",
)  # fmt: skip

MOE_1T = MoEModelSpec(
    "MoE-1T (hypothetical)", hidden=8192, n_layers=62, n_experts=320, top_k=8, expert_inter=2048,
    n_heads=64, n_kv_heads=8, head_dim=128, vocab=131072, source="hypothetical",
)  # fmt: skip
MOE_3T = MoEModelSpec(
    "MoE-3T (hypothetical)", hidden=12288, n_layers=80, n_experts=512, top_k=8, expert_inter=2048,
    n_heads=96, n_kv_heads=8, head_dim=128, vocab=131072, source="hypothetical",
)  # fmt: skip

#: benchmark suite, smallest to largest: (key, shape, research purpose)
SUITE: tuple[tuple[str, MoEModelSpec, str], ...] = (
    ("qwen3-4b", QWEN3_4B, "kernel validation"),
    ("qwen3-8b", QWEN3_8B, "fully GPU-resident baseline"),
    ("qwen3-14b", QWEN3_14B, "memory pressure and quantization"),
    ("qwen3-30b-a3b", QWEN3_30B_A3B, "expert routing and CPU/GPU offloading"),
    ("llama-3.3-70b", LLAMA_70B, "high-capacity heterogeneous execution"),
    ("deepseek-v3", DEEPSEEK_V3, "large-scale simulation"),
    ("moe-1t", MOE_1T, "long-term architectural feasibility"),
    ("moe-3t", MOE_3T, "long-term architectural feasibility"),
)

MODELS: dict[str, MoEModelSpec] = {key: spec for key, spec, _ in SUITE}
MODELS["tiny"] = tiny_spec()
