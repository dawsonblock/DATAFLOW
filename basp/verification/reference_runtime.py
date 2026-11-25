"""The reference: Hugging Face transformers running the same weights as the runtime.

``load_reference`` builds a transformers Qwen3 (or Qwen3-MoE) model from a GGUF file,
dequantising every tensor with the reference ``gguf`` package -- an implementation independent
of the runtime's -- so a disagreement is the runtime's computation, not its weights.
Needs PyTorch, transformers and gguf (``pip install torch transformers gguf``).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from basp.models.gguf import read_gguf
from basp.runtime.config import ModelConfig


class ReferenceUnavailable(ImportError):
    """PyTorch, transformers or gguf is not installed."""


def _imports():
    try:
        import gguf.quants
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ReferenceUnavailable("the reference needs torch, transformers and gguf installed") from exc
    return torch, transformers, gguf.quants


def hf_config(config: ModelConfig, transformers) -> Any:
    common = {"vocab_size": config.vocab, "hidden_size": config.hidden, "num_hidden_layers": config.n_layers,
              "num_attention_heads": config.n_head, "num_key_value_heads": config.n_kv_head,
              "head_dim": config.head_dim, "rms_norm_eps": config.eps,
              "max_position_embeddings": config.context_length, "tie_word_embeddings": config.tied_embeddings,
              "attention_bias": False, "torch_dtype": "float32"}  # fmt: skip
    rope = {"rope_parameters": {"rope_type": "default", "rope_theta": config.rope_theta}}
    if config.moe:
        cls = transformers.Qwen3MoeConfig
        extra = {"num_experts": config.n_experts, "num_experts_per_tok": config.top_k,
                 "moe_intermediate_size": config.expert_ffn, "intermediate_size": config.expert_ffn,
                 "norm_topk_prob": config.normalize_topk, "decoder_sparse_step": 1, "mlp_only_layers": []}  # fmt: skip
    else:
        cls, extra = transformers.Qwen3Config, {"intermediate_size": config.ffn}
    try:
        cfg = cls(**common, **extra, **rope)
    except TypeError:  # transformers 4: rope_theta is a plain argument
        cfg = cls(**common, **extra, rope_theta=config.rope_theta)
    if getattr(cfg, "rope_theta", config.rope_theta) != config.rope_theta and not hasattr(cfg, "rope_parameters"):
        raise RuntimeError("could not set the reference's RoPE base")
    return cfg


_NAMES = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
          "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
          "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
          "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj", "ffn_gate_inp": "mlp.gate"}  # fmt: skip


def load_reference(path: str | os.PathLike, *, device: str = "cpu"):
    """(transformers model in float32 eval mode, ModelConfig) for a GGUF file."""
    torch, transformers, quants = _imports()
    gguf = read_gguf(path)
    config = ModelConfig.from_gguf(gguf)
    cfg = hf_config(config, transformers)
    cls = transformers.Qwen3MoeForCausalLM if config.moe else transformers.Qwen3ForCausalLM
    model = cls(cfg).to(torch.float32)
    data = np.memmap(gguf.path, dtype=np.uint8, mode="r")
    weights: dict[str, np.ndarray] = {}
    for t in gguf.tensors:
        raw = data[gguf.data_offset + t.offset: gguf.data_offset + t.offset + t.nbytes]

        qtype = quants.GGMLQuantizationType(t.ggml_type)
        block, size = quants.GGML_QUANT_SIZES[qtype]
        shape = tuple(reversed(t.shape))
        byte_shape = (*shape[:-1], shape[-1] // block * size)
        weights[t.name] = np.asarray(quants.dequantize(np.asarray(raw).reshape(byte_shape), qtype),
                                     dtype=np.float32).reshape(shape)  # fmt: skip
    state: dict[str, Any] = {"model.embed_tokens.weight": weights["token_embd.weight"],
                             "model.norm.weight": weights["output_norm.weight"]}  # fmt: skip
    state["lm_head.weight"] = weights.get("output.weight", weights["token_embd.weight"])
    params = dict(model.named_parameters())
    for name, value in weights.items():
        if not name.startswith("blk."):
            continue
        _, layer, kind, _ = name.split(".")
        prefix = f"model.layers.{layer}."
        if kind in _NAMES:
            state[prefix + _NAMES[kind] + ".weight"] = value
        elif kind in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            if prefix + "mlp.experts.gate_up_proj" in params:  # transformers 5: fused experts
                if kind == "ffn_down_exps":
                    state[prefix + "mlp.experts.down_proj"] = value
                else:
                    state.setdefault(prefix + "mlp.experts._parts", {})[kind] = value
            else:  # transformers 4: one module per expert
                proj = {"ffn_gate_exps": "gate_proj", "ffn_up_exps": "up_proj", "ffn_down_exps": "down_proj"}[kind]
                for e in range(value.shape[0]):
                    state[f"{prefix}mlp.experts.{e}.{proj}.weight"] = value[e]
        else:
            raise RuntimeError(f"no reference parameter for {name}")
    for key in [k for k in state if k.endswith("._parts")]:
        parts = state.pop(key)
        state[key.replace("._parts", ".gate_up_proj")] = np.concatenate(
            [parts["ffn_gate_exps"], parts["ffn_up_exps"]], axis=1)  # fmt: skip
    tensors = {k: torch.from_numpy(np.array(v, dtype=np.float32, copy=True)) for k, v in state.items()}
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    missing = [m for m in missing if not (config.tied_embeddings and m == "lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"reference weights do not match: missing {missing[:4]}, unexpected {unexpected[:4]}")
    if config.tied_embeddings:
        model.tie_weights()
    model.eval()
    return model.to(device), config


def reference_forward(model, ids, *, capture_routing: bool = False):
    """Logits (T, vocab), hidden states and, for MoE models, the experts each router selected."""
    torch, _, _ = _imports()
    routing: list[np.ndarray] = []
    hooks = []
    if capture_routing:
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):  # transformers 5 router: (logits, scores, indices)
                routing.append(output[2].detach().cpu().numpy())
            else:  # transformers 4: the gate is a linear layer returning logits
                probs = torch.softmax(output.float(), dim=-1)
                k = model.config.num_experts_per_tok
                routing.append(torch.topk(probs, k, dim=-1).indices.cpu().numpy())

        for layer in model.model.layers:
            hooks.append(layer.mlp.gate.register_forward_hook(hook))
    try:
        with torch.no_grad():
            out = model(torch.tensor([list(map(int, ids))]), output_hidden_states=True, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    hidden = [h[0].float().numpy() for h in out.hidden_states]
    return out.logits[0].float().numpy(), hidden, routing
