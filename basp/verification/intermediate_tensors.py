"""Layer-by-layer comparison of hidden states (to locate where a divergence starts)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .logits import rel_l2


def compare_hidden(runtime: list[np.ndarray], reference: list[np.ndarray]) -> dict[str, Any]:
    """``runtime[i]``: output of layer i; ``reference``: transformers' ``hidden_states`` (the
    embedding, then each layer's output, the last one after the final norm)."""
    n = min(len(runtime), len(reference) - 2)  # the reference's last entry is normalised
    per_layer = [float(rel_l2(runtime[i], reference[i + 1]).max()) for i in range(n)]
    shapes_match = all(runtime[i].shape == reference[i + 1].shape for i in range(n))
    return {"layers_compared": n, "shapes_match": shapes_match, "rel_l2_per_layer": per_layer,
            "rel_l2_max": max(per_layer, default=0.0)}  # fmt: skip
