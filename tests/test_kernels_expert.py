"""MoE expert execution, weighted reduction and routing against the float64 reference."""

import warnings

import numpy as np
import pytest

from basp.kernels import reference
from basp.kernels.expert import QuantizedExpert, moe_combine, run_expert, silu_fp32
from basp.kernels.int4 import quantize_int4
from basp.kernels.matvec import KernelStats, matmul_int4

U32 = 2.0**-24  # FP32 unit roundoff
HIDDEN, INTER = 64, 96

# End-to-end budget against the unrounded float64 reference on the same decoded weights.
# The kernel rounds activations to FP16 twice (input and SwiGLU hidden state), each at most
# 2**-11 relative per element; SwiGLU multiplies two projections of the input, so the first
# rounding can count twice. 4 * 2**-11 covers that; observed error is several times smaller.
E2E_REL_L2 = 4 * 2.0**-11


def _expert(seed, hidden=HIDDEN, inter=INTER):
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(hidden)
    return QuantizedExpert.from_float(rng.normal(size=(inter, hidden)) * scale,
                                      rng.normal(size=(inter, hidden)) * scale,
                                      rng.normal(size=(hidden, inter)) * scale)


def _rel_l2(got, want):
    return np.linalg.norm(got - want) / np.linalg.norm(want)


@pytest.mark.parametrize("batched", [False, True])
def test_expert_matches_reference_on_dequantized_weights(batched):
    expert = _expert(20)
    rng = np.random.default_rng(21)
    x = rng.normal(size=(6, HIDDEN) if batched else HIDDEN)
    y = run_expert(expert, x)
    assert y.dtype == np.float32 and y.shape == x.shape
    assert _rel_l2(y, reference.swiglu_expert(x, *expert.dequantized())) < E2E_REL_L2


def test_expert_stages_chain_to_the_reference():
    """Each stage of the fused expert is tied to the reference with a rigorous tolerance."""
    expert = _expert(22)
    rng = np.random.default_rng(23)
    x = rng.normal(size=(4, HIDDEN))
    w_gate, w_up, w_down = (w.astype(np.float64) for w in expert.dequantized())
    x16 = x.astype(np.float16).astype(np.float64)

    def fp32_tol(t, abs_w, abs_in):  # see tests/test_kernels_matvec.py::_fp32_tol
        return 2 * (t.group_size + t.n_groups + 2) * U32 * (abs_in @ abs_w.T) + 1e-30

    gate = matmul_int4(expert.gate, x)
    up = matmul_int4(expert.up, x)
    assert np.all(np.abs(gate - reference.matmul(w_gate, x16))
                  <= fp32_tol(expert.gate, np.abs(w_gate), np.abs(x16)))
    assert np.all(np.abs(up - reference.matmul(w_up, x16))
                  <= fp32_tol(expert.up, np.abs(w_up), np.abs(x16)))
    # FP32 activation function vs float64: a handful of FP32 rounding steps.
    assert np.allclose(silu_fp32(gate), reference.silu(gate), rtol=1e-6, atol=1e-30)
    hidden = silu_fp32(gate) * up
    h16 = hidden.astype(np.float16).astype(np.float64)
    out = run_expert(expert, x)
    assert np.all(np.abs(out - reference.matmul(w_down, h16))
                  <= fp32_tol(expert.down, np.abs(w_down), np.abs(h16)))
    # And the fused kernel is exactly this composition of the public kernels.
    assert np.array_equal(out, matmul_int4(expert.down, hidden))


def test_expert_batch_rows_and_tiling_are_bit_exact():
    expert = _expert(24)
    x = np.random.default_rng(25).normal(size=(5, HIDDEN))
    batch = run_expert(expert, x)
    assert np.array_equal(batch, np.stack([run_expert(expert, row) for row in x]))
    for tile_rows in (1, 5, 1000):
        assert np.array_equal(run_expert(expert, x, tile_rows=tile_rows), batch)


def test_quantized_expert_sizes_and_stats():
    expert = _expert(26)
    assert expert.n_weights == 3 * HIDDEN * INTER
    assert expert.nbytes == expert.n_weights * 4.5 / 8
    w_gate, w_up, w_down = expert.dequantized()
    assert w_gate.shape == w_up.shape == (INTER, HIDDEN) and w_down.shape == (HIDDEN, INTER)
    assert all(w.dtype == np.float32 for w in (w_gate, w_up, w_down))
    stats = KernelStats()
    run_expert(expert, np.ones((3, HIDDEN)), stats=stats)
    assert stats.weight_bytes_read == expert.nbytes  # each matrix streamed exactly once
    assert stats.macs == 3 * expert.n_weights
    with pytest.raises(ValueError):
        QuantizedExpert(expert.gate, expert.up, quantize_int4(np.zeros((HIDDEN, INTER + 32))))
    with pytest.raises(ValueError):
        run_expert(expert, np.ones((2, 2, HIDDEN)))


def test_silu_is_stable_for_large_inputs():
    x = np.array([-1e4, -50.0, 0.0, 50.0, 1e4])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y64 = reference.silu(x)
        y32 = silu_fp32(x)
    assert y32.dtype == np.float32
    assert y64[2] == 0 and y64[0] == 0 and y64[4] == 1e4
    assert np.allclose(y64[[1, 3]], [-50.0 / (1 + np.exp(50.0)), 50.0])
    assert np.allclose(y32, y64, rtol=1e-6)


def test_moe_combine_matches_weighted_reduce():
    rng = np.random.default_rng(27)
    outputs = [rng.normal(size=(3, HIDDEN)).astype(np.float32) for _ in range(8)]
    coeffs = rng.dirichlet(np.ones(8))
    y = moe_combine(outputs, coeffs)
    assert y.dtype == np.float32 and y.shape == (3, HIDDEN)
    # k rounded products (FP32 coefficient included) and k FP32 additions.
    magnitude = sum(abs(c) * np.abs(o).astype(np.float64) for c, o in zip(coeffs, outputs))
    assert np.all(np.abs(y - reference.weighted_reduce(outputs, coeffs))
                  <= 2 * (2 * len(outputs) + 2) * U32 * magnitude)
    with pytest.raises(ValueError):
        moe_combine(outputs, coeffs[:-1])
    with pytest.raises(ValueError):
        moe_combine([], [])
    with pytest.raises(ValueError):
        reference.weighted_reduce(outputs, coeffs[:-1])


def test_moe_layer_matches_reference_end_to_end():
    """Route with the reference router, run the selected experts, reduce with coefficients."""
    n_experts, top_k = 8, 2
    experts = [_expert(100 + i) for i in range(n_experts)]
    rng = np.random.default_rng(28)
    x = rng.normal(size=HIDDEN)
    router = rng.normal(size=(n_experts, HIDDEN))
    selected, coeffs = reference.route_topk(reference.matvec(router, x), top_k)
    y = moe_combine([run_expert(experts[i], x) for i in selected], coeffs)
    want = reference.weighted_reduce(
        [reference.swiglu_expert(x, *experts[i].dequantized()) for i in selected], coeffs)
    assert _rel_l2(y, want) < E2E_REL_L2


def test_route_topk_properties():
    rng = np.random.default_rng(29)
    logits = rng.normal(size=16)
    probs = np.exp(logits) / np.exp(logits).sum()
    idx, coeffs = reference.route_topk(logits, 4)
    assert idx.dtype == np.int64 and coeffs.dtype == np.float64
    assert idx.tolist() == np.argsort(-logits)[:4].tolist()
    assert coeffs.sum() == pytest.approx(1.0)
    assert np.all(np.diff(coeffs) <= 0)
    assert np.allclose(coeffs, probs[idx] / probs[idx].sum())
    _, raw = reference.route_topk(logits, 4, normalize=False)
    assert np.allclose(raw, probs[idx])
    full_idx, full = reference.route_topk(logits, 16)
    assert sorted(full_idx.tolist()) == list(range(16)) and np.allclose(full, probs[full_idx])
    with pytest.raises(ValueError):
        reference.route_topk(logits, 0)
    with pytest.raises(ValueError):
        reference.route_topk(logits, 17)


def test_route_topk_tie_breaking_and_batch():
    idx, coeffs = reference.route_topk(np.array([1.0, 3.0, 3.0, 0.0, 3.0]), 2)
    assert idx.tolist() == [1, 2] and np.allclose(coeffs, [0.5, 0.5])
    # Saturated softmax must not reorder distinct logits.
    idx, _ = reference.route_topk(np.array([0.0, -2000.0, -1000.0, 5.0]), 3)
    assert idx.tolist() == [3, 0, 2]
    rng = np.random.default_rng(30)
    batch = rng.normal(size=(5, 12))
    bidx, bcoeffs = reference.route_topk(batch, 3)
    assert bidx.shape == bcoeffs.shape == (5, 3)
    assert np.allclose(bcoeffs.sum(axis=1), 1.0)
    for row, (ri, rc) in zip(batch, zip(bidx, bcoeffs)):
        idx, coeffs = reference.route_topk(row, 3)
        assert idx.tolist() == ri.tolist() and np.allclose(coeffs, rc)


def test_reference_rms_norm_and_matmul():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(4, HIDDEN)) * 7
    y = reference.rms_norm(x, np.ones(HIDDEN), eps=0.0)
    assert np.allclose(np.mean(y * y, axis=-1), 1.0)
    weight = rng.normal(size=HIDDEN)
    assert np.allclose(reference.rms_norm(x, weight, eps=0.0), y * weight)
    w = rng.normal(size=(9, HIDDEN))
    assert np.allclose(reference.matmul(w, x), np.stack([reference.matvec(w, row) for row in x]))
    with pytest.raises(ValueError):
        reference.matvec(w, x)
    with pytest.raises(ValueError):
        reference.matmul(w, x[0])
