"""v0.4 physical runtime: GGUF tensor data, dequantisation, the NumPy CPU backend, execution
plans that fail safely, sessions, the KV cache, expert caching, and unavailable backends."""

from __future__ import annotations

import numpy as np
import pytest

from basp.backends.cuda import CUDABackend
from basp.backends.execution import BackendUnavailable, Op, UnsupportedOperation
from basp.backends.ggml import GGMLBackend
from basp.backends.metal import MetalBackend
from basp.backends.remote import RemoteBackend
from basp.memory import Category
from basp.models.gguf import read_gguf
from basp.models.gguf_tensors import GGUFTensors
from basp.models.gguf_writer import tiny_qwen3, write_gguf
from basp.models.quants import UnsupportedFormat, dequantize, quantize_q8_0
from basp.runtime import (
    ContextOverflow,
    DeviceManager,
    Engine,
    UnsupportedModel,
    generate,
    make_plan,
)

gguf = pytest.importorskip("gguf")  # the reference package (a dev dependency)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    paths = {}
    for moe in (False, True):
        for wt in ("F32", "Q8_0"):
            path = root / f"{'moe' if moe else 'dense'}-{wt}.gguf"
            tiny_qwen3(path, moe=moe, weight_type=wt, seed=4)
            paths[(moe, wt)] = path
    return paths


# -- dequantisation against the reference package ----------------------------------------------
def _random_blocks(name: str, n: int, rng) -> np.ndarray:
    qtype = gguf.GGMLQuantizationType[name]
    _, size = gguf.GGML_QUANT_SIZES[qtype]
    raw = rng.integers(0, 256, size=(n, size), dtype=np.uint8)
    halves = {"Q8_0": (0,), "Q4_0": (0,), "Q5_0": (0,), "Q4_1": (0, 2), "Q5_1": (0, 2), "Q4_K": (0, 2),
              "Q5_K": (0, 2), "Q6_K": (208,)}[name]  # fmt: skip
    for off in halves:  # finite scales: random bytes may encode inf or NaN
        raw[:, off:off + 2] = np.float16(rng.uniform(-0.1, 0.1, n)).view(np.uint8).reshape(-1, 2)
    return raw


@pytest.mark.parametrize("name", ["Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q4_K", "Q5_K", "Q6_K"])
def test_dequantisation_is_bit_identical_to_the_reference(name):
    rng = np.random.default_rng(hash(name) % 2**32)
    raw = _random_blocks(name, 23, rng)
    qtype = gguf.GGMLQuantizationType[name]
    block, _ = gguf.GGML_QUANT_SIZES[qtype]
    expected = np.asarray(gguf.quants.dequantize(raw, qtype), dtype=np.float32).reshape(-1)
    np.testing.assert_array_equal(dequantize(raw.reshape(-1), int(qtype), 23 * block), expected)


def test_float_formats_and_q8_0_quantisation_match_the_reference():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32 * 16).astype(np.float32)
    np.testing.assert_array_equal(dequantize(x.astype("<f2").view(np.uint8), 1, x.size),
                                  x.astype(np.float16).astype(np.float32))  # fmt: skip
    bf16 = (x.view(np.uint32) >> 16).astype(np.uint16)
    np.testing.assert_array_equal(dequantize(bf16.view(np.uint8), 30, x.size), (bf16.astype(np.uint32) << 16).view(np.float32))
    ours = quantize_q8_0(x)
    np.testing.assert_array_equal(ours, np.asarray(gguf.quants.quantize(x.reshape(16, 32), gguf.GGMLQuantizationType.Q8_0)).reshape(-1))
    with pytest.raises(UnsupportedFormat, match="Q2_K"):
        dequantize(np.zeros(84, np.uint8), 10, 256)  # formats without a dequantiser fail loudly
    with pytest.raises(ValueError, match="expected"):
        dequantize(x.view(np.uint8), 0, x.size + 1)


def test_written_files_read_back_through_the_reference_reader(models, tmp_path):
    path = models[(True, "Q8_0")]
    reader = gguf.GGUFReader(path)
    ours = GGUFTensors(path)
    for t in reader.tensors[:12]:
        expected = np.asarray(gguf.quants.dequantize(np.asarray(t.data), t.tensor_type), dtype=np.float32)
        np.testing.assert_array_equal(ours.load(t.name).reshape(-1), expected.reshape(-1))
    exps = ours.load("blk.0.ffn_up_exps.weight")
    np.testing.assert_array_equal(ours.load_slice("blk.0.ffn_up_exps.weight", 5), exps[5])
    with pytest.raises(IndexError):
        ours.load_slice("blk.0.ffn_up_exps.weight", 8)


# -- the runtime ---------------------------------------------------------------------------------
@pytest.mark.parametrize("moe", [False, True])
def test_incremental_decoding_through_the_kv_cache_equals_full_prefill(models, moe):
    engine = Engine(models[(moe, "F32")])
    tokens = [5, 17, 3, 88, 41, 9, 60, 2]
    with engine.session(16) as s:
        full = s.prefill(tokens, all_logits=True)
    with engine.session(16) as s:
        first = s.prefill(tokens[:3], all_logits=True)
        rows = [first.logits]
        experts = [first.experts]
        for t in tokens[3:]:
            out = s.step(t)
            rows.append(out.logits)
            experts.append(out.experts)
    np.testing.assert_allclose(np.concatenate(rows), full.logits, rtol=0, atol=2e-5 * np.abs(full.logits).max())
    if moe:
        for layer in range(engine.config.n_layers):
            np.testing.assert_array_equal(np.concatenate([e[layer] for e in experts]), full.experts[layer])


def test_generation_is_deterministic_and_bounded_by_the_session(models):
    engine = Engine(models[(False, "Q8_0")])
    with engine.session(12) as s:
        a = generate(s, [1, 2, 3], 6)
    with engine.session(12) as s:
        b = generate(s, [1, 2, 3], 6)
    assert a.tokens == b.tokens and len(a.tokens) == 6
    with engine.session(12) as s:
        stopped = generate(s, [1, 2, 3], 6, stop={a.tokens[1]})
    assert stopped.tokens == a.tokens[:a.tokens.index(a.tokens[1]) + 1] and stopped.stopped == "stop_token"
    with engine.session(4) as s, pytest.raises(ContextOverflow):
        generate(s, [1, 2, 3], 6)
    with pytest.raises(ValueError, match="vocabulary"):
        engine.session(4).prefill([10_000])
    assert engine.telemetry.decode_tokens > 0 and engine.report()["plan"]["assignment"]["output"] == "numpy-cpu"


def test_memory_is_charged_to_the_host_domain_and_sessions_are_admitted(models):
    engine = Engine(models[(False, "F32")], memory_bytes=64 << 20, reserved_fraction=0.0)
    weights = engine.domain.used_by(Category.WEIGHTS)
    assert weights == pytest.approx(sum(a.nbytes for a in engine.weights._dense.values()), rel=0.01)
    s = engine.session(64)
    kv = engine.domain.used_by(Category.KV)
    assert kv >= s.kv.nbytes
    filler = engine.domain.allocate(engine.domain.free - (100 << 10), Category.ACTIVATIONS, "other work", refs=1)
    with pytest.raises(MemoryError, match="session refused"):
        engine.session(256)  # 256 KiB of KV cache; pinned weights cannot be evicted to make room
    engine.domain.release(filler, free_when_unused=True)
    s.close()
    assert engine.domain.used_by(Category.KV) == 0
    engine.domain.check_invariant()
    with pytest.raises(Exception, match="cannot allocate|limit"):
        Engine(models[(False, "F32")], memory_bytes=64 << 10)  # the weights alone do not fit


def test_a_bounded_expert_cache_changes_traffic_not_results(models):
    unbounded = Engine(models[(True, "Q8_0")])
    one_expert = 3 * 32 * 64 * 4
    small = Engine(models[(True, "Q8_0")], expert_cache_bytes=3 * one_expert)
    tokens = list(range(1, 30))
    with unbounded.session(32) as a, small.session(32) as b:
        np.testing.assert_array_equal(a.prefill(tokens[:3]).logits, b.prefill(tokens[:3]).logits)
        for t in tokens[3:]:  # one token at a time: experts are requested again and again
            np.testing.assert_array_equal(a.step(t).logits, b.step(t).logits)
    assert small.telemetry.expert_loads > unbounded.telemetry.expert_loads
    assert unbounded.telemetry.expert_loads <= 2 * 8 and unbounded.telemetry.expert_hits > 0
    assert len(small.weights._experts) <= 3
    small.domain.check_invariant()


# -- plans and backends that do not exist --------------------------------------------------------
@pytest.mark.parametrize("backend", [CUDABackend, MetalBackend, GGMLBackend, RemoteBackend])
def test_unimplemented_backends_refuse_and_never_compute(backend):
    b = backend()
    caps = b.capabilities()
    assert not caps.available and "not implemented" in caps.reason and not caps.ops
    for call in (lambda: b.allocate(16), lambda: b.submit(Op("matmul")), b.synchronize):
        with pytest.raises(BackendUnavailable):
            call()


def test_plans_fail_before_running_when_no_backend_supports_an_operation(models):
    config = Engine(models[(True, "F32")]).config
    devices = DeviceManager.discover()
    plan = make_plan(config, devices)
    assert set(plan.assignment.values()) == {"numpy-cpu"}
    assert {"cuda", "metal", "ggml", "remote"} <= set(plan.rejected) <= {"cuda", "metal", "ggml", "remote", "torch"}
    with pytest.raises(UnsupportedOperation, match="experts.*cuda: not implemented"):
        make_plan(config, devices, overrides={"experts": "cuda"})
    cpu_only = DeviceManager([b for b in devices.backends.values() if b.capabilities().kind != "cpu"])
    with pytest.raises(UnsupportedOperation):
        make_plan(config, cpu_only)


def test_models_the_runtime_does_not_implement_are_refused(tmp_path, models):
    base = read_gguf(models[(False, "F32")])
    meta = {k: v for k, v in base.metadata.items()}
    tensors = [(t.name, GGUFTensors(models[(False, "F32")]).load(t.name), "F32") for t in base.tensors]
    extra = tmp_path / "bias.gguf"
    write_gguf(extra, meta, [*tensors, ("blk.0.attn_q.bias", np.zeros(128, np.float32), "F32")])
    with pytest.raises(UnsupportedModel, match="does not implement"):
        Engine(extra)
    llama = tmp_path / "llama.gguf"
    write_gguf(llama, {**{k.replace("qwen3.", "llama."): v for k, v in meta.items()}, "general.architecture": "llama"},
               tensors)  # fmt: skip
    with pytest.raises(UnsupportedModel, match="architecture 'llama'"):
        Engine(llama)
    missing = tmp_path / "missing.gguf"
    write_gguf(missing, meta, [t for t in tensors if t[0] != "blk.1.attn_k_norm.weight"])
    with pytest.raises(UnsupportedModel, match="missing tensors"):
        Engine(missing)
