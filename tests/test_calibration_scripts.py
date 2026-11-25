"""The calibration scripts: pure logic, CPU measurements, and GPU code paths against a fake.

Nothing here touches a real GPU, PyTorch, transformers or llama.cpp: those paths cannot run in
the development container.  The llama-bench fixture reflects the *assumed* ``-o json`` format.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from basp.calibration import MeasurementSet
from basp.core.topology import Topology
from basp.workloads.trace import RoutingTrace
from benchmarks import profile_compression, profile_hardware, record_routing_trace, run_llama_bench

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# -- run_llama_bench -------------------------------------------------------------------------
def bench_record(**overrides) -> dict:
    """One result object with the field names llama-bench -o json is believed to emit."""
    record = {
        "build_commit": "abc1234", "build_number": 4242, "cpu_info": "11th Gen Intel i9-11900KF",
        "gpu_info": "NVIDIA GeForce RTX 3080 Ti", "backends": "CUDA",
        "model_filename": "Qwen3-30B-A3B-Q4_K_M.gguf", "model_type": "qwen3moe 30B.A3B Q4_K - Medium",
        "model_size": 18_556_000_000, "model_n_params": 30_532_000_000, "n_batch": 2048,
        "n_ubatch": 512, "n_threads": 8, "n_gpu_layers": 16, "n_prompt": 512, "n_gen": 0,
        "avg_ns": 2_043_000_000, "stddev_ns": 21_000_000, "avg_ts": 250.6, "stddev_ts": 2.6,
        "samples_ns": [2_043_000_000], "samples_ts": [250.6],
    }  # fmt: skip
    record.update(overrides)
    return record


FIXTURE = json.dumps([bench_record(), bench_record(n_prompt=0, n_gen=128, avg_ts=21.7, stddev_ts=0.4)])
PARSE = {"machine": "i9-3080ti", "split": "calibration", "model": "qwen3-30b-a3b",
         "bits_per_weight": 4.85}  # fmt: skip


def test_parse_llama_bench_separates_prefill_and_decode():
    prefill, decode = run_llama_bench.parse_llama_bench(FIXTURE, **PARSE)
    assert (prefill.kind, prefill.tokens_per_s, prefill.batch) == ("prefill", 250.6, 512)
    assert (decode.kind, decode.tokens_per_s, decode.batch) == ("decode", 21.7, 1)
    assert prefill.n_gpu_layers == decode.n_gpu_layers == 16 and decode.threads == 8
    assert decode.bits_per_weight == 4.85 and decode.stddev_tokens_per_s == 0.4
    assert decode.source == "llama-bench abc1234" and "n_gen=128" in decode.notes
    assert prefill.id != decode.id and decode.id.startswith("i9-3080ti/qwen3-30b-a3b/b4.85/ngl16/cmoe0/t8/d0/")
    assert prefill.signature != decode.signature
    # log lines around the JSON, and a single object instead of an array, are tolerated
    wrapped = "llama_model_load: loading\n" + FIXTURE + "\nllama_perf: done\n"
    assert run_llama_bench.parse_llama_bench(wrapped, **PARSE) == [prefill, decode]
    assert run_llama_bench.parse_llama_bench(json.dumps(bench_record()), **PARSE) == [prefill]
    small_batch = run_llama_bench.parse_llama_bench(json.dumps([bench_record(n_batch=256)]), **PARSE)
    assert small_batch[0].batch == 256


def test_parse_llama_bench_fails_loudly_when_the_format_differs():
    renamed = bench_record()
    renamed["tokens_per_second"] = renamed.pop("avg_ts")
    with pytest.raises(ValueError, match=r"lacks the expected key\(s\) \['avg_ts'\].*Keys received"):
        run_llama_bench.parse_llama_bench(json.dumps([renamed]), **PARSE)
    with pytest.raises(ValueError, match="no JSON array"):
        run_llama_bench.parse_llama_bench("error: failed to load model", **PARSE)
    with pytest.raises(ValueError, match="n_prompt == n_gen == 0"):
        run_llama_bench.parse_llama_bench(json.dumps([bench_record(n_prompt=0)]), **PARSE)
    skipped: list[str] = []
    mixed = json.dumps([bench_record(n_gen=128), bench_record()])
    assert len(run_llama_bench.parse_llama_bench(mixed, skipped=skipped, **PARSE)) == 1
    assert len(skipped) == 1 and "mixed prompt+generation" in skipped[0]
    repeats = run_llama_bench.parse_llama_bench(json.dumps([bench_record(), bench_record()]), **PARSE)
    assert repeats[1].id == repeats[0].id + "#2"


def test_measurements_accumulate_and_respect_the_split(tmp_path):
    target = MeasurementSet()
    first = run_llama_bench.parse_llama_bench(FIXTURE, **PARSE)
    for m in first:
        run_llama_bench.add_unique(target, m)
    again = run_llama_bench.add_unique(target, first[1])  # the same benchmark run a second time
    assert again.id == first[1].id + "@run2" and len(target) == 3
    validation = run_llama_bench.parse_llama_bench(FIXTURE, **{**PARSE, "split": "validation"})
    with pytest.raises(ValueError, match="both splits"):
        run_llama_bench.add_unique(target, validation[0])
    target.save(tmp_path / "m.json")
    assert len(MeasurementSet.load(tmp_path / "m.json").calibration) == 3


def test_llama_bench_cli_builds_the_matrix_and_names_a_missing_binary(capsys, tmp_path):
    base = ["--machine", "i9", "--split", "validation", "--model-key", "a=/models/a.gguf",
            "--model-key", "b=/models/b.gguf", "--ngl", "0", "--ngl", "99",
            "--bits-per-weight", "4.85", "--out", str(tmp_path / "m.json")]  # fmt: skip
    assert run_llama_bench.main([*base, "--threads", "8", "--n-gen", "64", "--dry-run"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == [
        "llama-bench -m /models/a.gguf -ngl 0 -o json -r 3 -t 8 -n 64",
        "llama-bench -m /models/a.gguf -ngl 99 -o json -r 3 -t 8 -n 64",
        "llama-bench -m /models/b.gguf -ngl 0 -o json -r 3 -t 8 -n 64",
        "llama-bench -m /models/b.gguf -ngl 99 -o json -r 3 -t 8 -n 64",
    ]
    with pytest.raises(SystemExit, match="cannot find the llama.cpp benchmark binary"):
        run_llama_bench.main([*base, "--llama-bench", "/nonexistent/llama-bench"])
    with pytest.raises(SystemExit):  # --split has no default
        run_llama_bench.main(["--machine", "i9", "--model-key", "a=/a.gguf", "--ngl", "0",
                              "--bits-per-weight", "4.5", "--out", "x.json"])  # fmt: skip


# -- record_routing_trace ----------------------------------------------------------------------
def test_router_collector_splits_prefill_from_decode(tmp_path):
    rng = np.random.default_rng(5)
    collector = record_routing_trace.RouterCollector(3)
    for layer in range(3):  # prompt pass: 6 tokens, delivered as [batch, tokens, experts]
        collector.record(layer, rng.standard_normal((1, 6, 16)))
    for _ in range(4):  # one pass per generated token
        for layer in range(3):
            collector.record(layer, rng.standard_normal((1, 16)))
    collector.record(0, rng.standard_normal((1, 16)))  # an interrupted pass is ignored
    assert len(collector.complete_passes()) == 5
    prefill, decode = collector.prefill_trace(4), collector.decode_trace(4)
    assert (prefill.n_steps, prefill.batch, prefill.n_layers, prefill.n_experts) == (1, 6, 3, 16)
    assert (decode.n_steps, decode.batch, decode.top_k) == (4, 1, 4)
    stats = record_routing_trace.write_outputs(collector, 4, str(tmp_path / "t"), {"model": "fake"})
    assert RoutingTrace.load(tmp_path / "t.prefill.json").batch == 6
    assert RoutingTrace.load(tmp_path / "t.decode.json").n_steps == 4
    saved = json.loads((tmp_path / "t.stats.json").read_text())
    assert saved["meta"] == {"model": "fake"} and saved["decode"]["n_steps"] == 4
    assert stats["prefill"]["stickiness"] is None and 0.0 <= stats["decode"]["stickiness"] <= 1.0

    with pytest.raises(ValueError, match="out of order"):
        record_routing_trace.RouterCollector(2).record(1, np.zeros((1, 4)))
    prompt_only = record_routing_trace.RouterCollector(1)
    prompt_only.record(0, np.zeros((3, 4)))
    assert prompt_only.decode_trace(2) is None
    with pytest.raises(ValueError, match="no complete forward pass"):
        record_routing_trace.RouterCollector(2).prefill_trace(1)


def test_find_routers_is_generic_and_lists_candidates():
    def linear(width):
        return SimpleNamespace(out_features=width)

    modules = [("model.layers.0.self_attn.q_proj", linear(2048)), ("model.layers.0.mlp.gate", linear(128)),
               ("model.layers.0.mlp.experts.0.gate_proj", linear(768)),
               ("model.layers.1.mlp.gate", linear(128)), ("lm_head", linear(151936))]  # fmt: skip
    model = SimpleNamespace(named_modules=lambda: iter(modules))
    found = record_routing_trace.find_routers(model, 128)
    assert [name for name, _ in found] == ["model.layers.0.mlp.gate", "model.layers.1.mlp.gate"]
    renamed = SimpleNamespace(named_modules=lambda: iter(
        [("blocks.0.moe.router", linear(64)), ("blocks.1.moe.router", linear(64))]))  # fmt: skip
    with pytest.raises(SystemExit, match=r"Candidate modules: blocks\.0\.moe\.router"):
        record_routing_trace.find_routers(renamed, 64)
    assert len(record_routing_trace.find_routers(renamed, 64, "router")) == 2
    with pytest.raises(SystemExit, match="PyTorch and transformers"):
        record_routing_trace.main(["--prompt", "hi", "--out", "unused"])


# -- profile_compression -----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["gaussian", "laplace", "student-t"])
def test_compression_profile_of_synthetic_weights(kind):
    weights = profile_compression.synthetic_weights(kind, 64, 2048, seed=1)
    result = profile_compression.profile_tensor(weights, block_kib=16, repeats=1)
    assert result["n_weights"] == 64 * 2048 and result["raw_bits_per_weight"] == 4.5
    assert result["blocks"] == 4  # 64 KiB of packed codes in 16 KiB blocks
    assert 1.0 < result["order0_entropy_bits_per_code"] < 4.0
    assert result["order0_bound_bits_per_weight"] == pytest.approx(
        result["order0_entropy_bits_per_code"] + 0.5)  # fmt: skip
    for stats in result["codecs"].values():  # measure_codec also verifies the round trip
        assert math.isfinite(stats["bits_per_weight"]) and 0.5 < stats["bits_per_weight"] < 5.0
        assert stats["decode_compressed_bytes_per_s"] > 0
    with pytest.raises(ValueError, match="unknown synthetic"):
        profile_compression.synthetic_weights("cauchy", 2, 2)


def test_compression_blocks_decode_independently_and_cli_writes_json(tmp_path, capsys):
    data = bytes(range(256)) * 40
    blocks = profile_compression.split_blocks(data, 4096)
    assert len(blocks) == 3 and b"".join(blocks) == data
    stats = profile_compression.measure_codec("zlib", blocks, repeats=1)
    assert stats["ratio"] < 0.1 and stats["decode_output_bytes_per_s"] > 0
    with pytest.raises(ValueError):
        profile_compression.split_blocks(data, 0)

    np.savez(tmp_path / "w.npz", proj=np.random.default_rng(0).standard_normal((32, 256)),
             bias=np.zeros(32), ids=np.arange(64).reshape(8, 8))  # fmt: skip
    loaded = profile_compression.load_weights(tmp_path / "w.npz")
    assert list(loaded) == ["proj"]  # vectors and integer arrays are skipped
    out = tmp_path / "c.json"
    assert profile_compression.main([str(tmp_path / "w.npz"), "--synthetic", "gaussian", "--quick",
                                     "--out", str(out)]) == 0  # fmt: skip
    payload = json.loads(out.read_text())
    assert set(payload["tensors"]) == {"synthetic.gaussian", "w.npz:proj"}
    assert payload["summary"]["n_weights"] == 256 * 4096 + 32 * 256
    printed = capsys.readouterr().out
    assert "order-0 entropy bound" in printed and "say nothing about an FPGA" in printed
    with pytest.raises(SystemExit, match="unsupported weight file"):
        profile_compression.load_weights(tmp_path / "w.bin")


# -- profile_hardware ------------------------------------------------------------------------
def test_cpu_measurements_are_positive_and_finite():
    mib = 1024 * 1024
    copy = profile_hardware.measure_copy(4 * mib, repeats=2)
    assert copy["traffic_bytes_per_s"] == 2 * copy["payload_bytes_per_s"] > 0
    assert profile_hardware.measure_sum(4 * mib, repeats=2)["read_bytes_per_s"] > 0
    parallel = profile_hardware.measure_parallel("copy", 4 * mib, 2, duration=0.05)
    assert parallel["processes"] == 2 and len(parallel["per_process_bytes_per_s"]) == 2
    assert math.isfinite(parallel["payload_bytes_per_s"]) and parallel["payload_bytes_per_s"] > 0
    assert profile_hardware.measure_parallel("read", 4 * mib, 2, duration=0.05)["read_bytes_per_s"] > 0
    kernel = profile_hardware.measure_int4_kernel(32, 256, repeats=1)
    assert kernel["macs_per_s"] > 0 and "not this" in kernel["note"]
    with pytest.raises(ValueError):
        profile_hardware.measure_parallel("write", mib, 1)


class FakeTensor:
    def __init__(self, torch, nbytes, device="cpu", shape=None):
        self.torch, self.nbytes, self.device, self.shape = torch, nbytes, device, shape

    def pin_memory(self):
        return self

    def to(self, device):
        self.torch.clock.now += self.torch.h2d[0] + self.nbytes / self.torch.h2d[1]
        return FakeTensor(self.torch, self.nbytes, device, self.shape)

    def cpu(self):
        self.torch.clock.now += self.torch.d2h[0] + self.nbytes / self.torch.d2h[1]
        return FakeTensor(self.torch, self.nbytes, "cpu", self.shape)

    def __add__(self, other):
        self.torch.clock.now += self.torch.launch
        return self

    def __matmul__(self, other):
        rows, cols = self.shape
        cost = max(rows * cols * 2 / self.torch.mem_bandwidth, rows * cols * other.shape[1] / self.torch.rate)
        self.torch.clock.now += self.torch.launch + cost
        return FakeTensor(self.torch, rows * other.shape[1] * 2, self.device, (rows, other.shape[1]))


class FakeClock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeTorch:
    """The handful of PyTorch calls the profiler makes, with a modelled clock."""

    uint8, float16 = "uint8", "float16"
    h2d, d2h = (8e-6, 22e9), (12e-6, 14e9)  # (latency, bandwidth)
    launch, mem_bandwidth, rate = 9e-6, 800e9, 12e12

    def __init__(self, device: str | None = "cuda") -> None:
        self.clock = FakeClock()
        self.cuda = SimpleNamespace(
            is_available=lambda: device == "cuda", synchronize=lambda: None,
            get_device_name=lambda _i: "Fake RTX", get_device_properties=lambda _i: SimpleNamespace(
                total_memory=12 * 1024**3))  # fmt: skip
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: device == "mps"))
        self.mps = SimpleNamespace(synchronize=lambda: None, recommended_max_memory=lambda: 16 * 1024**3)

    def empty(self, n, dtype=None):
        return FakeTensor(self, n)

    def zeros(self, n, device=None):
        return FakeTensor(self, n, device)

    def randn(self, shape, dtype=None, device=None):
        return FakeTensor(self, shape[0] * shape[1] * 2, device, shape)


def test_gpu_profiling_recovers_the_parameters_of_a_fake_device():
    torch = FakeTorch("cuda")
    gpu = profile_hardware.profile_gpu(torch, quick=True, clock=torch.clock)
    assert gpu["device"] == "cuda" and gpu["name"] == "Fake RTX" and gpu["memory_bytes"] == 12 * 1024**3
    assert gpu["h2d_fit"]["bandwidth_bytes_per_s"] == pytest.approx(22e9, rel=1e-6)
    assert gpu["h2d_fit"]["latency_s"] == pytest.approx(8e-6, rel=1e-6)
    assert gpu["d2h_fit"]["bandwidth_bytes_per_s"] == pytest.approx(14e9, rel=1e-6)
    assert gpu["launch_overhead_s"] == pytest.approx(9e-6)
    fit = gpu["roofline_fit"]
    assert fit["mem_bandwidth"] == pytest.approx(800e9, rel=0.01)
    assert fit["compute_rate"] == pytest.approx(12e12, rel=0.01)
    assert fit["startup_s"] == pytest.approx(9e-6, rel=0.05)
    assert profile_hardware.profile_gpu(FakeTorch(None), quick=True) is None
    assert profile_hardware.pick_device(FakeTorch("mps")) == "mps"
    info = profile_hardware.device_info(FakeTorch("mps"), "mps")
    assert "unified memory" in info["name"] and info["memory_bytes"] == 16 * 1024**3


def test_fitted_profile_marks_measured_fields_and_stays_loadable():
    torch = FakeTorch("cuda")
    measurements = {
        "machine": {"node": "test-box"},
        "cpu": {"sustained_bytes_per_s": 38e9},
        "gpu": profile_hardware.profile_gpu(torch, quick=True, clock=torch.clock),
    }
    template = profile_hardware.load_template("desktop_gpu12")
    profile = profile_hardware.fitted_profile(template, measurements, host_device="host",
                                              device="gpu", link="pcie-gpu")  # fmt: skip
    topology = Topology.from_dict(profile)
    assert topology.by_name("host").mem_bandwidth == 38e9
    assert topology.by_name("gpu").kernel_startup_s == pytest.approx(9e-6)
    link = topology.links[0]  # a = host, b = gpu: host->device is the forward direction
    assert link.bandwidth_ab == pytest.approx(22e9, rel=1e-6)
    assert link.bandwidth_ba == pytest.approx(14e9, rel=1e-6)
    assert link.latency_s == pytest.approx(10e-6, rel=1e-6)
    assert profile["name"] == "desktop_gpu12_measured" and profile["_note"].startswith("PARTLY MEASURED")
    assert profile["_calibration"]["machine"] == {"node": "test-box"}
    assert "devices.host.compute_rate" not in profile["_calibration"]["measured"]  # still assumed
    assert json.loads((CONFIGS / "desktop_gpu12.json").read_text()) == template  # template untouched
    cpu_only = profile_hardware.fitted_profile(template, {**measurements, "gpu": None},
                                               host_device="host", device="gpu", link="pcie-gpu")  # fmt: skip
    assert cpu_only["_calibration"]["measured"] == ["devices.host.mem_bandwidth"]
    with pytest.raises(SystemExit, match="not found"):
        profile_hardware.load_template("no_such_profile")
