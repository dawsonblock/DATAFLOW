"""v0.3: evidence levels, provenance, the candidate-specific gate, regressions for every defect
found in the v0.2 audit, unified memory, GGUF headers, and the calibration runner end to end."""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from dataclasses import replace

import numpy as np
import pytest

from basp.calibration import Measurement, MeasurementSet
from basp.core.topology import Topology
from basp.dataflow.base import DataflowMode, ExecutionPolicy
from basp.dse.hardware import DesignError, HardwareDesign
from basp.evidence import (
    CONFIG_DIR,
    EvidenceLevel,
    provenance,
    sha256_json,
    source_fingerprint,
    stale_reasons,
)
from basp.experiments.calibrate import (
    calibration_status,
    fit_profile,
    load_measurements,
    predict,
    score,
)
from basp.experiments.profiles import load_json, load_topology
from basp.experiments.variants import run_architecture, traces
from basp.memory.allocator import OutOfMemory
from basp.models import (
    GGML_TYPES,
    GGUFError,
    check_against,
    read_gguf,
    spec_from_gguf,
    write_minimal_gguf,
)
from basp.workloads.trace import RoutingStep, RoutingTrace
from basp.workloads.zoo import MODELS

ESTIMATES = load_json("design_estimates")
BASE = load_topology("desktop_gpu12")
THRESHOLDS = load_json("hardware_gate")
MACHINE = "i9-11900kf-3080ti"


# -- evidence and provenance ------------------------------------------------------------------
def test_evidence_levels_are_ordered_and_described():
    assert EvidenceLevel.E0 < EvidenceLevel.E1 < EvidenceLevel.E2 < EvidenceLevel.E3
    assert str(EvidenceLevel.E2) == "E2 (simulation validated on existing hardware)"
    assert EvidenceLevel.E3.description == "measurement"


def test_provenance_detects_changed_configs_inputs_and_parameters(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text("{}")
    first = provenance(configs=["desktop_gpu12"], inputs={"trace": trace, "absent": None},
                       parameters={"budget": 10})  # fmt: skip
    assert first["source"] == source_fingerprint() and len(first["source"]) == 64
    assert first["inputs"]["absent"] is None and first["environment"]["basp"]
    assert stale_reasons(first, provenance(configs=["desktop_gpu12"], inputs={"trace": trace, "absent": None},
                                           parameters={"budget": 10})) == []  # fmt: skip
    trace.write_text('{"changed": true}')
    later = provenance(configs=["desktop_gpu12"], inputs={"trace": trace, "absent": None}, parameters={"budget": 11})
    assert stale_reasons(first, later) == ["input 'trace' changed", "run parameters changed"]
    other = provenance(configs=["design_estimates"], inputs={"trace": trace, "absent": None}, parameters={"budget": 11})
    assert "config 'desktop_gpu12' changed" in stale_reasons(later, other)
    assert stale_reasons(None, later) == ["no provenance recorded"]
    assert sha256_json({"a": 1, "b": [1.0, 2]}) == sha256_json({"b": [1.0, 2], "a": 1})


# -- audit defect A: calibration rows carry their machine; the report can use them -------------------
def test_scored_rows_name_their_machine_and_the_report_finds_validated_profiles(tmp_path):
    from basp.experiments.report_v5 import validated_profiles

    ms = [Measurement(id=f"v{i}", machine=MACHINE, split="validation", kind="decode", model="qwen3-8b",
                      tokens_per_s=10.0, n_gpu_layers=4 * i) for i in range(3)]  # fmt: skip
    report = score(MeasurementSet(ms))
    assert {r["machine"] for r in report.per_measurement} == {MACHINE}
    # three benchmarks are too few, and no calibration record binds the profile: not validated
    assert report.status == "not_evaluated" and validated_profiles(report, tmp_path) == set()
    assert validated_profiles(calibration_status(None)) == set()


def test_measurements_are_merged_from_per_machine_evidence_folders(tmp_path):
    for machine, ngl in ((MACHINE, 0), ("macbook-m2", 99)):
        folder = tmp_path / machine
        folder.mkdir()
        MeasurementSet([Measurement(id=f"{machine}-1", machine=machine, split="validation", kind="decode",
                                    model="qwen3-8b", tokens_per_s=10.0, n_gpu_layers=ngl)]).save(
            folder / "measurements.json")  # fmt: skip
    merged = load_measurements(tmp_path)
    assert sorted(m.machine for m in merged) == ["i9-11900kf-3080ti", "macbook-m2"]
    assert len(load_measurements(tmp_path / MACHINE / "measurements.json")) == 1
    assert len(load_measurements(tmp_path / "nothing-here")) == 0


# -- audit defect B (a candidate is qualified only by its own evidence): tests/test_v04_evidence.py


# -- audit defect C: the local fabric uses each module's own link bandwidth ----------------------------
def test_module_link_bandwidth_shapes_the_local_fabric_and_the_price():
    slow, fast = (HardwareDesign(modules=4, fabric="local", link_gbps=g) for g in (8.0, 32.0))
    t_slow, t_fast = slow.topology(BASE, ESTIMATES), fast.topology(BASE, ESTIMATES)
    gpu, mod = t_slow.by_name("gpu").device_id, t_slow.by_name("mod0").device_id
    assert t_slow.transfer_rate(gpu, mod) == pytest.approx(8e9)
    assert t_fast.transfer_rate(gpu, mod) == pytest.approx(32e9)
    capped = HardwareDesign(modules=4, fabric="local", link_gbps=200.0).topology(BASE, ESTIMATES)
    port = next(k for k in capped.links if k.name == "fabric-mod0")
    assert port.bandwidth_ab == pytest.approx(64e9)  # no faster than a fabric port
    assert fast.cost_usd(ESTIMATES) > slow.cost_usd(ESTIMATES)
    spec = MODELS["qwen3-30b-a3b"]  # large enough that expert fills to the GPU cross the module port
    prior, held = traces(spec, tokens=2, prior_steps=8)
    speeds = [run_architecture("H", d, spec, prior, held, tokens=2, warmup_tokens=1,
                               replicas=1).tokens_per_s for d in (slow, fast)]  # fmt: skip
    assert speeds[0] != speeds[1]  # the parameter the price charges for now changes the hardware


# -- audit defect D: routing coefficients must be finite numbers -----------------------------------------
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.5"])
def test_routing_traces_reject_non_finite_coefficients(bad):
    step = RoutingStep(((0, 1),), ((bad, 0.5),))
    with pytest.raises(ValueError, match="finite numbers"):
        RoutingTrace(1, 4, 2, 1, [[step]]).validate()
    RoutingTrace(1, 4, 2, 1, [[RoutingStep(((0, 1),), ((0.5, 0.5),))]]).validate()


# -- audit defect F: GPU classes differ in performance, not only capacity ----------------------------------
def test_gpu_classes_scale_bandwidth_and_compute():
    rates = {}
    for gib in (12, 16, 24, 32, 48):
        gpu = HardwareDesign(gpu_gib=gib).topology(BASE, ESTIMATES).by_name("gpu")
        rates[gib] = (gpu.mem_bandwidth, gpu.compute_rate, gpu.mem_capacity >> 30)
    assert rates[12] == (500e9, 20e12, 12)  # the base class is the base profile's GPU, unchanged
    assert rates[32][0] > 1.8 * rates[12][0] and rates[16][0] < rates[12][0]
    assert rates[24][1] > rates[12][1]
    # pessimistic for the module proposal = the strongest conventional GPU
    p, o = (HardwareDesign(gpu_gib=24).topology(BASE, ESTIMATES, c).by_name("gpu") for c in ("pessimistic", "optimistic"))
    assert p.mem_bandwidth > o.mem_bandwidth and p.compute_rate > o.compute_rate
    with pytest.raises(DesignError, match="GPU class"):
        HardwareDesign(gpu_gib=20).topology(BASE, ESTIMATES)


def test_a_bigger_gpu_class_makes_the_conventional_system_faster():
    spec = MODELS["qwen3-30b-a3b"]
    prior, held = traces(spec, tokens=2, prior_steps=8)
    speed = {g: run_architecture("A", HardwareDesign(gpu_gib=g), spec, prior, held, tokens=2, warmup_tokens=1,
                                 replicas=1).tokens_per_s for g in (12, 32)}  # fmt: skip
    assert speed[32] > 1.5 * speed[12]


# -- unified memory ------------------------------------------------------------------------------------
def unified(shared: bool) -> Topology:
    data = load_topology("macbook_m2").to_dict()
    if not shared:
        for dev in data["devices"]:
            dev["memory_domain"] = ""
    return Topology.from_dict(data)


def test_unified_memory_shares_one_bus_and_one_capacity_pool():
    from basp.backends.virtual import VirtualSystem
    from basp.memory.residency import ResidencyMap
    from basp.memory.tensor_registry import TensorRegistry

    shared, split = unified(True), unified(False)
    assert shared.memory_owner(1) == 0 and split.memory_owner(1) == 1
    registry = TensorRegistry()
    system = VirtualSystem(shared, registry, ResidencyMap(registry, shared))
    assert system.res[("mem", 0)] is system.res[("mem", 1)]
    residency = ResidencyMap(registry, shared)
    before = residency.free_bytes(1)
    from basp.core.tensor import DType, TensorKind

    big = registry.add("w", TensorKind.WEIGHT, (1,), DType.BYTES, 10 << 30)
    residency.place(big.tensor_id, 0)  # 10 GiB on the CPU side ...
    assert residency.free_bytes(1) == before - (10 << 30)  # ... leaves less for the GPU side
    other = registry.add("w2", TensorKind.WEIGHT, (1,), DType.BYTES, 8 << 30)
    with pytest.raises(OutOfMemory):
        residency.place(other.tensor_id, 1)  # 18 GiB does not fit 16 GiB, whichever device asks
    residency.check_invariants()
    bad = shared.to_dict()
    bad["devices"][1]["mem_bandwidth"] = 1e9
    with pytest.raises(ValueError, match="share memory domain"):
        Topology.from_dict(bad)


def test_unified_memory_traffic_from_one_device_slows_the_other():
    from basp.core.graph import OpCode
    from basp.experiments.microbench import MicroSystem

    def run(topology):
        ms = MicroSystem(topology, ExecutionPolicy("t", frozenset({DataflowMode.WEIGHT_STATIONARY})))
        ops = []
        for device in (0, 1):
            w = ms.weight(f"w{device}", 64 << 20, device)
            x, y = ms.activation(f"x{device}", 4096, device), ms.activation(f"y{device}", 4096)
            ops.append(ms.op(OpCode.EXPERT, f"e{device}", inputs=(x,), weights=(w,), outputs=(y,),
                             macs=64 << 20, output_device=device))  # fmt: skip
        return ms.run(ops).actual_s

    alone, together = run(unified(False)), run(unified(True))
    assert together > 1.8 * alone  # two memory-bound kernels on one bus take about twice as long


def test_m2_predictions_use_the_whole_unified_memory():
    m = Measurement(id="x", machine="macbook-m2", split="validation", kind="decode", model="qwen3-14b",
                    tokens_per_s=1.0, bits_per_weight=4.85, n_gpu_layers=0)  # fmt: skip
    assert predict(m) is not None  # 9 GB on the CPU side fits the 16 GiB pool
    assert predict(replace(m, model="qwen3-30b-a3b")) is None  # 18 GB does not


# -- GGUF headers -----------------------------------------------------------------------------------------
Q4_K, Q6_K, F32 = 12, 14, 0


def qwen3_tensors(spec):
    """Tensor index of a Q4_K_M-style GGUF of ``spec`` (llama.cpp naming)."""
    h, hd, nh, nkv, m, e = spec.hidden, spec.head_dim, spec.n_heads, spec.n_kv_heads, spec.expert_inter, spec.n_experts
    out = [("token_embd.weight", (h, spec.vocab), Q4_K), ("output_norm.weight", (h,), F32)]
    if not spec.tied_embeddings:
        out.append(("output.weight", (h, spec.vocab), Q6_K))
    for i in range(spec.n_layers):
        p = f"blk.{i}."
        out += [(p + "attn_norm.weight", (h,), F32), (p + "attn_q.weight", (h, nh * hd), Q4_K),
                (p + "attn_k.weight", (h, nkv * hd), Q4_K), (p + "attn_v.weight", (h, nkv * hd), Q6_K if i % 2 else Q4_K),
                (p + "attn_output.weight", (nh * hd, h), Q4_K), (p + "attn_q_norm.weight", (hd,), F32),
                (p + "attn_k_norm.weight", (hd,), F32), (p + "ffn_norm.weight", (h,), F32)]  # fmt: skip
        if e > 1:
            out += [(p + "ffn_gate_inp.weight", (h, e), F32), (p + "ffn_gate_exps.weight", (h, m, e), Q4_K),
                    (p + "ffn_up_exps.weight", (h, m, e), Q4_K),
                    (p + "ffn_down_exps.weight", (m, h, e), Q6_K if i % 2 else Q4_K)]  # fmt: skip
        else:
            out += [(p + "ffn_gate.weight", (h, m), Q4_K), (p + "ffn_up.weight", (h, m), Q4_K),
                    (p + "ffn_down.weight", (m, h), Q6_K if i % 2 else Q4_K)]  # fmt: skip
    return out


def qwen3_metadata(spec):
    arch = "qwen3moe" if spec.n_experts > 1 else "qwen3"
    meta = {"general.architecture": arch, "general.name": spec.name, "general.file_type": 15,
            f"{arch}.block_count": spec.n_layers, f"{arch}.embedding_length": spec.hidden,
            f"{arch}.attention.head_count": spec.n_heads, f"{arch}.attention.head_count_kv": spec.n_kv_heads,
            f"{arch}.attention.key_length": spec.head_dim}  # fmt: skip
    if spec.n_experts > 1:
        meta |= {f"{arch}.expert_count": spec.n_experts, f"{arch}.expert_used_count": spec.top_k,
                 f"{arch}.expert_feed_forward_length": spec.expert_inter, f"{arch}.feed_forward_length": 6144}
    else:
        meta[f"{arch}.feed_forward_length"] = spec.expert_inter
    return meta


def write_model(path, key):
    spec = MODELS[key]
    write_minimal_gguf(path, qwen3_metadata(spec), qwen3_tensors(spec), sparse=True)
    return path


@pytest.mark.parametrize("key", ["qwen3-4b", "qwen3-8b", "qwen3-30b-a3b"])
def test_gguf_header_gives_shape_size_and_bits_of_a_real_layout(tmp_path, key):
    path = write_model(tmp_path / f"{key}.gguf", key)
    gguf = read_gguf(path)
    spec = MODELS[key]
    assert gguf.architecture == ("qwen3moe" if spec.n_experts > 1 else "qwen3") and gguf.file_type == "Q4_K_M"
    check = check_against(gguf, spec)
    assert check["ok"], check["problems"]
    assert check["relative_difference"] < 0.002  # the simulator's parameter formulas match the layout
    assert 4.5 < gguf.bits_per_weight < 5.2  # Q4_K with some Q6_K: what Q4_K_M files report
    assert spec_from_gguf(gguf).n_experts == spec.n_experts
    fp = gguf.fingerprint()
    assert fp["parameters"] == gguf.parameter_count and len(fp["header_sha256"]) == 64
    assert os.path.getsize(path) == gguf.file_size and gguf.data_offset % 32 == 0
    wrong = check_against(gguf, MODELS["qwen3-14b"])
    assert not wrong["ok"] and any(p.startswith("hidden") for p in wrong["problems"])


def test_gguf_block_sizes_match_the_reference_package():
    gguf = pytest.importorskip("gguf")
    for type_id, (name, block, size) in GGML_TYPES.items():
        ref = gguf.GGMLQuantizationType(type_id)
        assert ref.name == name and gguf.GGML_QUANT_SIZES[ref] == (block, size), name


def test_gguf_reader_agrees_with_the_reference_writer(tmp_path):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "ref.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen3")
    writer.add_block_count(2)
    writer.add_embedding_length(256)
    writer.add_head_count(2)
    writer.add_head_count_kv(1)
    writer.add_key_length(128)
    writer.add_feed_forward_length(512)
    writer.add_string("general.name", "tiny")
    writer.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(5000)])
    tensors = {"token_embd.weight": np.zeros((320, 256), np.float32), "blk.0.ffn_up.weight": np.ones((512, 256), np.float16)}
    for name, array in tensors.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    ours = read_gguf(path)
    assert ours.architecture == "qwen3" and ours.metadata["general.name"] == "tiny"
    assert ours.metadata["tokenizer.ggml.tokens"].length == 5000  # long arrays are summarised
    assert {t.name: t.shape for t in ours.tensors} == {"token_embd.weight": (256, 320), "blk.0.ffn_up.weight": (256, 512)}
    assert ours.data_bytes == 320 * 256 * 4 + 512 * 256 * 2
    reference = gguf.GGUFReader(str(path))
    assert ours.data_offset == reference.data_offset


def test_gguf_reader_rejects_damaged_files(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGML" + b"\0" * 32)
    with pytest.raises(GGUFError, match="bad magic"):
        read_gguf(bad)
    good = write_model(tmp_path / "m.gguf", "qwen3-4b")
    data = good.read_bytes()[:4096]
    (tmp_path / "short.gguf").write_bytes(data[:200])
    with pytest.raises(GGUFError):
        read_gguf(tmp_path / "short.gguf")
    truncated = tmp_path / "trunc.gguf"
    with open(good, "rb") as src, open(truncated, "wb") as dst:
        dst.write(src.read(read_gguf(good).data_offset + 1000))
    with pytest.raises(GGUFError, match="truncated"):
        read_gguf(truncated)
    huge = tmp_path / "huge.gguf"
    huge.write_bytes(b"GGUF" + (3).to_bytes(4, "little") + (2**40).to_bytes(8, "little") + (0).to_bytes(8, "little"))
    with pytest.raises(GGUFError, match="implausible"):
        read_gguf(huge)


# -- fitting a profile on the calibration split ---------------------------------------------------------------
def synthetic_measurements(truth_scales, models=("qwen3-4b", "qwen3-8b"), validation_layers=(6, 18, 24, 30)):
    """What a machine whose real parameters differ from the prior measures: generation and prompts."""
    from basp.experiments.calibrate import _scaled

    prior = load_json("i9_11900kf_rtx3080ti")
    machine = {MACHINE: Topology.from_dict(_scaled(prior, truth_scales))}
    out = []
    for model in models:
        for split, layers in (("calibration", (0, 99)), ("validation", validation_layers)):
            for ngl in layers:
                for kind, batch, ctx in (("decode", 1, 64), ("prefill", 512, 0)):
                    m = Measurement(id=f"{model}-{kind}-ngl{ngl}", machine=MACHINE, split=split, kind=kind,
                                    model=model, tokens_per_s=1.0, bits_per_weight=4.85, n_gpu_layers=ngl,
                                    batch=batch, context_len=ctx)  # fmt: skip
                    out.append(replace(m, tokens_per_s=predict(m, machine)))
    return prior, out


TRUTH = {"host.compute_rate": 0.5, "host.mem_bandwidth": 0.8, "gpu.mem_bandwidth": 0.7, "gpu.compute_rate": 1.6}


@pytest.fixture(scope="module")
def fitted():
    prior, ms = synthetic_measurements(TRUTH)
    return prior, ms, fit_profile(ms, MACHINE, prior)


def test_fitting_recovers_every_parameter_when_prompts_are_measured(fitted):
    prior, ms, fit = fitted
    assert score(MeasurementSet(ms)).status == "fail"  # the prior is wrong for this machine
    for name, truth in TRUTH.items():
        assert fit.scales[name] == pytest.approx(truth, rel=0.06), name
    assert fit.median_error_after < 0.01 < fit.median_error_before
    after = score(MeasurementSet(ms), {MACHINE: Topology.from_dict(fit.profile)})
    assert after.status == "pass" and after.validation.n == 16
    assert fit.profile["_calibration"]["split"] == "calibration"
    assert prior["devices"][0]["compute_rate"] == 400e9  # the prior itself is untouched


def test_the_fit_never_sees_validation_measurements(fitted):
    prior, ms, fit = fitted
    corrupted = [replace(m, tokens_per_s=m.tokens_per_s * 7) if m.split == "validation" else m for m in ms]
    again = fit_profile(corrupted, MACHINE, prior)
    assert again.scales == fit.scales and again.used == fit.used
    with pytest.raises(ValueError, match="no calibration-split measurements"):
        fit_profile([m for m in ms if m.split == "validation"], MACHINE, prior)
    with pytest.raises(ValueError, match="cannot fit"):
        fit_profile(ms, MACHINE, prior, parameters=("host.idle_w",))


# -- the calibration runner, end to end, with a fake llama-bench --------------------------------------------------
FAKE_BENCH = '''#!{python}
import json, sys
args = sys.argv[1:]
def opt(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default
model, ngl = opt("-m"), int(opt("-ngl"))
depth, cmoe = int(opt("-d", 0)), int(opt("-ncmoe", 0))
truth = json.load(open({truth!r}))
key = f"{{model.rsplit('/', 1)[-1]}}|{{ngl}}|{{depth}}|{{cmoe}}"
if key not in truth:
    print("llama_model_load: error: out of device memory", file=sys.stderr)
    sys.exit(1)
records = []
for kind in ("prefill", "decode"):
    n_prompt, n_gen = (512, 0) if kind == "prefill" else (0, 128)
    records.append({{"build_commit": "fake123", "model_filename": model, "n_gpu_layers": ngl, "n_cpu_moe": cmoe,
                    "n_threads": 8, "n_batch": 2048, "n_ubatch": 512, "n_prompt": n_prompt, "n_gen": n_gen,
                    "n_depth": depth, "avg_ts": truth[key][kind], "stddev_ts": 0.1, "flash_attn": 0}})
print(json.dumps(records))
'''


def test_calibration_runner_runs_the_plan_resumes_and_fits(tmp_path, capsys):
    from basp.experiments.calibrate import _scaled
    from benchmarks import run_calibration

    plan = {"version": 9, "defaults": {"repetitions": 1, "n_prompt": 512, "n_gen": 128},
            "models": {"qwen3-4b": {"file": "Qwen3-4B-Q4_K_M.gguf", "repository": "x"},
                       "qwen3-8b": {"file": "Qwen3-8B-Q4_K_M.gguf", "repository": "x"}},
            "machines": {MACHINE: {"profile": "i9_11900kf_rtx3080ti", "runs": [
                {"split": "calibration", "model": "qwen3-4b", "ngl": [0, 99]},
                {"split": "calibration", "model": "qwen3-8b", "ngl": [0, 99]},
                {"split": "validation", "model": "qwen3-8b", "ngl": [6, 12, 18, 24, 30]},
                {"split": "validation", "model": "qwen3-4b", "ngl": [18], "depth": [4096]},
                {"split": "validation", "model": "qwen3-8b", "ngl": [99], "depth": [99999]}]}}}  # fmt: skip
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    models = tmp_path / "models"
    models.mkdir()
    for key, info in plan["models"].items():
        write_model(models / info["file"], key)
    # the "real" machine differs from the prior; its measurements come from the simulator
    truth_machine = {MACHINE: Topology.from_dict(_scaled(load_json("i9_11900kf_rtx3080ti"), TRUTH))}
    truth = {}
    for run in run_calibration.plan_runs(plan, MACHINE):
        if run["depth"] == 99999:
            continue  # this configuration "does not fit": the fake bench fails
        file = plan["models"][run["model"]]["file"]
        bits = read_gguf(models / file).bits_per_weight
        rates = {}
        for kind, batch, ctx in (("prefill", 512, run["depth"]), ("decode", 1, run["depth"] + 64)):
            m = Measurement(id="t", machine=MACHINE, split="validation", kind=kind, model=run["model"],
                            tokens_per_s=1.0, bits_per_weight=round(bits, 4), n_gpu_layers=run["ngl"],
                            batch=batch, context_len=ctx)  # fmt: skip
            rates[kind] = predict(m, truth_machine)
        truth[f"{file}|{run['ngl']}|{run['depth']}|{run['n_cpu_moe']}"] = rates
    (tmp_path / "truth.json").write_text(json.dumps(truth))
    bench = tmp_path / "llama-bench"
    bench.write_text(FAKE_BENCH.format(python=sys.executable, truth=str(tmp_path / "truth.json")))
    bench.chmod(bench.stat().st_mode | stat.S_IEXEC)
    evidence = tmp_path / "evidence"
    argv = ["--machine", MACHINE, "--plan", str(plan_path), "--models-dir", str(models), "--llama-bench", str(bench),
            "--evidence", str(evidence)]
    assert run_calibration.main(argv) == 0
    out = capsys.readouterr().out
    assert "10 of 11 configurations ran; 1 failed" in out and "-> PASS" in out
    measurements = MeasurementSet.load(evidence / "measurements.json")
    assert len(measurements) == 20 and len(measurements.calibration) == 8
    first = next(iter(measurements))
    assert first.provenance["plan_version"] == 9 and first.provenance["model_file"]["file"] == "Qwen3-4B-Q4_K_M.gguf"
    assert first.provenance["llama_bench_build"] == "fake123"
    runs = json.loads((evidence / "runs.json").read_text())
    assert [r["status"] for r in runs].count("failed") == 1
    assert "out of device memory" in next(r for r in runs if r["status"] == "failed")["stderr_tail"]
    calibration = json.loads((evidence / "calibration.json").read_text())
    assert calibration["prior"]["status"] == "fail" and calibration["fitted"]["status"] == "pass"
    assert (evidence / "profile_fitted.json").exists() and (evidence / "environment.json").exists()
    # a second run resumes: nothing is re-measured, the failure is not retried
    assert run_calibration.main(argv) == 0
    assert "running:" not in capsys.readouterr().out
    assert len(MeasurementSet.load(evidence / "measurements.json")) == 20


def test_calibration_runner_refuses_mislabelled_and_missing_models(tmp_path):
    from benchmarks import run_calibration

    plan = json.loads((CONFIG_DIR / "calibration_plan.json").read_text())
    runs = run_calibration.plan_runs(plan, MACHINE)
    assert len(runs) == 18 and sum(r["split"] == "calibration" for r in runs) == 4
    with pytest.raises(SystemExit, match="model files not found"):
        run_calibration.find_models(plan, runs, tmp_path, {}, check_shape=True)
    wrong = write_model(tmp_path / "eight.gguf", "qwen3-8b")
    with pytest.raises(SystemExit, match="does not look like qwen3-14b"):
        run_calibration.find_models(plan, [r for r in runs if r["model"] == "qwen3-14b"], None,
                                    {"qwen3-14b": str(wrong)}, check_shape=True)  # fmt: skip
    with pytest.raises(SystemExit, match="not in the plan"):
        run_calibration.plan_runs(plan, "someone-elses-laptop")


def test_llama_bench_records_depth_cpu_experts_and_physical_batch():
    from benchmarks.run_llama_bench import parse_llama_bench

    record = {"n_gpu_layers": 99, "n_cpu_moe": 24, "n_threads": 8, "n_batch": 2048, "n_ubatch": 512,
              "n_depth": 4096, "avg_ts": 40.0}  # fmt: skip
    pp, tg = parse_llama_bench(json.dumps([{**record, "n_prompt": 512, "n_gen": 0},
                                           {**record, "n_prompt": 0, "n_gen": 128}]),
                               machine=MACHINE, split="validation", model="qwen3-30b-a3b", bits_per_weight=4.9)  # fmt: skip
    assert (pp.batch, pp.context_len, pp.n_cpu_moe) == (512, 4096, 24)
    assert (tg.batch, tg.context_len) == (1, 4096 + 64)  # average KV length over the generated tokens
    assert pp.signature != tg.signature and "cmoe24" in pp.id and "d4096" in pp.id
    assert math.isfinite(predict(tg)) and predict(replace(tg, n_cpu_moe=48)) < predict(tg)


def test_the_numeric_payload_implements_the_backend_contract():
    from basp.backends.base import Payload
    from basp.backends.cpu import NumericPayload

    assert all(callable(getattr(NumericPayload, name)) for name in ("begin_step", "route", "execute", "end_step"))
    assert issubclass(NumericPayload, Payload)


# -- the real model files ---------------------------------------------------------------------------------------
REAL = json.loads((CONFIG_DIR.parent / "tests" / "fixtures" / "qwen3_gguf_headers.json").read_text())["files"]


@pytest.mark.parametrize("key", sorted(REAL))
def test_simulator_shapes_equal_the_real_gguf_files(key):
    """Parameter counts of the real Q4_K_M files on Hugging Face, read from their headers."""
    real, spec = REAL[key], MODELS[key]
    assert spec.total_params == real["parameters"]  # exactly: every tensor of the file is accounted for
    meta = {k.split(".", 1)[1]: v for k, v in real["metadata"].items()}
    assert (meta["embedding_length"], meta["block_count"]) == (spec.hidden, spec.n_layers)
    assert (meta["attention.head_count"], meta["attention.head_count_kv"]) == (spec.n_heads, spec.n_kv_heads)
    assert meta.get("expert_count", 1) == spec.n_experts and meta.get("expert_used_count", 1) == spec.top_k
    # the layout the synthetic-file tests use has the same parameter count as the real file
    assert sum(math.prod(shape) for _, shape, _ in qwen3_tensors(spec)) == real["parameters"]
    assert 4.85 < real["bits_per_weight"] < 4.96 and real["file"] == f"{spec.name}-Q4_K_M.gguf"


def test_plan_names_the_real_files_and_every_configuration_can_be_simulated():
    """A configuration the simulator cannot place would make a machine's validation 'not evaluated'."""
    from benchmarks import run_calibration

    plan = json.loads((CONFIG_DIR / "calibration_plan.json").read_text())
    for key, info in plan["models"].items():
        assert info["file"] == REAL[key]["file"] and info["repository"] == REAL[key]["repository"]
    for machine, section in plan["machines"].items():
        load_topology(section["profile"])
        for run in run_calibration.plan_runs(plan, machine):
            bits = REAL[run["model"]]["bits_per_weight"]
            for kind, batch, ctx in (("prefill", 512, run["depth"]), ("decode", 1, run["depth"] + 64)):
                m = Measurement(id="x", machine=machine, split=run["split"], kind=kind, model=run["model"],
                                tokens_per_s=1.0, bits_per_weight=bits, n_gpu_layers=run["ngl"],
                                n_cpu_moe=run["n_cpu_moe"], batch=batch, context_len=ctx)  # fmt: skip
                assert predict(m, {machine: section["profile"]}, tokens=1, warmup_tokens=0) is not None, (machine, run, kind)


def test_a_prompt_computes_only_the_last_tokens_logits():
    from basp.core.graph import Graph
    from basp.experiments.harness import build
    from basp.scheduler import placement

    topology = load_topology("macbook_m2")
    spec = MODELS["qwen3-4b"]
    policy = ExecutionPolicy("s", frozenset({DataflowMode.WEIGHT_STATIONARY}), frozenset({0, 1}), joint_rounds=0)
    sizes = {}
    for prompt in (False, True):
        setup = build(topology, spec, policy, lambda p, r: placement.place_static_split(p, r, 1, 0, 18),
                      batch=64, context_len=0, max_new_tokens=1, prompt=prompt)  # fmt: skip
        graph = Graph()
        head = setup.program.lm_head(graph, 0, setup.program.input_tensor(graph, 0)).ops[0]
        sizes[prompt] = (head.batch, head.macs)
    assert sizes[True] == (1, spec.head_params) and sizes[False] == (64, 64 * spec.head_params)


def test_update_profile_refuses_a_profile_that_failed_validation(tmp_path, monkeypatch):
    from benchmarks import run_calibration

    failing = {"prior": {"status": "fail", "validation": {"n": 9, "median": 0.3, "p95": 0.5}, "reasons": ["bad"]},
               "fitted": {"status": "fail", "validation": {"n": 9, "median": 0.3, "p95": 0.5}, "reasons": ["bad"]}}
    seen = {}
    monkeypatch.setattr(run_calibration, "fit_and_score",
                        lambda evidence, machine, profile: seen.update(profile=profile) or failing)
    prior = tmp_path / "seeded.json"
    prior.write_text(json.dumps(load_json("colab_t4")))
    argv = ["--machine", "colab-t4", "--fit-only", "--update-profile", "--evidence", str(tmp_path), "--prior", str(prior)]
    with pytest.raises(SystemExit, match="validation did not pass"):
        run_calibration.main(argv)
    assert seen["profile"] == str(prior)  # the fit starts from the micro-profiled prior
    assert load_json("colab_t4")["name"] == "colab_t4"  # the configs/ profile is untouched


def test_the_colab_notebook_runs_the_plans_colab_machine():
    from benchmarks import run_calibration

    nb = json.loads((CONFIG_DIR.parent / "notebooks" / "calibrate_colab.ipynb").read_text())
    assert nb["nbformat"] == 4 and nb["metadata"]["colab"]["gpuType"] == "T4"
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    plan = json.loads((CONFIG_DIR / "calibration_plan.json").read_text())
    runs = run_calibration.plan_runs(plan, "colab-t4")
    n_cal = sum(r["split"] == "calibration" for r in runs)
    assert f"{len(runs)} configurations ({n_cal} calibration, {len(runs) - n_cal} validation)" in text
    assert "--machine colab-t4" in text and "--prior {EVIDENCE}/profile_prior.json" in text
    for key in plan["machines"]["colab-t4"]["profile"], "colab_t4":
        load_topology(key)
    sizes = {REAL[run["model"]]["file"].split("-")[1] for run in runs}  # "4B", "8B", "14B"
    assert all(f"'{size}'" in text for size in sizes)  # the notebook downloads every model the plan uses
    assert "PUSH = False" in text  # nothing leaves the runtime unless the user says so
