"""Run and analyse a paired request-latency study under the protocol.

    # exploratory study on the development prompts (arms A and B of this runtime)
    python -m benchmarks.run_study run --study feasibility --comparison internal \
        --target Qwen3-1.7B-Q8_0.gguf --draft Qwen3-0.6B-Q8_0.gguf --tokenizer <HF tokenizer dir> \
        --backend torch --out evidence/studies/feasibility-1.7b

    # try the candidate configurations the protocol allows (development prompts only)
    python -m benchmarks.run_study tune --target ... --draft ... --tokenizer ... --out evidence/studies/tuning-1.7b

    # after tuning: freeze the comparison, then run the decision study on the held-out prompts
    python -m benchmarks.run_study freeze --comparison internal
    python -m benchmarks.run_study run --study decision --comparison internal ... --out evidence/studies/decision

    # re-analyse stored observations; headroom of a profiled component
    python -m benchmarks.run_study analyze evidence/studies/decision
    python -m benchmarks.run_study headroom --fraction 0.30 --reduction 0.50 --target 1.20

Every request is recorded in ``observations.jsonl`` as it finishes (failures included; a rerun
resumes).  ``summary.json`` holds the analysis, the decision under the protocol's rule, the
study's identity and a time projection for the larger studies.  Native-engine arms (C, D, E)
need a running server pinned in the protocol; they have never been run against a real engine
from this repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from basp.bench import Protocol, analyze, decide, headroom, required_fraction
from basp.bench.arms import BaspArms, NativeArm
from basp.evidence import CONFIG_DIR, validity


def _tool_files() -> list[Path]:
    import basp.bench

    return [Path(__file__), *sorted(Path(basp.bench.__file__).parent.glob("*.py"))]


def _encoder(tokenizer: str | None):
    """Text to token ids.  ``bytes`` maps UTF-8 bytes into the vocabulary: for validating the
    harness on small random-weight models only -- it is not any model's tokenizer."""
    if tokenizer == "bytes":
        return lambda text, vocab: [b % vocab for b in text.encode()][:96]
    if not tokenizer:
        raise SystemExit("arms of this runtime need --tokenizer (a Hugging Face tokenizer directory)")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    return lambda text, _vocab: tok(text)["input_ids"]


def _load(out: Path) -> list[dict[str, Any]]:
    path = out / "observations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarise(out: Path, protocol: Protocol, study: str, comparison: str, *, max_new_tokens: int,
              arms: dict[str, dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    observations = _load(out)
    c, crit = protocol.comparison(comparison), protocol.criteria
    spec = protocol.study(study)
    result = analyze(observations, baseline=c["baseline"], candidate=c["candidate"], prompts=meta["prompts"],
                     expected_tokens=max_new_tokens, require_same_output=bool(c.get("same_output_required")),
                     resamples=int(crit["bootstrap_resamples"]), ci_level=float(crit["ci_level"]),
                     seed=int(crit["bootstrap_seed"]))  # fmt: skip
    exploratory = bool(spec.get("exploratory")) or meta.get("reduced", False) or meta.get("harness_validation_only", False)
    decision = decide(result, min_speedup=float(c["min_speedup"]),
                      ci_lower_bound_above=float(crit["ci_lower_bound_above"]),
                      category_worsening_max=float(crit["category_median_latency_worsening_max"]),
                      exploratory=exploratory)  # fmt: skip
    per_arm: dict[str, Any] = {}
    for arm in (c["baseline"], c["candidate"]):
        rows = [o for o in observations if o["arm"] == arm]
        good = [o for o in rows if o["ok"]]
        seconds = sum(o["latency_s"] for o in rows)
        tokens = sum(o["tokens_returned"] for o in good)
        entry: dict[str, Any] = {"requests": len(rows), "failed": len(rows) - len(good), "seconds": seconds,
                                 "returned_tokens": tokens,
                                 "tokens_per_s": tokens / sum(o["latency_s"] for o in good) if good else None}  # fmt: skip
        work = [o["work"] for o in good if "work" in o]
        if work and tokens:
            entry["per_returned_token"] = {k: sum(w[k] for w in work) / tokens for k in work[0]}
        mem = [o["memory"] for o in rows if "memory" in o]
        if mem:
            entry["peak_expert_resident_bytes"] = max(m["peak_expert_resident_bytes"] for m in mem)
            entry["expert_ceiling_bytes"] = mem[0]["expert_ceiling_bytes"]
            entry["process_peak_rss_bytes"] = max((m["process_peak_rss_bytes"] or 0) for m in mem) or None
        io = [o["io"] for o in rows if o.get("io") and o["io"].get("storage_read_bytes") is not None]
        if io:
            entry["storage_read_bytes"] = sum(i["storage_read_bytes"] for i in io)
            entry["io_source"] = io[0]["source"]
        spec_rows = [o["speculation"] for o in good if "speculation" in o]
        if spec_rows:
            verified = sum(s["verified"] for s in spec_rows)
            entry["acceptance"] = sum(s["accepted"] for s in spec_rows) / verified if verified else None
        per_arm[arm] = entry
    # how long the protocol's studies would take at these speeds (prompt processing not separated)
    projection = {}
    for name, st in protocol.data["studies"].items():
        n = len(protocol.prompts(name)) * int(st["repetitions"]) * int(st["max_new_tokens"])
        rates = [per_arm[a]["tokens_per_s"] for a in per_arm]
        projection[name] = {"generated_tokens_per_arm": n,
                            "seconds": sum(n / r for r in rates) if all(rates) else None}  # fmt: skip
    summary = {"protocol": {"name": protocol.data["name"], "version": protocol.data["version"],
                            "comparison_fingerprint": protocol.fingerprint(comparison),
                            "frozen": protocol.frozen(comparison)},
               "study": study, "comparison": comparison, "max_new_tokens": max_new_tokens, "arms": arms,
               "analysis": result, "decision": decision, "per_arm": per_arm, "projection": projection, **meta}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=float) + "\n")
    return summary


def describe(summary: dict[str, Any]) -> str:
    a, d = summary["analysis"], summary["decision"]
    lines = []
    if a["speedup"] is not None:
        lines.append(f"request speed-up of {a['candidate']} over {a['baseline']}: {a['speedup']:.3f}x "
                     f"({a['ci_level']:.0%} interval {a['ci_low']:.3f}-{a['ci_high']:.3f}) over "
                     f"{a['prompts_with_valid_pairs']} of {a['prompts_planned']} prompts, {a['valid_pairs']} valid pairs")  # fmt: skip
        for name, cat in a["categories"].items():
            lines.append(f"  {name:10s} {cat['speedup']:.3f}x; median request latency {cat['median_latency_change']:+.1%}")
    for arm, e in summary["per_arm"].items():
        rate = f"{e['tokens_per_s']:.2f} tokens/s" if e["tokens_per_s"] else "no valid request"
        lines.append(f"  arm {arm}: {e['requests']} requests, {e['failed']} failed, {rate}")
    lines.append(f"verdict: {d['verdict'].replace('_', ' ')} -- " + "; ".join(d["reasons"]))
    return "\n".join(lines)


def run(args) -> int:
    protocol = Protocol.load(args.protocol)
    arms = {k: dict(v) for k, v in protocol.arms.items()}
    c = protocol.comparison(args.comparison)
    study = protocol.study(args.study)
    reduced = False
    if args.arm:  # tuning: an alternative candidate configuration, development prompts only
        if study.get("requires_frozen"):
            raise SystemExit("arm overrides are for tuning on the development prompts, never for a decision study")
        allowed = protocol.data.get("tuning", {}).get(c["candidate"], [])
        override = json.loads(args.arm)
        if override not in allowed:
            raise SystemExit(f"that configuration is not among the protocol's tuning candidates: {allowed}")
        arms[c["candidate"]] = {**override, "engine": "basp", "label": "tuning candidate"}
        protocol.arms[c["candidate"]] = arms[c["candidate"]]
    protocol.check_runnable(args.study, args.comparison)
    if protocol.frozen(args.comparison) and protocol.uses_basp(args.comparison):
        protocol.check_setup(target=args.target, draft=args.draft, backend=args.backend,
                             expert_cache_gib=args.expert_cache_gib, embedding=args.embedding)  # fmt: skip
    max_new = int(study["max_new_tokens"])
    prompts = protocol.prompts(args.study)
    if args.tokens or args.limit_prompts:
        if study.get("requires_frozen"):
            raise SystemExit("a decision study runs exactly as frozen: no --tokens or --limit-prompts")
        reduced = True  # a timing pilot: never a basis for a decision
        max_new = args.tokens or max_new
        prompts = prompts[: args.limit_prompts or len(prompts)]
    keep = {p["id"] for p in prompts}
    trials = [t for t in protocol.schedule(args.study, args.comparison) if t.prompt_id in keep]
    if args.repetitions:
        if study.get("requires_frozen"):
            raise SystemExit("a decision study runs exactly as frozen: no --repetitions")
        reduced = True
        trials = [t for t in trials if t.repetition < args.repetitions]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engines = {arms[k]["engine"] for k in (c["baseline"], c["candidate"])}
    basp = text_of = None
    models: dict[str, str] = {}
    if "basp" in engines:
        encode = _encoder(args.tokenizer)
        prefer = ("torch",) if args.backend == "torch" else ("cpu",)
        cache = None if args.expert_cache_gib is None else int(args.expert_cache_gib * (1 << 30))
        basp = BaspArms(args.target, args.draft, prefer=prefer, expert_cache_bytes=cache, embedding=args.embedding)
        models = {"target": args.target, **({"draft": args.draft} if args.draft else {})}
        ids = {p["id"]: encode(p["text"], basp.target.config.vocab) for p in prompts}
    text_of = {p["id"]: p["text"] for p in prompts}
    native = {k: NativeArm(arms[k], timeout_s=args.timeout) for k in (c["baseline"], c["candidate"])
              if arms[k]["engine"] == "native"}  # fmt: skip
    meta = {
        "prompts": [{"id": p["id"], "category": p["category"], "context": p.get("context")} for p in prompts],
        "regime": {k: v for k, v in protocol.data["regime"].items() if not k.startswith("_")},
        "reduced": reduced,
        "harness_validation_only": args.tokenizer == "bytes",
        "startup_seconds": basp.startup_s if basp else None,
        "identity": validity.measurement_identity(
            tool="benchmarks.run_study", tool_files=_tool_files(), models=models,
            parameters={"study": args.study, "comparison": args.comparison, "max_new_tokens": max_new,
                        "arms": {k: arms[k] for k in (c["baseline"], c["candidate"])},
                        "expert_cache_gib": args.expert_cache_gib, "embedding": args.embedding,
                        "protocol": protocol.fingerprint(args.comparison)},
            backend=basp.backend if basp else "native"),
    }  # fmt: skip
    policy = protocol.data["regime"]["expert_cache"]

    def execute(arm_key: str, prompt_id: str, n: int) -> dict[str, Any]:
        arm = arms[arm_key]
        if arm["engine"] == "native":
            return native[arm_key].request(text_of[prompt_id], n)
        return basp.request(arm, ids[prompt_id], n, expert_cache=policy, timeout_s=args.timeout)

    done = {(o["pair"], o["arm"]) for o in _load(out)}
    if not done:  # one unmeasured warm-up request per arm
        for arm_key in (c["baseline"], c["candidate"]):
            warm = execute(arm_key, prompts[0]["id"], min(8, max_new))
            if not warm["ok"]:
                print(f"warm-up of arm {arm_key} failed: {warm['failure']}", flush=True)
    start = time.perf_counter()
    with open(out / "observations.jsonl", "a") as log:
        for i, trial in enumerate(trials, 1):
            if (trial.pair, trial.arm) in done:
                continue
            obs = execute(trial.arm, trial.prompt_id, max_new)
            obs.update(pair=trial.pair, prompt_id=trial.prompt_id, category=trial.category,
                       repetition=trial.repetition, arm=trial.arm, position=trial.position)  # fmt: skip
            log.write(json.dumps(obs, default=float) + "\n")
            log.flush()
            state = f"{obs['latency_s']:.1f} s" if obs["ok"] else f"FAILED: {obs['failure']}"
            print(f"[{i}/{len(trials)}] {trial.prompt_id} rep {trial.repetition} arm {trial.arm}: {state} "
                  f"({time.perf_counter() - start:.0f} s elapsed)", flush=True)  # fmt: skip
    summary = summarise(out, protocol, args.study, args.comparison, max_new_tokens=max_new,
                        arms={k: arms[k] for k in (c["baseline"], c["candidate"])}, meta=meta)  # fmt: skip
    print(describe(summary))
    print(f"wrote {out}/summary.json")
    return 0


def tune(args) -> int:
    """Run every tuning candidate the protocol allows against the baseline, development prompts."""
    protocol = Protocol.load(args.protocol)
    c = protocol.comparison(args.comparison)
    rows = []
    for i, candidate in enumerate(protocol.data["tuning"][c["candidate"]]):
        options = {**vars(args), "study": "feasibility", "arm": json.dumps(candidate),
                   "out": str(Path(args.out) / f"candidate-{i}"), "limit_prompts": None}  # fmt: skip
        sub = argparse.Namespace(**options)
        run(sub)
        summary = json.loads((Path(sub.out) / "summary.json").read_text())
        rows.append({"candidate": candidate, "speedup": summary["analysis"]["speedup"],
                     "ci": [summary["analysis"]["ci_low"], summary["analysis"]["ci_high"]],
                     "verdict": summary["decision"]["verdict"]})  # fmt: skip
    rows.sort(key=lambda r: -(r["speedup"] or 0))
    (Path(args.out) / "tuning.json").write_text(json.dumps(rows, indent=1) + "\n")
    for r in rows:
        print(f"{r['speedup'] or float('nan'):.3f}x  {r['candidate']}")
    print("Write the chosen configuration into the protocol's candidate arm, then freeze the comparison.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "tune"):
        p = sub.add_parser(name)
        p.add_argument("--protocol", default=str(CONFIG_DIR / "benchmark_protocol.json"))
        p.add_argument("--comparison", default="internal")
        p.add_argument("--target", help="target model (GGUF), for arms of this runtime")
        p.add_argument("--draft", help="draft model (GGUF)")
        p.add_argument("--tokenizer", help="directory with the models' Hugging Face tokenizer")
        p.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
        p.add_argument("--expert-cache-gib", type=float, help="hard ceiling on resident expert arrays")
        p.add_argument("--embedding", choices=["resident", "rows"], default="resident")
        p.add_argument("--timeout", type=float, help="seconds per request")
        p.add_argument("--tokens", type=int, help="timing pilot: tokens per request instead of the study's")
        p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--study", default="feasibility")
            p.add_argument("--arm", help="tuning: JSON configuration of the candidate arm (must be a tuning candidate)")
            p.add_argument("--limit-prompts", type=int, help="timing pilot: only the first N prompts")
            p.add_argument("--repetitions", type=int, help="timing pilot: fewer repetitions")
        else:
            p.set_defaults(repetitions=None)
    p = sub.add_parser("analyze")
    p.add_argument("out")
    p.add_argument("--protocol", default=str(CONFIG_DIR / "benchmark_protocol.json"))
    p = sub.add_parser("freeze")
    p.add_argument("--protocol", default=str(CONFIG_DIR / "benchmark_protocol.json"))
    p.add_argument("--comparison", required=True)
    p = sub.add_parser("headroom")
    p.add_argument("--fraction", type=float, required=True, help="share of request time the component takes")
    p.add_argument("--reduction", type=float, required=True, help="share of that component the intervention removes")
    p.add_argument("--target", type=float, default=1.20)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "tune":
        return tune(args)
    if args.command == "freeze":
        protocol = Protocol.load(args.protocol)
        print(f"comparison {args.comparison!r} frozen at {protocol.freeze(args.comparison)}")
        protocol.save()
        return 0
    if args.command == "headroom":
        gain = headroom(args.fraction, args.reduction)
        need = required_fraction(args.target, args.reduction)
        print(f"removing {args.reduction:.0%} of a component that takes {args.fraction:.0%} of request time gives "
              f"{gain:.3f}x ({1 - 1 / gain:.1%} less time)")  # fmt: skip
        print(f"to reach {args.target:g}x with that reduction the component must take at least {need:.0%} of the time"
              + (" -- impossible" if need > 1 else ""))  # fmt: skip
        return 0 if gain >= args.target else 1
    out = Path(args.out)
    previous = json.loads((out / "summary.json").read_text())
    protocol = Protocol.load(args.protocol)
    meta = {k: previous.get(k) for k in ("prompts", "regime", "reduced", "harness_validation_only",
                                         "startup_seconds", "identity")}  # fmt: skip
    summary = summarise(out, protocol, previous["study"], previous["comparison"],
                        max_new_tokens=previous["max_new_tokens"], arms=previous["arms"], meta=meta)  # fmt: skip
    print(describe(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
