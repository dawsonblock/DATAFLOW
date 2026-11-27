"""Analysis of a paired request-latency study, and the decision it supports.

The independent unit is the **prompt**.  Repetitions of a prompt are not independent prompts:
each prompt contributes one number, the geometric mean of its paired ratios

    speed-up = baseline request latency / candidate request latency     (same prompt, same pair)

and the study's estimate is the geometric mean of those over prompts, every prompt weighted
equally.  Uncertainty comes from resampling prompts (within each category, so category weights
are preserved), each resampled prompt bringing all its repetitions with it.

Nothing is dropped silently: a pair in which either request failed, timed out, returned a
different number of tokens than asked, or (where outputs must agree) a different output, is
listed as a failure, and a prompt without a valid pair is missing coverage -- which makes the
result inconclusive, not better.

``decide`` applies the pre-declared rule and can answer ``meets the criterion``, ``no
improvement shown`` (the interval includes 1), ``improvement below the criterion``, ``slower``,
or ``inconclusive``.  A 1.20x speed-up is about 16.7% less time, not 20%.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def headroom(fraction: float, reduction: float) -> float:
    """Speed-up if ``fraction`` of total time is addressable and an intervention removes
    ``reduction`` of that part, everything else unchanged: ``1 / (1 - f r)``."""
    if not (0.0 <= fraction <= 1.0 and 0.0 <= reduction <= 1.0):
        raise ValueError("fraction and reduction must be within [0, 1]")
    return 1.0 / (1.0 - fraction * reduction) if fraction * reduction < 1.0 else math.inf


def required_fraction(target_speedup: float, reduction: float) -> float:
    """Share of total time a component must have for removing ``reduction`` of it to reach
    ``target_speedup`` (above 1: impossible with that reduction)."""
    if target_speedup <= 1.0 or not 0.0 < reduction <= 1.0:
        raise ValueError("need a target above 1 and a reduction in (0, 1]")
    return (1.0 - 1.0 / target_speedup) / reduction


def _geomean(values: Sequence[float]) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=np.float64)))))


def _invalid(obs: Mapping[str, Any], expected_tokens: int | None) -> str | None:
    """Why an observation cannot be used (``None``: it can)."""
    if not obs.get("ok", False):
        return str(obs.get("failure") or "failed")
    latency = obs.get("latency_s")
    if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
        return "no valid latency"
    if expected_tokens is not None and obs.get("tokens_returned") != expected_tokens:
        return f"returned {obs.get('tokens_returned')} tokens, {expected_tokens} were asked for"
    return None


def analyze(observations: Sequence[Mapping[str, Any]], *, baseline: str, candidate: str,
            prompts: Sequence[Mapping[str, Any]], expected_tokens: int | None = None,
            require_same_output: bool = False, resamples: int = 10_000, ci_level: float = 0.95,
            seed: int = 0) -> dict[str, Any]:  # fmt: skip
    """Paired analysis of one comparison.

    ``observations``: one record per request (``pair``, ``prompt_id``, ``arm``, ``ok``,
    ``latency_s``, ``tokens_returned``, optionally ``token_ids``).  ``prompts``: the planned
    prompts (``id``, ``category``) -- coverage is judged against the plan, not against what
    happened to produce data.
    """
    by_pair: dict[Any, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for obs in observations:
        if obs["arm"] in (baseline, candidate):
            by_pair[(obs["prompt_id"], obs["pair"])][obs["arm"]] = obs
    ratios: dict[str, list[float]] = defaultdict(list)
    latencies: dict[str, dict[str, list[float]]] = {baseline: defaultdict(list), candidate: defaultdict(list)}
    failures: list[dict[str, Any]] = []
    orders = {"baseline_first": 0, "candidate_first": 0}
    for (prompt_id, pair), arms in sorted(by_pair.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        problems = []
        for arm in (baseline, candidate):
            if arm not in arms:
                problems.append(f"arm {arm}: no observation")
            else:
                reason = _invalid(arms[arm], expected_tokens)
                if reason:
                    problems.append(f"arm {arm}: {reason}")
        if not problems and require_same_output:
            a, b = arms[baseline].get("token_ids"), arms[candidate].get("token_ids")
            if a is None or b is None:
                problems.append("outputs were not recorded, so they cannot be compared")
            elif list(a) != list(b):
                first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                problems.append(f"the candidate's output differs from the baseline's from token {first}")
        if problems:
            failures.append({"prompt_id": prompt_id, "pair": pair, "problems": problems})
            continue
        ratios[prompt_id].append(arms[baseline]["latency_s"] / arms[candidate]["latency_s"])
        for arm in (baseline, candidate):
            latencies[arm][prompt_id].append(arms[arm]["latency_s"])
        orders["baseline_first" if arms[baseline].get("position", 0) == 0 else "candidate_first"] += 1

    category_of = {p["id"]: p["category"] for p in prompts}
    planned = [p["id"] for p in prompts]
    covered = [pid for pid in planned if ratios.get(pid)]
    missing = [pid for pid in planned if not ratios.get(pid)]
    per_prompt = {pid: _geomean(ratios[pid]) for pid in covered}
    out: dict[str, Any] = {
        "baseline": baseline, "candidate": candidate, "prompts_planned": len(planned),
        "prompts_with_valid_pairs": len(covered), "prompts_missing": missing, "valid_pairs": sum(map(len, ratios.values())),
        "failures": failures, "order_balance": orders, "per_prompt_speedup": per_prompt, "ci_level": ci_level,
        "resamples": resamples,
    }  # fmt: skip
    if not covered:
        out.update(speedup=None, ci_low=None, ci_high=None, categories={})
        return out
    out["speedup"] = _geomean(list(per_prompt.values()))
    out["time_saved_fraction"] = 1.0 - 1.0 / out["speedup"]

    # resample prompts within categories; a prompt brings every one of its repetitions
    strata: dict[str, list[str]] = defaultdict(list)
    for pid in covered:
        strata[category_of[pid]].append(pid)
    rng = np.random.default_rng(seed)
    logs = {pid: math.log(v) for pid, v in per_prompt.items()}
    total = len(covered)
    samples = np.zeros(resamples)
    for ids in strata.values():
        values = np.array([logs[pid] for pid in ids])
        draws = rng.integers(0, len(ids), size=(resamples, len(ids)))
        samples += values[draws].sum(axis=1)
    samples = np.exp(samples / total)
    alpha = (1.0 - ci_level) / 2.0
    out["ci_low"], out["ci_high"] = (float(x) for x in np.quantile(samples, [alpha, 1.0 - alpha]))
    out["ci_method"] = "percentile bootstrap over prompts, stratified by category"
    if len(covered) < 2 or any(len(ids) < 2 for ids in strata.values()):
        out["ci_note"] = "a category with a single prompt contributes no resampling variation: the interval is too narrow"

    categories = {}
    for category, ids in sorted(strata.items()):
        medians = {arm: float(np.median([np.median(latencies[arm][pid]) for pid in ids])) for arm in (baseline, candidate)}
        categories[category] = {
            "prompts": len(ids), "speedup": _geomean([per_prompt[pid] for pid in ids]),
            "median_latency_s": medians,
            # positive: the candidate's median request in this category takes longer
            "median_latency_change": medians[candidate] / medians[baseline] - 1.0,
        }  # fmt: skip
    out["categories"] = categories
    return out


def decide(analysis: Mapping[str, Any], *, min_speedup: float, ci_lower_bound_above: float = 1.0,
           category_worsening_max: float = 0.05, exploratory: bool = False) -> dict[str, Any]:  # fmt: skip
    """Apply the pre-declared rule.  ``exploratory`` studies never produce a go: at best they say
    whether a decision study is worth running."""
    reasons: list[str] = []
    speedup, low, high = analysis.get("speedup"), analysis.get("ci_low"), analysis.get("ci_high")
    guards = {c: v["median_latency_change"] for c, v in analysis.get("categories", {}).items()}
    worse = {c: v for c, v in guards.items() if v > category_worsening_max}
    if analysis["prompts_missing"] or analysis["failures"]:
        n = len(analysis["prompts_missing"])
        reasons.append(f"{len(analysis['failures'])} pair(s) failed or were invalid; {n} prompt(s) have no valid pair")
    if speedup is None:
        verdict = "inconclusive"
        reasons.append("no valid pair at all")
    elif analysis["prompts_missing"]:
        verdict = "inconclusive"
        reasons.append("coverage is incomplete: the planned prompts were not all measured")
    elif high < 1.0:
        verdict = "slower"
        reasons.append(f"the candidate is slower: {speedup:.3f}x ({low:.3f}-{high:.3f})")
    elif low <= ci_lower_bound_above:
        verdict = "inconclusive"
        reasons.append(f"the {analysis['ci_level']:.0%} interval {low:.3f}-{high:.3f} includes no improvement")
    elif speedup < min_speedup:
        verdict = "below_criterion"
        reasons.append(f"an improvement of {speedup:.3f}x is shown, below the required {min_speedup:g}x")
    elif worse:
        verdict = "category_regression"
        reasons.append("median request latency worsens by more than "
                       f"{category_worsening_max:.0%} in: " + ", ".join(f"{c} ({v:+.1%})" for c, v in sorted(worse.items())))  # fmt: skip
    else:
        verdict = "meets_criterion"
        reasons.append(f"{speedup:.3f}x ({low:.3f}-{high:.3f}), at least {min_speedup:g}x with the lower bound above "
                       f"{ci_lower_bound_above:g}; no category worsens by more than {category_worsening_max:.0%}")  # fmt: skip
    if exploratory:
        reasons.append("exploratory study: too small to support an acceleration claim; it only says whether a "
                       "decision study is worth running")  # fmt: skip
    return {"verdict": verdict, "supports_decision": verdict == "meets_criterion" and not exploratory,
            "min_speedup": min_speedup, "exploratory": exploratory, "reasons": reasons,
            "category_guard": guards}  # fmt: skip
