"""The pre-registered calibration plan (``configs/calibration_plan.json``), expanded into runs."""

from __future__ import annotations

from typing import Any


def plan_runs(plan: dict[str, Any], machine: str) -> list[dict[str, Any]]:
    """Every planned configuration for ``machine``, one dictionary per llama-bench invocation."""
    if machine not in plan["machines"]:
        raise SystemExit(f"machine {machine!r} is not in the plan; known: {sorted(plan['machines'])}")
    defaults = plan.get("defaults", {})
    out = []
    for entry in plan["machines"][machine]["runs"]:
        if entry["split"] not in ("calibration", "validation"):
            raise SystemExit(f"plan entry {entry} has no valid split")
        for ngl in entry["ngl"]:
            for depth in entry.get("depth", [0]):
                for cmoe in entry.get("n_cpu_moe", [0]):
                    out.append({"split": entry["split"], "model": entry["model"], "ngl": ngl, "depth": depth,
                                "n_cpu_moe": cmoe, "repetitions": entry.get("repetitions", defaults.get("repetitions", 3)),
                                "n_prompt": entry.get("n_prompt", defaults.get("n_prompt")),
                                "n_gen": entry.get("n_gen", defaults.get("n_gen"))})  # fmt: skip
    return out
