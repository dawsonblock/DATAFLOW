"""Critical-path analysis of an executed decoding run.

For every decoded token the realised critical path is walked backwards from
the operation that finished last through, at each step, the dependency whose
result arrived last.  The time on that chain is attributed

* by **component**: ``kernel`` (computing), ``input`` (waiting for activations
  or weights to arrive), ``queue`` (inputs present, device busy), ``output``
  (returning results) and ``host`` (the gap between a result being available
  and the host issuing the operations that depend on it), and
* by **operation kind**: trunk, expert, reduce, head.

A token's interval runs from the delivery of the previous token's last result
to the delivery of its own, so the host round trip between two tokens is
counted (as ``host``) and the intervals add up to the run.  The chain's
components sum to the token time, so the breakdown says exactly where a
token's latency goes.  From it follows what more modules can and cannot
buy: only the expert phase is divisible among modules; everything else on the
chain is a serial dependency.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from basp.backends.virtual.runtime import OpRecord

COMPONENTS = ("kernel", "input", "queue", "output", "host")


@dataclass
class CriticalPathReport:
    tokens: int
    token_time_s: float  # mean time per token
    by_component: dict[str, float] = field(default_factory=dict)  # seconds per token
    by_kind: dict[str, float] = field(default_factory=dict)  # seconds per token (host separate)
    expert_work_s: float = 0.0  # kernel time of *all* expert operations per token (divisible work)
    expert_phase_s: float = 0.0  # time expert operations occupy on the critical path per token
    expert_ops: float = 0.0  # expert operations per token
    longest_expert_s: float = 0.0  # mean kernel time of one expert operation
    residual_s: float = 0.0  # |token time - sum of components|, should be ~0

    @property
    def serial_s(self) -> float:
        """Critical-path time that adding modules cannot divide."""
        return self.token_time_s - self.expert_phase_s

    def share(self, name: str) -> float:
        value = self.by_component.get(name, self.by_kind.get(name, 0.0))
        return value / self.token_time_s if self.token_time_s > 0 else 0.0


def analyse_critical_path(op_log: Sequence[OpRecord], *, skip_steps: int = 0) -> CriticalPathReport:
    """Analyse a run recorded with ``keep_log=True``; the first ``skip_steps`` steps are ignored."""
    groups: dict[tuple[int, int], list[OpRecord]] = defaultdict(list)
    ends: dict[tuple[int, int], float] = {}  # (stream, step) -> delivery of its last result
    for record in op_log:
        if record.delivered is None or record.started is None:
            continue
        key = (record.stream, record.step)
        ends[key] = max(ends.get(key, record.delivered), record.delivered)
        if record.step >= skip_steps:
            groups[key].append(record)
    component: dict[str, float] = defaultdict(float)
    kind: dict[str, float] = defaultdict(float)
    total = residual = work = phase = 0.0
    n_expert = 0
    for (stream, step), records in groups.items():
        by_id = {r.op_id: r for r in records}
        begin = ends.get((stream, step - 1), min(r.issued for r in records))
        current = max(records, key=lambda r: (r.delivered, r.op_id))
        token_time = current.delivered - begin
        accounted = 0.0
        while current is not None:
            deps = [by_id[d] for d in current.deps if d in by_id]
            pred = max(deps, key=lambda r: (r.delivered, r.op_id)) if deps else None
            base = pred.delivered if pred is not None else begin
            ready = max(base, current.issued)
            arrived = current.inputs_ready if current.inputs_ready is not None else ready
            parts = {
                "host": max(current.issued - base, 0.0),
                "input": min(max(arrived - ready, 0.0), max(current.started - ready, 0.0)),
                "kernel": current.finished - current.started,
                "output": current.delivered - current.finished,
            }
            parts["queue"] = max(current.started - ready - parts["input"], 0.0)
            for name, value in parts.items():
                component[name] += value
                accounted += value
            own = sum(v for k, v in parts.items() if k != "host")
            kind[current.kind] += own
            if current.kind == "expert":
                phase += own
            current = pred
        total += token_time
        residual += abs(token_time - accounted)
        experts = [r for r in records if r.kind == "expert"]
        work += sum(r.finished - r.started for r in experts)
        n_expert += len(experts)
    n = len(groups)
    if n == 0:
        return CriticalPathReport(0, 0.0)
    return CriticalPathReport(
        tokens=n,
        token_time_s=total / n,
        by_component={k: component[k] / n for k in COMPONENTS},
        by_kind={k: v / n for k, v in sorted(kind.items())},
        expert_work_s=work / n,
        expert_phase_s=phase / n,
        expert_ops=n_expert / n,
        longest_expert_s=work / n_expert if n_expert else 0.0,
        residual_s=residual / n,
    )


def scaling_limit(report: CriticalPathReport, measured_modules: int, modules: int, *,
                  layers: int) -> dict[str, float]:  # fmt: skip
    """What ``modules`` modules could deliver if the expert work divided perfectly.

    ``report`` is the critical path measured with ``measured_modules`` modules.
    Its serial part is kept; the expert phase is replaced by the best case in
    which each layer's expert work is spread evenly (but no layer can finish
    faster than one expert kernel).  Returns the ideal token time and the
    scaling efficiency relative to one module under the same idealisation.
    """
    per_layer_work = report.expert_work_s / layers
    one_expert = report.longest_expert_s

    def ideal(n: int) -> float:
        return report.serial_s + layers * max(per_layer_work / n, one_expert)

    t1, tn = ideal(1), ideal(modules)
    return {
        "ideal_token_time_s": tn,
        "ideal_speedup": t1 / tn,
        "ideal_efficiency": (t1 / tn) / modules,
        "measured_modules": float(measured_modules),
    }
