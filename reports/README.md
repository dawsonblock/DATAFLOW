# Reports

| Directory | Produced by | Status |
|---|---|---|
| `v0.2/` | `python -m benchmarks.explore` (writes `dse.json`), then `python -m benchmarks.run_report` | Current. Simulated on an uncalibrated simulator; see the note at the top of the report |
| `v0.1/` | the v0.1 code (`python -m benchmarks.run_report --release v0.1` at that commit) | Historical record. The v0.1 model charged one GPU launch per expert, pinned the output projection to the modules and assumed a slower host CPU; its numbers are superseded and cannot be regenerated from the current code |

Both generators are deterministic. Nothing in either directory is a measurement of real hardware.

One statement in the v0.1 report does not generalise: it records that fused and unfused graphs
gave bit-identical logits. That was observed for one model seed under a policy that accumulates
partial sums on devices, where it holds only by chance. Bit-identity is guaranteed only between
dataflows that keep the reduction order; see section 8 of the v0.2 report.
