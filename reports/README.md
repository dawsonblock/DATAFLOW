# Reports

| Directory | Produced by | Status |
|---|---|---|
| `v0.3/` | `python -m benchmarks.explore` (writes `dse.json`), then `python -m benchmarks.run_report` | Current. Simulated; the header of the report states the calibration status, the routing and the compression data it used. `provenance.json` records the code, configurations, inputs and environment |
| `v0.2/` | the v0.2 code | Historical record, superseded. Produced before the v0.2 audit fixes: module links on the local fabric were free, price corners favoured modules, the host CPU was modelled too slow for prompts, and the gate borrowed one architecture's scaling for another |
| `v0.1/` | the v0.1 code | Historical record, superseded. One GPU launch per expert, output projection pinned to the modules, slow host CPU |

Neither historical report can be regenerated from the current code. Nothing in any directory is a
measurement of real hardware.

The v0.1 report records that fused and unfused graphs gave bit-identical logits. That was
observed for one model seed under a policy that accumulates partial sums on devices, where it
holds only by chance; bit-identity is guaranteed only between dataflows that keep the reduction
order.
