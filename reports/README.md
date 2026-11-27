# Reports

| Directory | Produced by | Status |
|---|---|---|
| `v0.6/` | `python -m benchmarks.explore --trace evidence/traces/qwen3-30b-a3b.decode.json` (writes `dse.json`), then `python -m benchmarks.run_report --trace ...` | Current. Simulated, except the runtime sections (9 and 11), which are measurements on the verification machine with their validity stated. The header states the calibration status, the routing and the compression data used; `provenance.json` records the code, configurations, inputs and environment |
| `v0.5/`, `v0.4/` | the v0.5 and v0.4 code | Historical records, superseded. Produced before the v0.6 repairs: no embedding, final norm or shared experts in the lowering, and evidence without complete identities |
| `v0.3/` | the v0.3 code | Historical record, superseded. Simulated, with its provenance |
| `v0.2/` | the v0.2 code | Historical record, superseded. Produced before the v0.2 audit fixes: module links on the local fabric were free, price corners favoured modules, the host CPU was modelled too slow for prompts, and the gate borrowed one architecture's scaling for another |
| `v0.1/` | the v0.1 code | Historical record, superseded. One GPU launch per expert, output projection pinned to the modules, slow host CPU |

No historical report can be regenerated from the current code. Apart from the runtime sections of
the current report, nothing in any directory is a measurement of real hardware, and none is a
measurement of the target machines.

The v0.1 report records that fused and unfused graphs gave bit-identical logits. That was
observed for one model seed under a policy that accumulates partial sums on devices, where it
holds only by chance; bit-identity is guaranteed only between dataflows that keep the reduction
order.
