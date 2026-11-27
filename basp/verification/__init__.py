"""Verification.

* Simulation checks (v0.1-v0.3): numerical co-simulation, dependency and accounting checks.
* Runtime parity (v0.4): the physical runtime against the transformers reference on the
  same GGUF weights -- logits, layer outputs, expert routing, greedy generation, perplexity
  (``parity_report.run_parity``; needs torch and transformers).
"""

from basp.verification.accounting import check_accounting
from basp.verification.dependency import check_dependencies
from basp.verification.intermediate_tensors import compare_hidden
from basp.verification.logits import compare_logits, rel_l2
from basp.verification.numerical import compare_to_reference
from basp.verification.perplexity import perplexity
from basp.verification.routing import compare_routing

__all__ = ["check_accounting", "check_dependencies", "compare_hidden", "compare_logits", "compare_routing",
           "compare_to_reference", "perplexity", "rel_l2"]  # fmt: skip
