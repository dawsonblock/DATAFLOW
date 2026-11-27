"""Controlled request-level benchmarking: a protocol fixed before results exist, paired and
counterbalanced trials, prompts as the independent units, and explicit continue/stop rules.

* ``protocol`` -- the study definition (arms, prompts, regime, order) and its fingerprint;
* ``analysis`` -- paired latency ratios, prompt-level bootstrap, category guards, decisions;
* ``arms`` -- how one request is executed and measured, for this runtime and for a native
  inference server.
"""

from .analysis import analyze, decide, headroom, required_fraction
from .protocol import Protocol, ProtocolError, Trial

__all__ = ["Protocol", "ProtocolError", "Trial", "analyze", "decide", "headroom", "required_fraction"]
