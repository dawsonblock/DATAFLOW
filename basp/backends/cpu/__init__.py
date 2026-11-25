"""CPU backends: ``NumericPayload`` (values for the simulator's dataflow IR, INT4 reference
kernels) and ``CPUBackend`` (the physical runtime's NumPy execution backend)."""

from basp.backends.cpu.backend import CPUBackend
from basp.backends.cpu.executor import NumericPayload

__all__ = ["CPUBackend", "NumericPayload"]
