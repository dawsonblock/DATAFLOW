"""Execution backends.

* ``virtual`` simulates timing and ``cpu.NumericPayload`` computes values for the simulator
  (the ``Payload`` contract, ``base``);
* ``cpu.CPUBackend`` executes real models for the physical runtime (the ``ExecutionBackend``
  contract, ``execution``); ``cuda``, ``metal``, ``ggml`` and ``remote`` are declared but not
  implemented: they report themselves unavailable and refuse every call.
"""
