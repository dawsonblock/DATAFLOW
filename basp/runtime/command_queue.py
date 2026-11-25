"""In-order command queue per backend."""

from __future__ import annotations

from basp.backends.execution import Completion, ExecutionBackend, Op


class CommandQueue:
    def __init__(self, backend: ExecutionBackend) -> None:
        self.backend = backend
        self.submitted = 0
        self._pending: list[Completion] = []

    def enqueue(self, op: Op) -> Completion:
        completion = self.backend.submit(op)
        self.submitted += 1
        if not completion.done:
            self._pending.append(completion)
        return completion

    def flush(self) -> None:
        """Wait for everything submitted so far."""
        if self._pending:
            self.backend.synchronize()
            self._pending = [c for c in self._pending if not c.done]
            if self._pending:
                raise RuntimeError(f"{len(self._pending)} operations still pending after synchronize")
