"""Discrete-event simulation kernel."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from basp.simulator.event_queue import Event, EventQueue


class SimulationError(RuntimeError):
    pass


class Simulator:
    """Event-driven clock.  Time is in seconds and never moves backwards."""

    def __init__(self) -> None:
        self.now = 0.0
        self.events_processed = 0
        self._queue = EventQueue()

    def at(self, time: float, fn: Callable[..., Any], *args: Any) -> Event:
        if time < self.now:
            raise SimulationError(f"cannot schedule at {time} before now={self.now}")
        return self._queue.push(time, fn, *args)

    def after(self, delay: float, fn: Callable[..., Any], *args: Any) -> Event:
        if delay < 0:
            raise SimulationError(f"negative delay {delay}")
        return self._queue.push(self.now + delay, fn, *args)

    def cancel(self, event: Event | None) -> None:
        if event is not None:
            self._queue.cancel(event)

    @property
    def pending(self) -> int:
        return len(self._queue)

    def run(self, until: float | None = None, max_events: int | None = None) -> float:
        """Process events until the queue drains, ``until`` passes or the budget ends."""
        budget = max_events
        while True:
            next_time = self._queue.peek_time()
            if next_time is None or (until is not None and next_time > until):
                break
            if budget is not None:
                if budget == 0:
                    raise SimulationError("event budget exhausted (possible livelock)")
                budget -= 1
            event = self._queue.pop()
            assert event is not None
            self.now = event.time
            self.events_processed += 1
            event.fn(*event.args)
        if until is not None and until > self.now:
            self.now = until
        return self.now
