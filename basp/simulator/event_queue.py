"""Deterministic event queue: events at equal times fire in scheduling order."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import Any


class Event:
    __slots__ = ("args", "cancelled", "fn", "seq", "time")

    def __init__(self, time: float, seq: int, fn: Callable[..., Any], args: tuple[Any, ...]):
        self.time = time
        self.seq = seq
        self.fn = fn
        self.args = args
        self.cancelled = False

    def __lt__(self, other: Event) -> bool:
        return (self.time, self.seq) < (other.time, other.seq)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._seq = 0
        self._live = 0

    def push(self, time: float, fn: Callable[..., Any], *args: Any) -> Event:
        event = Event(time, self._seq, fn, args)
        self._seq += 1
        self._live += 1
        heapq.heappush(self._heap, event)
        return event

    def cancel(self, event: Event) -> None:
        if not event.cancelled:
            event.cancelled = True
            self._live -= 1

    def pop(self) -> Event | None:
        """Next live event, or ``None`` when the queue is empty."""
        while self._heap:
            event = heapq.heappop(self._heap)
            if not event.cancelled:
                self._live -= 1
                event.cancelled = True  # a fired event can no longer be cancelled
                return event
        return None

    def peek_time(self) -> float | None:
        while self._heap and self._heap[0].cancelled:
            heapq.heappop(self._heap)
        return self._heap[0].time if self._heap else None

    def __len__(self) -> int:
        return self._live
