"""Critical-path analysis of an operation batch.

``upward_ranks`` gives each operation the length of the longest path from it to
any sink (its own cost included).  Scheduling in descending rank order visits
operations in a valid topological order and handles the critical path first,
as in HEFT list scheduling.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable

from basp.core.graph import Graph


def upward_ranks(graph: Graph, ops: Iterable[int], cost: Callable[[int], float]) -> dict[int, float]:
    ids = set(ops)
    order = graph.topo_order(ids)
    rank: dict[int, float] = {}
    for op_id in reversed(order):
        tail = max((rank[d] for d in graph.dependents(op_id) if d in ids), default=0.0)
        rank[op_id] = cost(op_id) + tail
    return rank


def priority_order(graph: Graph, ops: Iterable[int], cost: Callable[[int], float]) -> list[int]:
    """Topological order that always takes the ready operation of highest upward rank."""
    ids = set(ops)
    rank = upward_ranks(graph, ids, cost)
    waiting = {i: sum(1 for d in graph.deps(i) if d in ids) for i in ids}
    ready = [(-rank[i], i) for i, n in waiting.items() if n == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        _, op_id = heapq.heappop(ready)
        order.append(op_id)
        for user in graph.dependents(op_id):
            if user in ids:
                waiting[user] -= 1
                if waiting[user] == 0:
                    heapq.heappush(ready, (-rank[user], user))
    return order


def critical_path(graph: Graph, ops: Iterable[int], cost: Callable[[int], float]) -> tuple[list[int], float]:
    """The longest dependency chain in the batch and its total cost."""
    ids = set(ops)
    rank = upward_ranks(graph, ids, cost)
    if not rank:
        return [], 0.0
    current = max(sorted(ids), key=lambda i: rank[i])
    path = [current]
    while True:
        nxt = [d for d in graph.dependents(current) if d in ids]
        if not nxt:
            break
        current = max(sorted(nxt), key=lambda i: rank[i])
        path.append(current)
    return path, rank[path[0]]
