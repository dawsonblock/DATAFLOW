"""A collection of receipts, stored as JSON, queried by metric, candidate and regime."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .receipts import Receipt


class ReceiptRegistry:
    def __init__(self, receipts: Iterable[Receipt] = ()) -> None:
        self._items: dict[str, Receipt] = {}
        for receipt in receipts:
            self.add(receipt)

    def add(self, receipt: Receipt) -> Receipt:
        """Add a receipt (identical content is stored once).  Invalid receipts are kept: the
        qualification step reports why they cannot count, rather than losing them silently."""
        self._items.setdefault(receipt.receipt_id, receipt)
        return receipt

    def extend(self, receipts: Iterable[Receipt]) -> None:
        for receipt in receipts:
            self.add(receipt)

    def find(self, metric: str, *, candidate: str | None = None, regime: str | None = None) -> list[Receipt]:
        return [r for r in self._items.values() if r.metric == metric
                and (candidate is None or r.candidate == candidate) and (regime is None or r.regime == regime)]  # fmt: skip

    def __iter__(self) -> Iterator[Receipt]:
        return iter(sorted(self._items.values(), key=lambda r: (r.metric, r.candidate, r.regime, r.receipt_id)))

    def __len__(self) -> int:
        return len(self._items)

    def to_json(self) -> list[dict]:
        return [r.to_json() for r in self]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ReceiptRegistry:
        return cls(Receipt.from_json(item) for item in json.loads(Path(path).read_text()))
