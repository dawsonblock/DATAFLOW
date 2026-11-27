"""Logical tensor namespace: one id space across physically separate memory."""

from __future__ import annotations

from basp.core.tensor import (
    CompressionFormat,
    DType,
    FormatTable,
    TensorDescriptor,
    TensorKind,
    TensorSpec,
)


class TensorRegistry:
    def __init__(self, formats: FormatTable | None = None) -> None:
        self.formats = formats or FormatTable()
        self._specs: dict[int, TensorSpec] = {}
        self._versions: dict[int, int] = {}
        self._owners: dict[int, int] = {}
        self._next_id = 1

    def new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def add(
        self,
        name: str,
        kind: TensorKind,
        shape: tuple[int, ...],
        dtype: DType,
        nbytes: int,
        *,
        fmt: CompressionFormat | None = None,
        n_weights: int = 0,
    ) -> TensorSpec:
        spec = TensorSpec(self.new_id(), name, kind, tuple(shape), dtype, int(nbytes), fmt, n_weights)
        return self.register(spec)

    def register(self, spec: TensorSpec) -> TensorSpec:
        if spec.tensor_id in self._specs:
            raise ValueError(f"tensor {spec.tensor_id} already registered")
        self._specs[spec.tensor_id] = spec
        self._versions[spec.tensor_id] = 0
        self._next_id = max(self._next_id, spec.tensor_id + 1)
        return spec

    def unregister(self, tensor_id: int) -> None:
        self._specs.pop(tensor_id, None)
        self._versions.pop(tensor_id, None)
        self._owners.pop(tensor_id, None)

    def __contains__(self, tensor_id: int) -> bool:
        return tensor_id in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def spec(self, tensor_id: int) -> TensorSpec:
        return self._specs[tensor_id]

    def version(self, tensor_id: int) -> int:
        return self._versions[tensor_id]

    def bump(self, tensor_id: int) -> int:
        if self._specs[tensor_id].policy.immutable:
            raise ValueError(f"tensor {tensor_id} is immutable")
        self._versions[tensor_id] += 1
        return self._versions[tensor_id]

    def owner(self, tensor_id: int) -> int | None:
        return self._owners.get(tensor_id)

    def set_owner(self, tensor_id: int, device: int) -> None:
        self._owners[tensor_id] = device

    def nbytes(self, tensor_id: int, fmt: CompressionFormat | None = None) -> int:
        """Stored size of the tensor, optionally in a non-canonical weight format."""
        spec = self._specs[tensor_id]
        if fmt is None or spec.fmt is None or fmt == spec.fmt:
            return spec.nbytes
        return self.formats.weight_bytes(spec.n_weights, fmt)

    def descriptor(self, tensor_id: int) -> TensorDescriptor:
        owner = self._owners.get(tensor_id, 0xFFFFFFFF)
        return self._specs[tensor_id].descriptor(owner, self._versions[tensor_id])
