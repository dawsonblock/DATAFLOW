"""Tensor data of a GGUF file: memory-mapped, dequantised on request.

``GGUFTensors`` maps the file read-only and returns each tensor as float32 in NumPy order (the
GGUF shape reversed: a weight matrix comes back as ``(out_features, in_features)``).  Expert
tensors of mixture-of-experts models can be read one expert at a time without touching the
others.
"""

from __future__ import annotations

import os

import numpy as np

from .gguf import GGML_TYPES, GGUFError, GGUFFile, TensorInfo, read_gguf
from .quants import dequantize


class GGUFTensors:
    def __init__(self, source: str | os.PathLike | GGUFFile) -> None:
        self.file = source if isinstance(source, GGUFFile) else read_gguf(source)
        self._map = np.memmap(self.file.path, dtype=np.uint8, mode="r")
        self.bytes_read = 0

    def __contains__(self, name: str) -> bool:
        return self.file.tensor(name) is not None

    def info(self, name: str) -> TensorInfo:
        t = self.file.tensor(name)
        if t is None:
            raise KeyError(f"tensor {name!r} is not in {os.path.basename(self.file.path)}")
        return t

    def raw(self, name: str) -> np.ndarray:
        t = self.info(name)
        start = self.file.data_offset + t.offset
        return self._map[start:start + t.nbytes]

    def shape(self, name: str) -> tuple[int, ...]:
        """NumPy shape (GGUF dimensions reversed)."""
        return tuple(reversed(self.info(name).shape))

    def load(self, name: str) -> np.ndarray:
        t = self.info(name)
        data = self.raw(name)
        self.bytes_read += data.size
        return dequantize(data, t.ggml_type, t.n_elements).reshape(self.shape(name))

    def slice_nbytes(self, name: str) -> int:
        """Stored bytes of one slice of the outermost dimension (one expert of an expert tensor)."""
        t = self.info(name)
        return t.nbytes // self.shape(name)[0]

    def load_rows(self, name: str, rows) -> np.ndarray:
        """Rows of a matrix (e.g. token embeddings for a few ids), reading only those rows."""
        t = self.info(name)
        n_rows, n_cols = self.shape(name)
        _, block, size = GGML_TYPES[t.ggml_type]
        if n_cols % block:
            raise GGUFError(f"{name}: a row is not a whole number of blocks")
        row_bytes = n_cols // block * size
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if rows.size and (rows.min() < 0 or rows.max() >= n_rows):
            raise IndexError(f"{name}: row outside 0..{n_rows - 1}")
        start = self.file.data_offset + t.offset
        view = self._map[start:start + t.nbytes].reshape(n_rows, row_bytes)
        data = view[rows]
        self.bytes_read += data.size
        return dequantize(data.reshape(-1), t.ggml_type, rows.size * n_cols).reshape(rows.size, n_cols)

    def load_slice(self, name: str, index: int) -> np.ndarray:
        """Slice ``index`` of the outermost dimension (one expert of an expert tensor)."""
        t = self.info(name)
        shape = self.shape(name)
        if not 0 <= index < shape[0]:
            raise IndexError(f"{name}: index {index} outside 0..{shape[0] - 1}")
        _, block, size = GGML_TYPES[t.ggml_type]
        per = t.n_elements // shape[0]
        if per % block:
            raise GGUFError(f"{name}: a slice is not a whole number of blocks")
        nbytes = per // block * size
        start = self.file.data_offset + t.offset + index * nbytes
        data = self._map[start:start + nbytes]
        self.bytes_read += data.size
        return dequantize(data, t.ggml_type, per).reshape(shape[1:])
