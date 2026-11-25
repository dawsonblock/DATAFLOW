"""Real model files: reading GGUF headers (shapes, sizes, fingerprints). No tensor data yet."""

from .gguf import (
    GGML_TYPES,
    GGUFError,
    GGUFFile,
    TensorInfo,
    check_against,
    compare_shapes,
    read_gguf,
    spec_from_gguf,
    write_minimal_gguf,
)

__all__ = [
    "GGML_TYPES",
    "GGUFError",
    "GGUFFile",
    "TensorInfo",
    "check_against",
    "compare_shapes",
    "read_gguf",
    "spec_from_gguf",
    "write_minimal_gguf",
]
