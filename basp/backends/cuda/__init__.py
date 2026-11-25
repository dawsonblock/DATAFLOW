"""CUDA backend: not implemented.  Declares itself unavailable; no CUDA inference exists in
this release (roadmap: native C++/CUDA engine after the CPU runtime is verified)."""

from basp.backends.execution import UnavailableBackend


class CUDABackend(UnavailableBackend):
    kind = "cuda"

    def __init__(self) -> None:
        super().__init__("not implemented in v0.4 (no CUDA kernels exist; nothing runs on a GPU)")


__all__ = ["CUDABackend"]
