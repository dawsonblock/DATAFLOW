"""GGML backend (llama.cpp kernels through a native binding): not implemented.  Declares
itself unavailable.  The NumPy runtime reads GGUF files and dequantises GGML formats itself;
it does not call ggml."""

from basp.backends.execution import UnavailableBackend


class GGMLBackend(UnavailableBackend):
    kind = "ggml"

    def __init__(self) -> None:
        super().__init__("not implemented in v0.4 (no native ggml binding is built)")


__all__ = ["GGMLBackend"]
