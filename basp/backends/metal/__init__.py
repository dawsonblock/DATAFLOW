"""Metal backend: not implemented.  Declares itself unavailable; no Metal inference exists in
this release."""

from basp.backends.execution import UnavailableBackend


class MetalBackend(UnavailableBackend):
    kind = "metal"

    def __init__(self) -> None:
        super().__init__("not implemented in v0.4 (no Metal kernels exist; nothing runs on Apple GPUs)")


__all__ = ["MetalBackend"]
