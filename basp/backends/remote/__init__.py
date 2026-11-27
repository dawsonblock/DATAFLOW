"""Remote backend (a device in another process or machine): not implemented.  Declares
itself unavailable."""

from basp.backends.execution import UnavailableBackend


class RemoteBackend(UnavailableBackend):
    kind = "remote"

    def __init__(self) -> None:
        super().__init__("not implemented in v0.4 (no transport exists)")


__all__ = ["RemoteBackend"]
