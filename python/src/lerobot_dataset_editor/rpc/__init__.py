"""Framed JSON-RPC backend used by the Electron desktop shell."""

from .server import PROTOCOL_VERSION, run_server

__all__ = ["PROTOCOL_VERSION", "run_server"]
