"""
chroma_client.py

ChromaDB client — PersistentClient (local) or HttpClient (remote).

Pinned to chromadb 0.4.x which uses Python chroma-hnswlib (C++ bindings).
chromadb 1.x switched to Rust HNSW bindings that SIGSEGV on macOS 26
(Darwin 25.x) with KERN_INVALID_ADDRESS at 0x44.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_client = None

_EMBEDDED_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "chroma"
)


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def get_chroma_client():
    """Return the singleton ChromaDB client.

    Uses PersistentClient (embedded) unless CHROMADB_HOST is set,
    in which case it connects via HttpClient.
    """
    global _client
    if _client is not None:
        return _client

    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError(
            "ChromaDB is not installed. Run: pip install 'chromadb==0.4.24'"
        ) from e

    host = os.getenv("CHROMADB_HOST", "")

    if host:
        port = int(os.getenv("CHROMADB_PORT", "8100"))
        if not _port_open(host, port):
            raise RuntimeError(
                f"ChromaDB not reachable at {host}:{port}. "
                f"Check CHROMADB_HOST/CHROMADB_PORT or unset CHROMADB_HOST."
            )
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()
        logger.info(f"ChromaDB HTTP connected: {host}:{port}")
    else:
        Path(_EMBEDDED_PATH).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=_EMBEDDED_PATH)
        logger.info(f"ChromaDB persistent client: {_EMBEDDED_PATH}")

    _client = client
    return _client


def reset_client() -> None:
    """Reset the singleton (e.g. after a config change)."""
    global _client
    _client = None
