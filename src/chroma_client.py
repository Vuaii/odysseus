"""
chroma_client.py

ChromaDB client — local-server mode by default, HTTP when CHROMADB_HOST is set.

Local-server mode spawns `chroma run` as a subprocess and connects via
HttpClient on 127.0.0.1. This avoids the chromadb_rust_bindings SIGSEGV
that PersistentClient triggers on macOS 26 (Darwin 25.x / Tahoe) with
chromadb 1.5.x — the Rust HNSW bindings NULL-deref at offset 0x44 on
that OS. The data directory is the same as before (data/chroma), so
no migration is needed.
"""

import atexit
import os
import shutil
import socket
import subprocess
import sys
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_client = None
_server_proc: subprocess.Popen | None = None

_CONNECT_TIMEOUT = float(os.getenv("CHROMADB_CONNECT_TIMEOUT", "2.0"))
_SERVER_STARTUP_TIMEOUT = float(os.getenv("CHROMADB_STARTUP_TIMEOUT", "30.0"))
_LOCAL_HOST = "127.0.0.1"

# Storage path — unchanged from the original embedded layout.
_EMBEDDED_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "chroma"
)


def _port_open(host: str, port: int, timeout: float = None) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout or _CONNECT_TIMEOUT):
            return True
    except OSError:
        return False


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((_LOCAL_HOST, 0))
        return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    interval = 0.2
    while time.monotonic() < deadline:
        if _port_open(host, port, timeout=0.5):
            return True
        time.sleep(interval)
        interval = min(interval * 1.5, 2.0)
    return False


def _find_chroma_bin() -> str:
    # Prefer the binary next to sys.executable (handles venv / brew setups).
    candidate = Path(sys.executable).parent / "chroma"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("chroma")
    if found:
        return found
    raise RuntimeError(
        "'chroma' CLI not found. Is chromadb installed? Run: pip install chromadb"
    )


def _start_local_server(data_path: str) -> tuple[subprocess.Popen, int]:
    """Spawn `chroma run` and return (proc, port)."""
    port_override = os.getenv("CHROMADB_LOCAL_PORT", "")
    port = int(port_override) if port_override else _find_free_port()

    os.makedirs(data_path, exist_ok=True)

    chroma_bin = _find_chroma_bin()
    cmd = [chroma_bin, "run", "--path", data_path, "--host", _LOCAL_HOST, "--port", str(port)]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # don't forward Ctrl-C to the server
    )

    if not _wait_for_server(_LOCAL_HOST, port, _SERVER_STARTUP_TIMEOUT):
        proc.terminate()
        raise RuntimeError(
            f"Local ChromaDB server failed to start within {_SERVER_STARTUP_TIMEOUT}s "
            f"(port {port}, pid {proc.pid}). "
            f"Check that chromadb is properly installed."
        )

    logger.info(f"ChromaDB local server started: {_LOCAL_HOST}:{port} (pid={proc.pid})")
    return proc, port


def _stop_local_server() -> None:
    global _server_proc
    if _server_proc is not None and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        logger.info("ChromaDB local server stopped")
    _server_proc = None


atexit.register(_stop_local_server)


def get_chroma_client():
    """Return the singleton ChromaDB client.

    Uses local-server mode (spawns `chroma run`) unless CHROMADB_HOST is
    explicitly set, in which case it connects to that external server via HTTP.
    """
    global _client, _server_proc
    if _client is not None:
        return _client

    try:
        import chromadb
    except ImportError as e:
        raise RuntimeError(
            "ChromaDB is not installed. Run: pip install chromadb"
        ) from e

    host = os.getenv("CHROMADB_HOST", "")

    if host:
        # HTTP mode — explicit host configured (Docker / remote server)
        port = int(os.getenv("CHROMADB_PORT", "8100"))
        if not _port_open(host, port):
            raise RuntimeError(
                f"ChromaDB is not reachable at {host}:{port}. "
                f"Check CHROMADB_HOST / CHROMADB_PORT or unset CHROMADB_HOST "
                f"to use local-server mode."
            )
        client = chromadb.HttpClient(host=host, port=port)
        client.heartbeat()
        logger.info(f"ChromaDB HTTP connected: {host}:{port}")
    else:
        # Local-server mode: spawn `chroma run` subprocess so the Rust
        # HNSW bindings run in an isolated server process rather than being
        # dlopen'd into this process (which causes SIGSEGV on macOS 26).
        proc, port = _start_local_server(_EMBEDDED_PATH)
        _server_proc = proc
        client = chromadb.HttpClient(host=_LOCAL_HOST, port=port)
        client.heartbeat()
        logger.info(f"ChromaDB local-server mode: {_LOCAL_HOST}:{port}")

    _client = client
    return _client


def reset_client() -> None:
    """Reset the singleton — stops local server too (e.g. after config change)."""
    global _client
    _stop_local_server()
    _client = None
