"""Test configuration — blocks outbound network and sets hypothesis defaults.

Safety: no test can accidentally trigger real downloads (prevents the 170GB incident).
Performance: hypothesis examples capped at 50 for faster CI runs.
"""

import socket
from unittest.mock import patch

import pytest
from hypothesis import settings

# ---------------------------------------------------------------------------
# Block all outbound network in tests
# ---------------------------------------------------------------------------

_original_connect = socket.socket.connect


def _blocked_connect(self, address):
    """Block all socket connections in tests to prevent accidental downloads."""
    host = address[0] if isinstance(address, tuple) else address
    # Allow localhost connections (for SQLite, Gradio testing, etc.)
    if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
        return _original_connect(self, address)
    raise RuntimeError(
        f"Tests attempted outbound network connection to {address}. "
        f"All network calls must be mocked in tests."
    )


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Autouse fixture that blocks all outbound network connections."""
    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)


# ---------------------------------------------------------------------------
# Hypothesis settings — reduced examples for speed
# ---------------------------------------------------------------------------

settings.register_profile(
    "ci",
    max_examples=50,
    deadline=None,
)
settings.register_profile(
    "default",
    max_examples=50,
    deadline=None,
)
settings.load_profile("default")
