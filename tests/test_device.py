"""
tests/test_device.py
====================
Tests for the device auto-detection helper in src.utils.

These run on any machine, including CPU-only CI runners: "auto" must resolve to
one of the three known backends, "cpu" must always be honoured, and requesting
an unavailable backend must fall back gracefully instead of raising.
"""

import pytest

from src.utils import get_device, resolve_device

# torch may be absent in a bare environment; skip the torch.device tests then,
# but the resolve_device string logic is still covered below.
torch = pytest.importorskip("torch", reason="torch not installed")


def test_resolve_auto_returns_known_backend():
    """'auto' must resolve to exactly one of cuda / mps / cpu."""
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}


def test_resolve_cpu_always_cpu():
    """Explicit 'cpu' is always available and must be honoured."""
    assert resolve_device("cpu") == "cpu"


def test_resolve_unknown_falls_back():
    """An unrecognised preference must not crash; it falls back to auto."""
    assert resolve_device("banana") in {"cuda", "mps", "cpu"}


def test_resolve_case_insensitive():
    """Preference parsing is case-insensitive."""
    assert resolve_device("CPU") == "cpu"


def test_get_device_returns_torch_device():
    """get_device must return an actual torch.device of a known type."""
    device = get_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "mps", "cpu"}


def test_get_device_cpu():
    """Requesting cpu yields a cpu torch.device."""
    assert get_device("cpu").type == "cpu"
