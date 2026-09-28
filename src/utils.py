"""
src/utils.py
============
Small shared helpers used across the RetinaProgress AI codebase.

Currently provides:
  * load_config  - read the YAML config into a plain dict.
  * get_device   - resolve the compute device, with "auto" detection
                   (CUDA if present, else Apple Silicon MPS, else CPU).

Keeping these in one place means training, inference and the API all agree on
where config lives and how the device is chosen, instead of each re-detecting.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

# Repo root = one directory above this file's directory (src/ -> repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default config location, resolved relative to the repo root so the helper
# works no matter what the current working directory is.
DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs", "config.yaml")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the project YAML config into a dictionary.

    Parameters
    ----------
    path : str, optional
        Path to a YAML config file. When omitted, the bundled
        ``configs/config.yaml`` is used (resolved relative to the repo root, so
        this works from any working directory).

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(preference: str = "auto") -> str:
    """Resolve a device preference string to a concrete device string.

    This is the torch-free core of the logic so it can be unit-tested and
    reused without importing torch. ``get_device`` wraps it and returns an
    actual ``torch.device``.

    Resolution order for "auto":
        1. CUDA (NVIDIA GPU) if available.
        2. MPS (Apple Silicon) if available.
        3. CPU otherwise.

    An explicit preference ("cpu", "cuda", "mps") is honoured as long as it is
    actually available; if it is not, the function falls back to "auto"
    detection rather than crashing at an awkward point later in training.

    Parameters
    ----------
    preference : str
        One of "auto", "cpu", "cuda", "mps" (case-insensitive).

    Returns
    -------
    str
        A concrete device string: "cuda", "mps" or "cpu".
    """
    pref = (preference or "auto").strip().lower()

    # Import torch lazily so importing this helper (e.g. in a config-only test)
    # does not require torch, and so the import cost is only paid when needed.
    try:
        import torch
    except ImportError:
        # No torch installed at all -> the only safe answer is CPU.
        return "cpu"

    cuda_ok = torch.cuda.is_available()
    # torch.backends.mps may not exist on older builds; guard with getattr.
    mps_backend = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps_backend) and mps_backend.is_available()

    if pref == "cuda":
        return "cuda" if cuda_ok else _auto_device(cuda_ok, mps_ok)
    if pref == "mps":
        return "mps" if mps_ok else _auto_device(cuda_ok, mps_ok)
    if pref == "cpu":
        return "cpu"
    # "auto" or anything unrecognised -> best available.
    return _auto_device(cuda_ok, mps_ok)


def _auto_device(cuda_ok: bool, mps_ok: bool) -> str:
    """Pick the best available device given availability flags."""
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"


def get_device(preference: str = "auto"):
    """Return a ``torch.device`` for the requested preference.

    Thin wrapper over :func:`resolve_device` that constructs the actual
    ``torch.device``. Training and inference code should call this rather than
    hard-coding "cuda".

    Parameters
    ----------
    preference : str
        One of "auto", "cpu", "cuda", "mps".

    Returns
    -------
    torch.device
        The resolved device.
    """
    import torch  # imported here so the module has no hard torch dependency

    return torch.device(resolve_device(preference))
