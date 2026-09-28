"""
tests/test_config.py
====================
Validates that configs/config.yaml loads and contains the keys the training
and inference code depends on. This is a cheap guardrail: a typo or a bad
indent in the YAML fails CI immediately rather than deep inside a training run.
"""

from src.utils import load_config


def test_config_loads():
    """The bundled config must parse into a dict."""
    cfg = load_config()
    assert isinstance(cfg, dict)


def test_top_level_sections_present():
    """The sections consumed elsewhere in the codebase must exist."""
    cfg = load_config()
    for section in ("project", "runtime", "paths",
                    "vessel_seg", "disc_seg", "lesion_seg", "biomarkers"):
        assert section in cfg, f"missing config section: {section}"


def test_runtime_device_is_auto():
    """Device must be auto-detected, never hard-coded to a specific backend."""
    cfg = load_config()
    assert cfg["runtime"]["device"] == "auto"


def test_no_hardcoded_cuda_in_vessel_seg():
    """The old hard-coded vessel_seg.device: cuda must be gone."""
    cfg = load_config()
    assert "device" not in cfg["vessel_seg"]


def test_task_hyperparameters_have_expected_types():
    """Image sizes are ints and learning rates are positive floats."""
    cfg = load_config()
    for task in ("vessel_seg", "disc_seg", "lesion_seg"):
        assert isinstance(cfg[task]["image_size"], int)
        assert isinstance(cfg[task]["batch_size"], int)
        assert float(cfg[task]["learning_rate"]) > 0
