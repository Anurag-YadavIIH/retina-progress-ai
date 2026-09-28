# scripts/verify_setup.py
# Run this script to confirm every dependency is installed correctly.
# Usage: python scripts/verify_setup.py

import os
import sys

# Make the repo root importable so we can use src.utils regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check(name, import_str, extra=None):
    """Try to import a package and print its version."""
    try:
        mod = __import__(import_str)
        version = getattr(mod, '__version__', 'installed')
        print(f"  [OK] {name:<30} {version}")
        if extra:
            extra(mod)
    except ImportError as e:
        print(f"  [FAIL] {name:<30} NOT FOUND. run: pip install {import_str}")

print("\n=== RetinaProgress AI: Environment Check ===\n")

print("Python:")
print(f"  [OK] Python {sys.version}\n")

print("Deep learning:")
check("PyTorch",        "torch",       lambda m: print(f"         CUDA available: {m.cuda.is_available()}"))
check("torchvision",    "torchvision")

print("\nComputer vision:")
check("OpenCV",         "cv2")
check("Pillow",         "PIL")
check("scikit-image",   "skimage")
check("albumentations", "albumentations")

print("\nScientific computing:")
check("NumPy",          "numpy")
check("SciPy",          "scipy")
check("scikit-learn",   "sklearn")
check("pandas",         "pandas")

print("\nVisualization:")
check("matplotlib",     "matplotlib")
check("seaborn",        "seaborn")

print("\nModel libraries:")
check("segmentation-models-pytorch", "segmentation_models_pytorch")
check("timm",           "timm")

print("\nUtilities:")
check("tqdm",           "tqdm")
check("tensorboard",    "tensorboard")
check("pyyaml",         "yaml")

print("\nReport & API:")
check("reportlab",      "reportlab")
check("FastAPI",        "fastapi")
check("uvicorn",        "uvicorn")

print("\nResolved compute device:")
try:
    # Show what runtime.device ("auto") resolves to on this machine.
    from src.utils import get_device, load_config
    cfg = load_config()
    preference = cfg.get("runtime", {}).get("device", "auto")
    device = get_device(preference)
    print(f"  [OK] preference '{preference}' -> {device}")
except Exception as e:  # noqa: BLE001 - report any resolution problem
    print(f"  [FAIL] could not resolve device: {e!r}")

print("\n=== Check complete ===\n")