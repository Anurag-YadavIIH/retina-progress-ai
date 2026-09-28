"""Root conftest.

Ensures the repository root is on ``sys.path`` so tests can ``import src.*``
regardless of how pytest is invoked. pytest also honours the ``pythonpath``
option in ``pytest.ini``; this file is a belt-and-braces fallback for older
pytest versions that predate that option.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
