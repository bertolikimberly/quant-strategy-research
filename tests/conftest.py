"""Shared pytest fixtures + import-path setup.

Adding the project root to sys.path here means tests can `from src...`
without the user having to `pip install -e .` first. For a course
project that trade-off (one line of test boilerplate vs a setup.py)
is the right call.
"""

from __future__ import annotations

import sys
from pathlib import Path

# tests/conftest.py lives one directory deep, so parent = repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
