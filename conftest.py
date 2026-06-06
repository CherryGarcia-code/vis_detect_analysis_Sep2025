"""Ensure pytest imports `visdetect` from this repo's ``src/`` directory.

Prepends the repo's ``src/`` to ``sys.path`` so tests run against the code in
this checkout rather than any editable install elsewhere on the machine.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
