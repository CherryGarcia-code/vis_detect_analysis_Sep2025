from __future__ import annotations
import sys
from pathlib import Path

# Ensure repo root and scripts are on path
_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from make_kept_gallery import main as _main  # type: ignore

if __name__ == "__main__":
    raise SystemExit(_main())
