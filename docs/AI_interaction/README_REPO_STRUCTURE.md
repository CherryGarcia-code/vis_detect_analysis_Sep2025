# vis_detect_analysis_Sep2025: Project Structure and Import Guide

## Key Points
- All core code is in `src/visdetect/` (not `src/` directly).
- Use `from visdetect...` for all imports in scripts and notebooks.
- The repo root should be added to `sys.path` for all scripts/notebooks (see below).
- Compatibility shims exist for legacy pickle loading: `src/visdetect/session.py`, `src/visdetect/io.py`.
- A stub for `src/unit_tracking.py` is provided for legacy scripts.

## How to Import
**In scripts and notebooks:**
```python
import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from visdetect.core.session import Session, Trial, Cluster
from visdetect.io import load_mat_file_to_session
```

## Directory Structure
- `src/visdetect/` — main package
  - `core/` — dataclasses, IO, QC
  - `analysis/` — analysis modules (tuning, responsiveness, etc.)
  - `utils/` — utility code
  - `session.py`, `io.py` — compatibility shims for pickle loading
- `scripts/` — analysis and batch scripts (now use `visdetect` imports)
- `notebooks/` — Jupyter notebooks (add repo root to `sys.path`)
- `tests/` — unit tests

## Pickle Compatibility
- Pickles created with older helpers may require the repo root in `sys.path` and the compatibility shims in `src/visdetect/`.
- If you see `ModuleNotFoundError: No module named 'src'`, ensure the above and re-save pickles with the new structure if possible.

## Maintenance Checklist
- Always use `visdetect` for imports, not `src`.
- Add the repo root to `sys.path` in all entry points.
- Keep compatibility shims for legacy pickle support.
- Update or remove the `src/unit_tracking.py` stub as you migrate code.
