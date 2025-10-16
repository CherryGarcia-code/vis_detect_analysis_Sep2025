README - Development quickstart
================================

This file describes quick developer steps for working with the `vis_detect_analysis` project locally.

1) Create / activate the Conda environment

If you use Conda, create the environment from the provided file and activate it:

```bash
conda env create -f environment.yml -n copilot_ephys
conda activate copilot_ephys
```

2) Install the package in editable mode (recommended during development)

From the repository root:

```bash
pip install -e .
# Or, if you prefer not to install, add the source dir to PYTHONPATH:
export PYTHONPATH=$(pwd)/src
```

3) Run the unit tests (pytest)

Run a selected test file (fast):

```bash
pytest -q tests/test_session.py
```

Or run all tests:

```bash
pytest -q
```

4) Quick Python usage examples

Run a small interactive check (from repo root):

```bash
python - <<'PY'
import sys
sys.path.insert(0, 'src')
from visdetect.io import parse_good_cluster_ids, load_mat_file_to_session
import numpy as np
print('scalar ->', parse_good_cluster_ids(5))
print('array ->', parse_good_cluster_ids(np.array([1,2,3])))
print('bytes ->', parse_good_cluster_ids(b'6'))
# load a mat file (replace path below with a small test file)
# session = load_mat_file_to_session('data/BG_031_260325.mat')
# print(session.subject, len(session.clusters))
PY
```

5) Notes on repository layout and outputs

- Code: `src/visdetect/` (package used by notebooks and scripts)
- Notebooks (exploration): `notebooks/`
- Raw data (not tracked): `data/`
- Derived tables/figures: `table_output/`, `png_output/`, `results/` — these directories are listed in `.gitignore` to avoid flooding Git with generated files.

6) CI / contributions

If you add tests, create small, focused tests for numeric code (PETHs, alignment, parsing). Add a GitHub Actions workflow to run `pytest` on PRs.

If anything above fails on your machine, tell me the exact error and I'll help adapt commands for your environment.

---
Created to help onboarding and development. Edit as needed.
