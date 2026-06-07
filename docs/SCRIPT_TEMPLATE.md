# Analysis Script Template and Conventions

Standard template for all `analysis_suite/` scripts, plus naming conventions, pre-commit checklist, and data-flow overview.

---

## Template (analysis_suite)

```python
"""Fig{NN}: {Title} — {one-line description}."""
import os, sys, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Suite infrastructure
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STAGE_ORDER, STAGE_COLORS, CACHE_DIR, SESSION_FILTER
from loader import load_staging_manifest, load_session, session_iterator
from utils import get_good_cluster_ids, build_population_tensor, smooth_psth
from plotting import setup_style, save_figure, add_stage_background

# Library imports (when suite wrappers don't cover the need)
from visdetect.analysis.align import get_event_times, align_spikes_to_events
from visdetect.analysis.constants import EVENT_VALID_OUTCOMES, DEFAULT_BIN_SIZE

setup_style()

# ── Cache management ────────────────────────────────────────────
CACHE_FILE = os.path.join(CACHE_DIR, "my_analysis_cache.csv")

def compute_or_load(force=False):
    if os.path.exists(CACHE_FILE) and not force:
        return pd.read_csv(CACHE_FILE)

    manifest = load_staging_manifest(qc_only=True)
    rows = []
    for _, mrow in manifest.iterrows():
        sname = str(mrow["session_name"])
        sess = load_session(sname)
        cluster_ids = get_good_cluster_ids(sess)
        # ... analysis per session ...
        del sess; gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(CACHE_FILE, index=False)
    return df

# ── Main ────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = compute_or_load()

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    # ... panels ...

    save_figure(fig, "fig{NN}_name", "module_name")
```

---

## Naming Conventions

- Script: `{letter}_{descriptive_name}.py` (e.g., `d_post_error_psychometric.py`)
- Figure output: `figures/{module}/fig{NN}_{name}.png`
- Cache: `cache/{descriptive_name}.csv`
- Stats: `figures/{module}/{name}_stats.csv`

---

## Checklist Before Finalizing

- [ ] Imports constants from canonical source (`visdetect.analysis.constants`) — not hardcoded
- [ ] Uses `load_staging_manifest(qc_only=True)` for session selection
- [ ] Uses `get_good_cluster_ids()` or `load_kept_ids()` for unit selection
- [ ] Filters event alignments by `EVENT_VALID_OUTCOMES`
- [ ] Calls `setup_style()` before plotting
- [ ] Uses `save_figure()` for output
- [ ] Cleans up sessions with `del sess; gc.collect()`
- [ ] Color palette matches project conventions (`STAGE_COLORS`, `OUTCOME_COLORS`, etc.)
- [ ] No duplicate implementation of existing utility functions

---

## Data Flow

```
.mat (MATLAB)  →  .pkl (Session dataclass)  →  Analysis
                  ↓
                  Session
                  ├── trials: List[Trial]     (outcome, change_size, RT, change_time)
                  ├── clusters: List[Cluster]  (cluster_id, spike_times)
                  ├── ni_events: Dict          (Baseline_ON, Change_ON, Laser times)
                  ├── good_cluster_ids         (Kilosort "good")
                  └── good_and_stable_ids      (UnitMatch stable)
                                                ↓
                  ┌─────────────────────────────┤
                  ↓                             ↓
           Behavioral analysis           Neural alignment
           (SDT, psychometrics)     (align_spikes_to_events → PETH)
                                                ↓
                                    Population tensor (trials × bins × units)
                                                ↓
                                    ┌───────────┼───────────┐
                                    ↓           ↓           ↓
                                 Decoding   Coding Dir   Heatmaps
                                    ↓           ↓           ↓
                                    └───────────┼───────────┘
                                                ↓
                                    Figure + Stats CSV + Notes
```
