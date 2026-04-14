# TF Manual Labeling — Operations Manual

This manual documents how to run the TF cell manual labeling system, from pre-caching through labeling to eventual ML classifier training.

---

## Overview

The labeling system has three components:

| Component | File | Purpose |
|-----------|------|---------|
| **Library** | `src/visdetect/analysis/tf_labeling.py` | Label I/O, priority queue, data loading |
| **GUI** | `scripts/tf_labeling/run_labeling_gui.py` | Interactive matplotlib viewer for labeling |
| **Raster cache** | `scripts/tf_labeling/precache_rasters.py` | Pre-compute pulse-aligned spike rasters |

Data files:

| File | Location | Purpose |
|------|----------|---------|
| **Classification CSV** | `analysis_suite/cache/tf_cell_classification.csv` | Algorithmic tier labels (input) |
| **Manual labels** | `data/labels/tf_manual_labels.csv` | Human-assigned labels (output, grows incrementally) |
| **NPZ trace cache** | `data/cache/tf_traces/BG_046/` | Z-scored fast/slow traces per session |
| **Raster cache** | `data/cache/tf_raster_cache/` | Per-unit pulse-aligned spike rasters |

---

## Prerequisites

Before labeling, these must already exist:

1. **Session .pkl files** in `data/pkls/BG_046/` (from Step 1 of `RUNNING.md`)
2. **Staging manifest** at `data/BG_046_staging_manifest.csv` (from Step 2)
3. **NPZ trace caches** in `data/cache/tf_traces/BG_046/` (from `08_tf_pulse/a_tf_responsiveness.py` or equivalent)
4. **Classification CSV** at `analysis_suite/cache/tf_cell_classification.csv` (from `08_tf_pulse/g_tf_cell_classifier.py`)

Verify prerequisites:

```bash
py -c "
from visdetect.analysis.tf_labeling import CLASSIFICATION_CSV, RASTER_CACHE_DIR
import os, pandas as pd
csv = pd.read_csv(CLASSIFICATION_CSV)
print(f'Classification CSV: {len(csv)} units, {csv[\"session_name\"].nunique()} sessions')
print(f'Tier distribution:')
print(csv['tier'].value_counts().to_string())
"
```

---

## Step 1: Pre-cache Rasters (One-Time)

The GUI displays spike rasters aligned to each TF pulse. Pre-caching this data avoids loading ~100 MB session files during interactive labeling.

### Quick test (2 sessions)

```bash
py scripts/tf_labeling/precache_rasters.py --max-sessions 2
```

### Full run (all sessions)

```bash
py scripts/tf_labeling/precache_rasters.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--force` | off | Recompute even if cache files already exist |
| `--max-sessions N` | all | Limit to first N sessions (for testing) |
| `--n-workers N` | `min(cpu_count, 8)` | Parallel workers per session |

### What happens

For each session:
1. Loads the session .pkl (serial, I/O bound)
2. Collects fast/slow TF pulse times from the baseline stochastic TF trajectory
3. For each unit, extracts spike times in a (-0.4, 0.5)s window around every pulse
4. Saves per-unit NPZ files to `data/cache/tf_raster_cache/`

Uses `ProcessPoolExecutor` to parallelize per-unit raster extraction within each session (matching the pattern in `g_tf_cell_classifier.py`).

### Expected output

```
Found 4725 units across 25 sessions
Workers: 8
[1/25] Session 27062025: 189 units... cached 189
[2/25] Session 1072025: 156 units... cached 156
...
Done: 4725 rasters cached in 847s
```

### Timing estimate

- ~30-60s per session (dominated by pkl loading + pulse collection)
- ~15-25 minutes total for all 25 sessions
- After first run, re-runs complete in seconds (skips existing caches)

### Note

The GUI **works without rasters** — those panels simply show a placeholder message. You can start labeling immediately using just the z-scored traces while the raster cache builds in the background.

---

## Step 2: Launch the Labeling GUI

```bash
py scripts/tf_labeling/run_labeling_gui.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--reviewer NAME` | `BG` | Reviewer name (saved in the label CSV) |
| `--include-labeled` | off | Include already-labeled units (re-review mode) |
| `--start-idx N` | 0 | Start at index N in the queue |

### GUI Layout

```
+--------------------------------------+-----------------+
|  Z-scored traces (fast=blue, slow=red)|   METADATA      |
|  with SEM shading [d=toggle detrend] |   Session       |
|                                       |   Cluster ID    |
+---------------------------------------+   Algo tier     |
|  Splitter test (Fast vs -Slow mirror)|   Sub-type      |
|                                       |   Peak/AUC/etc  |
+------------------+--------------------+                 |
| Fast raster      | Slow raster        |   CURRENT LABEL |
| (black ticks)    | (black ticks)      |   Confidence    |
| + PSTH strip     | + PSTH strip       |                 |
+------------------+--------------------+                 |
| Fast spike       | Slow spike         |   Shortcuts     |
| density heatmap  | density heatmap    |                 |
+------------------+--------------------+-----------------+
  Title bar: "Unit 142/4725 | Labeled: 58 | Splitter: 12  Uni: 21 ..."
```

### Detrended View (press `d`)

The standard z-scored traces use a **grand-average** baseline (mean/std across all pulses). If a neuron has slow firing rate drifts (state changes, arousal ramps), these can contaminate the pulse-triggered average and produce spurious "responses."

Pressing `d` toggles **per-pulse baseline correction**: each pulse's own local baseline ([-200, -50] ms before that pulse) is subtracted before averaging. This removes slow drifts and reveals the true stimulus-locked response. The title turns **red** when detrended view is active.

**When to use it:**
- A unit shows a sloping baseline in the pre-window → press `d` to check if the post-pulse "response" survives correction
- A unit has a high `Trend` ratio in the features panel → the detrended view shows what remains after drift removal
- Borderline cases: if the response is only visible in one view but not the other, it suggests the effect is drift-driven

### Keyboard Shortcuts

**Tier assignment** (first keypress):

| Key | Action |
|-----|--------|
| `1` | Select Tier 1 (Splitter) — then pick sub-type |
| `2` | Select Tier 2 (Unilateral) — then pick sub-type |
| `3` | Select Tier 3 (Omni) — then pick sub-type |
| `0` | Non-responsive (saves immediately, no sub-type needed) |

**Sub-type** (second keypress, after tier):

| Tier | Key | Sub-type |
|------|-----|----------|
| Splitter | `f` | Fast+/Slow- |
| Splitter | `s` | Slow+/Fast- |
| Unilateral | `f` | Fast+ |
| Unilateral | `F` (shift) | Fast- |
| Unilateral | `s` | Slow+ |
| Unilateral | `S` (shift) | Slow- |
| Omni | `+` or `=` | Both+ |
| Omni | `-` | Both- |

**Navigation and controls**:

| Key | Action |
|-----|--------|
| `j` or `Right` | Next unit |
| `k` or `Left` | Previous unit |
| `d` | Toggle detrended view (per-pulse baseline correction) |
| `c` | Cycle confidence: high -> medium -> low -> high |
| `n` | Add a note (types in the terminal, press Enter to confirm) |
| `h` | Toggle help overlay |
| `Escape` | Cancel pending tier selection |
| `q` | Quit |

### Labeling workflow

1. The GUI shows the highest-priority unlabeled unit
2. Inspect the z-scored traces (top panel): look for consistent post-pulse responses
3. Check the rasters (bottom panels): look for trial-to-trial reliability
4. Look at the metadata sidebar: algo tier, p-values, mirror score, pulse counts
5. Press a tier key (`1`/`2`/`3`/`0`), then a sub-type key
6. The label auto-saves and the GUI advances to the next unit
7. Use `j`/`k` to go back and re-check previous units

### Tips for efficient labeling

- **Trust the rasters**: If trials are consistent, the z-score is real. If the z-trace looks big but the raster is noisy, likely a false positive.
- **Mirror symmetry**: Splitters should show clear opposite responses to fast vs slow. If both traces go the same direction, it's Omni, not Splitter.
- **Trend-excluded units**: These appear early in the queue. Look at whether the pre-pulse baseline is truly flat — if there's a slow drift, the trend filter was right to exclude them.
- **Low pulse counts**: Units with <50 pulses are less reliable. Mark confidence as `medium` or `low`.
- **Use `c` for confidence**: `high` = clearly this tier, `medium` = reasonable but some ambiguity, `low` = borderline call.
- **Use `n` for notes**: Record why you made a particular call on ambiguous units (helps later analysis).

---

## Step 3: Check Progress

### From Python

```bash
py -c "
from visdetect.analysis.tf_labeling import get_label_stats, load_labels
stats = get_label_stats()
print(f'Total labeled: {stats[\"total\"]}')
for tier, count in stats.get('by_tier', {}).items():
    print(f'  {tier}: {count}')
for conf, count in stats.get('by_confidence', {}).items():
    print(f'  Confidence {conf}: {count}')
"
```

### View the raw labels file

```bash
py -c "
import pandas as pd
df = pd.read_csv('data/labels/tf_manual_labels.csv')
print(df.to_string())
"
```

### Agreement with algorithmic labels

```bash
py -c "
import pandas as pd
df = pd.read_csv('data/labels/tf_manual_labels.csv')
agree = (df['manual_tier'] == df['algo_tier']).mean()
print(f'Agreement: {agree:.1%}')
print(pd.crosstab(df['algo_tier'], df['manual_tier'], margins=True))
"
```

---

## Label File Format

The labels CSV at `data/labels/tf_manual_labels.csv` has these columns:

| Column | Type | Description |
|--------|------|-------------|
| `session_name` | int | Session identifier (DDMMYYYY format) |
| `cluster_id` | int | Unit identifier |
| `manual_tier` | str | Human label: `Tier 1 (Splitter)`, `Tier 2 (Unilateral)`, `Tier 3 (Omni)`, `Non-responsive` |
| `manual_sub_type` | str | Fine label: `Fast+/Slow-`, `Slow+/Fast-`, `Fast+`, `Fast-`, `Slow+`, `Slow-`, `Both+`, `Both-`, `None` |
| `confidence` | str | `high`, `medium`, or `low` |
| `notes` | str | Free-text notes for ambiguous cases |
| `algo_tier` | str | Algorithmic tier at time of review |
| `algo_sub_type` | str | Algorithmic sub-type at time of review |
| `reviewer` | str | Who labeled this unit |
| `timestamp` | str | ISO 8601 UTC datetime of label assignment |

Labels auto-save after every assignment (atomic write via temp file + rename). Re-labeling the same (session_name, cluster_id) pair updates in place rather than duplicating.

---

## Queue Priority Logic

Units are presented in priority order (highest first). The priority scoring:

| Priority band | Score range | Which units |
|---------------|------------|-------------|
| **Trend-excluded** | ~900+ | Algorithmically excluded by trend filter — may be wrong |
| **Responsive + borderline p** | ~700-900 | Classified responsive but p-values near alpha boundaries |
| **Responsive, clear** | ~550-700 | Confidently classified responsive (verify true positives) |
| **High-z non-responsive** | ~300-500 | z > 1.5 but failed significance — potential false negatives |
| **Low-z non-responsive** | 0-150 | Clearly inactive — true negatives, lowest priority |

You can skip units with `j` and return with `k` at any time.

---

## Re-Review Mode

To re-review previously labeled units (e.g., after changing your mind on criteria):

```bash
py scripts/tf_labeling/run_labeling_gui.py --include-labeled
```

This includes all units in the queue (labeled and unlabeled), still sorted by priority. Already-labeled units show their existing label in the info panel.

---

## Troubleshooting

### "No NPZ trace data" on main panels

The NPZ trace cache for that session is missing. Run:

```bash
cd analysis_suite && py 08_tf_pulse/a_tf_responsiveness.py
```

Or check that `data/cache/tf_traces/BG_046/` has NPZ files for all sessions.

### "No raster cache — run precache_rasters.py" on raster panels

Run Step 1 above. The GUI still works for labeling — just without raster visualization.

### GUI doesn't open / "FigureCanvasAgg is non-interactive"

**Cause**: Several modules in the `visdetect` package (`core/qc.py`, `analysis/tf_pulse.py`, `analysis/unit_selection.py`) call `matplotlib.use("Agg")` at module level. Because Python's package `__init__.py` imports trigger these automatically, the Agg backend can override TkAgg even though the GUI sets it first.

**Built-in fix**: The GUI already handles this by re-asserting `TkAgg` after all imports:
```python
matplotlib.use("TkAgg", force=True)
plt.switch_backend("TkAgg")
```
If you still see this error, verify tkinter is available:
```bash
py -c "import tkinter; print('Tk available')"
```
If Tk is not available, install it or use a different Python distribution.

### Slow navigation between units

Trace loading is ~30ms per unit (from NPZ). If navigation feels slow, the bottleneck is likely matplotlib redrawing. Close other matplotlib windows and avoid resizing during labeling.

### Labels file looks corrupted

The save uses atomic write (write to `.tmp`, then `os.replace`). If the process was killed mid-write, check for `tf_manual_labels.csv.tmp` — if it exists and is valid, rename it to `tf_manual_labels.csv`.

---

## Future Steps (Not Yet Implemented)

### Phase 2: Agreement Analysis (after ~300 labels)

- Confusion matrix: algorithmic tier vs manual tier
- Cohen's kappa for inter-rater reliability
- Per-tier precision/recall of the rule-based system
- Script location (planned): `scripts/tf_labeling/evaluate_classifier.py`

### Phase 3: ML Classifier (after ~300-500 labels)

- Random Forest / XGBoost on existing 18 features
- Stratified 5-fold cross-validation
- Compare to rule-based system on held-out manual labels
- Script location (planned): `scripts/tf_labeling/train_classifier.py`

### Phase 4: Iteration

- Label disagreements between algo and ML
- Optional: trace-based features (1D CNN on raw z-traces)
- Retrain and refine

See `docs/AI_interaction/tf_manual_labeling_plan.md` for the full plan.

---

## File Inventory

```
src/visdetect/analysis/
    tf_labeling.py                  # Library: labels, queue, data loading

scripts/tf_labeling/
    run_labeling_gui.py             # Interactive GUI (TkAgg)
    precache_rasters.py             # Pre-compute raster NPZs (parallel)

data/labels/
    tf_manual_labels.csv            # Ground truth (created on first label)

data/cache/tf_raster_cache/
    {session}_{cluster}_raster.npz  # Per-unit raster data

docs/AI_interaction/
    tf_manual_labeling_plan.md      # Design plan
```
