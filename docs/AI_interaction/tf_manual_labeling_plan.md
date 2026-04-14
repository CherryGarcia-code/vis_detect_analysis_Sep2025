# TF Manual Labeling & ML Classification Plan

**Created**: 2026-04-01
**Goal**: Build ground-truth manual labels for TF-responsive cell classification, then train an ML classifier to replace/augment the current rule-based tiered system.

---

## Current State

| Component | Status |
|-----------|--------|
| TF pulse trace computation | Mature — NPZ caches with z-scored fast/slow traces for all units |
| Binary responsiveness screen | Simple `|z| ≥ 3.0` threshold (Fig 35) |
| Tiered classification | Rule-based: permutation p-values → deterministic decision tree with rescue + trend filter (Fig 41) |
| Feature set | 18+ features per unit in `tf_cell_classification.csv` |
| Visual gallery | Static paginated PNGs, 50 units/page, grouped by tier (Fig 41g) |
| Manual labels | **None** |
| GUI / interactive tools | **None** |
| ML classifier | **None** |

### Existing Classification Tiers

| Tier | Label | Criteria |
|------|-------|---------|
| Tier 1 | Splitter | Both fast AND slow significant, **opposite** sign |
| Tier 2 | Unilateral | Only fast OR only slow significant |
| Tier 3 | Omni | Both fast AND slow significant, **same** sign |
| Non-responsive | — | Neither direction significant |

Sub-types: `Fast+/Slow-`, `Slow+/Fast-`, `Fast+`, `Fast-`, `Slow+`, `Slow-`, `Both+`, `Both-`, `Trend-excluded`, `None`.

### Existing Features (per unit, in `tf_cell_classification.csv`)

**Magnitude**: `peak_fast`, `peak_slow`, `auc_fast`, `auc_slow`, `z_abs_max_npz`
**Shape**: `half_width_fast_ms`, `half_width_slow_ms`
**Cross-direction**: `mirror_score`
**Statistical**: `p_peak_fast`, `p_peak_slow`, `p_auc_fast`, `p_auc_slow`
**Quality**: `n_fast_pulses`, `n_slow_pulses`, `trend_ratio`

---

## Implementation Plan

### Phase 1a: Label I/O & Data Infrastructure

**File**: `src/visdetect/analysis/tf_labeling.py`

- `LabelRecord` dataclass: session_name, cluster_id, manual_tier, manual_sub_type, confidence, notes, algo_tier, reviewer, timestamp
- `load_labels(path) -> DataFrame`: Load existing labels CSV
- `save_label(path, record)`: Append/update single label (crash-safe)
- `get_labeling_queue(classification_csv, labels_csv) -> DataFrame`: Return units sorted by priority:
  1. Borderline cases (near decision boundaries)
  2. Algorithmically responsive units (verify)
  3. High-z non-responsive (potential false negatives)
  4. Remaining units
- `compute_priority_score(row) -> float`: Priority metric for queue ordering

**File**: `data/labels/tf_manual_labels.csv`

Columns: `session_name, cluster_id, manual_tier, manual_sub_type, confidence, notes, algo_tier, reviewer, timestamp`

### Phase 1b: Interactive GUI

**File**: `scripts/tf_labeling/run_labeling_gui.py`

Matplotlib interactive viewer (no new dependencies). Layout:

```
┌──────────────────────────────────────┬─────────────────┐
│  Z-scored traces (fast blue/slow red)│   Metadata       │
│  with SEM bands                      │   - Session      │
│                                      │   - Cluster ID   │
├──────────────────────────────────────┤   - Algo tier     │
│  Raw PSTH (Hz) — fast + slow         │   - Sub-type     │
│                                      │   - Key features │
├──────────────┬───────────────────────┤   - Pulse counts │
│ Fast raster  │  Slow raster          │   - Progress     │
│              │                       │                   │
└──────────────┴───────────────────────┴─────────────────┘
  Keyboard: 1=Splitter 2=Unilateral 3=Omni 0=NR  j/k=nav  s=skip  n=note
```

**Features**:
- One unit at a time, full screen
- Keyboard-driven labeling (fast workflow)
- Auto-save after every label
- Progress tracking in title bar
- Smart ordering (borderline cases first)
- Navigate forward/back through queue
- Sub-type refinement via second keypress

### Phase 1c: Pre-cache Rasters

**File**: `scripts/tf_labeling/precache_rasters.py`

Pre-compute raster data (spike times aligned to fast/slow pulses) for all units and save to NPZ. This makes the GUI snappy instead of loading 100+ MB session pickles on every click.

### Phase 2: Agreement Analysis

Once ~300 labels exist:
- Confusion matrix: algo tier vs. manual tier
- Cohen's kappa
- Per-tier precision/recall of rule-based system
- Identify systematic failure modes

### Phase 3: ML Classifier

**File**: `scripts/tf_labeling/train_classifier.py`

- Random Forest / XGBoost on existing 18 features
- Stratified 5-fold CV
- Report macro-F1, per-tier precision/recall
- Compare to rule-based system on same held-out data
- Feature importance analysis

### Phase 4+: Iterate

- Label disagreements between algo and ML
- Add trace-based features (CNN on raw z-traces) if feature-based plateaus
- Retrain and refine

---

## Codebase Organization

```
src/visdetect/analysis/
    tf_labeling.py              # Label I/O, queue logic, data loading

scripts/tf_labeling/
    run_labeling_gui.py         # CLI entry point for GUI
    precache_rasters.py         # Pre-compute raster data for all units
    train_classifier.py         # Train ML model (Phase 3)
    evaluate_classifier.py      # Compare algo vs. ML vs. human (Phase 3)

data/labels/
    tf_manual_labels.csv        # Ground truth (grows incrementally)

analysis_suite/08_tf_pulse/
    i_classification_eval.py    # Fig 44: agreement matrix, feature importance
```

---

## GUI Display Elements (Phase 1)

1. **Z-scored fast + slow traces with SEM** — already in NPZ cache
2. **Raw PSTH (Hz)** — un-normalize using baseline mean/std from NPZ
3. **Raster plot for fast and slow pulses** — pre-cached in Phase 1c
4. **Text metadata**: session, cluster ID, pulse counts, algo tier, key features, progress

---

## Label Schema

| Column | Type | Values |
|--------|------|--------|
| `session_name` | str | Session identifier |
| `cluster_id` | int | Unit identifier |
| `manual_tier` | str | `Tier 1 (Splitter)`, `Tier 2 (Unilateral)`, `Tier 3 (Omni)`, `Non-responsive` |
| `manual_sub_type` | str | `Fast+/Slow-`, `Slow+/Fast-`, `Fast+`, `Fast-`, `Slow+`, `Slow-`, `Both+`, `Both-`, `None` |
| `confidence` | str | `high`, `medium`, `low` |
| `notes` | str | Free text |
| `algo_tier` | str | Algorithmic tier at time of review |
| `reviewer` | str | Reviewer name |
| `timestamp` | str | ISO 8601 datetime |
