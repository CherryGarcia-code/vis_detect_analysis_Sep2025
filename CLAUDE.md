# CLAUDE.md — Project Manual for Claude Code

## Project Identity

Single-subject (mouse BG_046) chronic Neuropixels 2.0 recordings from medial striatum during a **visual change-detection task**. The mouse reports changes in temporal frequency (TF) of drifting gratings by licking. Across ~42 sessions, the mouse progresses Naive → Learning → Expert. The project studies how striatal circuits (D1/D2 SPNs, FSIs) support perceptual learning, action selection, and behavioral state regulation.

**Reference**: Khilkevich & Lohse, Nature 2024 (brain-wide dynamics, ~250 ms integration timescale).

---

## Critical Analysis Rules

These rules are **non-negotiable**. Violating them produces scientifically invalid results.

### Task Structure and Trial Types

The task has a **stimulus change detection** structure:
- **Baseline period**: Drifting grating at a base TF with stochastic fluctuations (~50 ms updates)
- **Change event**: TF changes by a ratio (go trials) or stays the same (catch trials)
- **Response window**: Mouse licks to report detection

Trial classification by `change_size` (NOT by behavioral label):
- **Go trial**: `change_size > 1.0` (stimulus actually changed). Ratios: [1.25, 1.35, 1.5, 2.0, 4.0]
- **Catch trial**: `change_size ≈ 1.0` (no real change)

### Outcome Definitions — Two Distinct Systems

**Behavioral software labels** (the `trialoutcome` field in each Trial):
| Label | Meaning |
|-------|---------|
| `hit` | Mouse licked in the response window after change onset |
| `miss` | Mouse withheld lick on a go or catch trial |
| `fa` | **Early/anticipatory lick** during baseline, BEFORE the change event |
| `abort` | Trial terminated early (no change stimulus was ever presented) |
| `ref` | Reflex lick (too fast to be a real detection response) |

**SDT (Signal Detection Theory) classification** used for d' and psychometrics:
| SDT Category | How Defined | Trial Type |
|-------------|-------------|------------|
| **SDT Hit** | `outcome='hit'` AND `change_size > 1.0` | Go trial, correct detection |
| **SDT Miss** | `outcome='miss'` AND `change_size > 1.0` | Go trial, missed change |
| **SDT False Alarm** | `outcome='hit'` AND `change_size ≈ 1.0` | Catch trial, licked when no change |
| **SDT Correct Rejection** | `outcome='miss'` AND `change_size ≈ 1.0` | Catch trial, correctly withheld |
| **Excluded from SDT** | `fa`, `ref`, `abort` outcomes | Not valid for sensitivity computation |

**The `fa` label is NOT an SDT false alarm.** It means the mouse licked before the change event (anticipatory/impulsive lick). SDT false alarms are catch-trial `hit` outcomes.

### Event Alignment Rules — CRITICAL

| Event | Valid Outcomes | Why |
|-------|---------------|-----|
| `Change_ON` | `hit`, `miss` ONLY | On `fa`/`abort` trials, **the change stimulus was never presented**. Aligning to "change time" on these trials is scientifically meaningless. |
| `FA` (early lick) | `fa` only | Motor-aligned: the lick itself is the event |
| `Hit` (response lick) | `hit` only | Motor-aligned: lick after detected change |
| `Baseline_ON` | All outcomes | Every trial has a baseline period |

These rules are codified in `EVENT_VALID_OUTCOMES` in `visdetect/analysis/constants.py`. Always use them.

### Unit Quality and Selection

**Priority order for cluster IDs** (used consistently across all modules):
1. `session.good_and_stable_ids` — UnitMatch-tracked stable units (best)
2. `session.good_cluster_ids` — Kilosort "good" label
3. All clusters (last resort, only for exploratory analysis)

**QC filters** (from `config/qc_profiles.yml`):
- Minimum firing rate: 1.0 Hz (`DEFAULT_MIN_FR`)
- ISI violations: < 20% (`max_isi_viol_frac`)
- Minimum total spikes: 500

### Session Selection

Controlled by `SESSION_FILTER` in `visdetect/analysis/config.py`:
- `merge_naive_learning = True` → Naive relabeled as Learning. Only 2 stages: **Learning**, **Expert**
- `min_trials = 150`, `min_dprime = 0.8`
- All scripts use `load_staging_manifest(qc_only=True)` which applies this filter automatically

### Key Constants (single source of truth: `visdetect/analysis/constants.py`)

| Constant | Value | Used For |
|----------|-------|----------|
| `DEFAULT_BIN_SIZE` | 0.025 s (25 ms) | PSTH bin width |
| `DEFAULT_SIGMA_MS` | 25.0 ms | Gaussian smoothing sigma |
| `DEFAULT_Z_THRESH_TF` | 3.0 | TF responsiveness z-score threshold |
| `TF_PULSE_PRE_WINDOW` | (-0.4, 0.0) s | TF pulse baseline window |
| `TF_PULSE_POST_WINDOW` | (0.0, 0.5) s | TF pulse response window |
| `TF_FAST_THRESH_LOG2` | 0.25 | Fast TF pulse threshold (log2 scale) |
| `TF_SLOW_THRESH_LOG2` | -0.25 | Slow TF pulse threshold (log2 scale) |
| `FA_RT_SPLIT` | 3.0 s | Early vs late FA split |
| `CHANGE_SIZES` | [1.25, 1.35, 1.5, 2.0, 4.0] | Go-trial TF change ratios |

---

## Neuroscience Best Practices

Follow these standards for all analyses. These reflect current best practices in systems neuroscience and decision-making research.

### Spike Data Analysis
- **Never average raw firing rates across units without normalization.** Use z-scoring (baseline-subtracted, divided by baseline SD) or baseline-subtracted rates to compare across units with different firing rates.
- **PSTH smoothing**: Gaussian kernel with σ=25 ms is standard for striatal neurons. Do not over-smooth (σ>50 ms obscures temporal dynamics).
- **Permutation tests** (500-1000 shuffles) for single-unit significance. Do not rely solely on parametric tests for neural data.
- **FDR correction** (Benjamini-Hochberg, α=0.05) for mass screening across units. Do NOT apply FDR across separate scientific questions/figures.
- **Report effect sizes** alongside every p-value. Neural data with large n can yield tiny p-values for biologically meaningless effects.

### Population Analysis
- **Trial-match conditions** when comparing population responses (e.g., Hit vs Miss). Unequal trial counts bias variance estimates. Subsample the larger group or report both matched and unmatched results.
- **Cross-validation** for all decoders. Never test on training data. Use stratified k-fold (k=5) to maintain class balance.
- **Null distributions** for decoding: label-shuffle permutation (≥200 shuffles), report chance as mean ± 2 SD of null.
- **Coding direction (CD) vectors**: Compute on training folds, project on held-out data to avoid circularity.

### Signal Detection Theory
- d' = z(hit_rate) - z(fa_rate), with log-linear correction clipping rates to [0.01, 0.99]
- Hit rate computed on **go trials only**. FA rate on **catch trials only**.
- Always report criterion c alongside d' when assessing response bias.

### Statistical Standards
- **Non-parametric by default** for neural data (Kruskal-Wallis, Mann-Whitney U, Spearman rho). Neural distributions are almost never Gaussian.
- **Two-sided tests** unless the biological hypothesis is strongly directional.
- **Bootstrap CI** (1000 resamples, seed=42, percentile method) for key estimates.
- See the Research Statistician skill for the full decision framework and effect size formulas.

### Normalization Best Practices

**The Golden Rule**: Normalize each unit separately using a **shared baseline definition** across all conditions being compared.

#### When to Normalize
- **Always normalize** when comparing firing rates across neurons (population averages, heatmaps, decoding)
- **Never average raw rates** across units — high-FR neurons will dominate
- **Within-unit comparisons** (single neuron, single trial) may use raw rates

#### Decision Tree

| Analysis Type | Method | Rationale |
|---------------|--------|-----------|
| Single-unit responsiveness | Per-trial Δrate + permutation test | Paired comparison within unit |
| Population heatmaps | Per-unit z-score (shared baseline) | Equalizes units for visualization |
| Coding directions | Δrate (baseline-subtracted) | Preserves Hz units, interpretable projections |
| Grand averages across sessions | Shared baseline z-score | Preserves relative magnitude between conditions |
| Decoding | Z-score to shared baseline | Removes baseline confounds, units contribute equally |
| TF responsiveness screening | Per-unit z-score | Single-unit significance testing |
| Modulation strength comparison | Percent change (if FR > 1 Hz) | True equalization for multiplicative effects |

#### Normalization Methods

**Z-score** (most common):
```python
from utils import compute_zscore_normalized
tensor_z = compute_zscore_normalized(tensor, bin_centers, baseline_window)
# Returns: (rate - baseline_mean) / baseline_std per unit
```
- **Use for**: Heatmaps, population comparisons, significance testing
- **Strengths**: Units are interpretable (SD of baseline noise), removes baseline differences
- **Pitfalls**: Requires stable baseline, can inflate low-FR units if SD is tiny

**Baseline-subtracted (Δrate)**:
```python
from utils import compute_baseline_subtracted
tensor_delta = compute_baseline_subtracted(tensor, bin_centers, baseline_window)
# Returns: rate - baseline_mean per unit (preserves Hz units)
```
- **Use for**: Coding directions, population averages where Hz units matter
- **Strengths**: Preserves firing rate units, robust to small SD
- **Pitfalls**: Still biased toward high-FR units (not fully equalized)

**Percent change** (special cases):
```python
percent_change = 100 * (rate - baseline_mean) / max(baseline_mean, 1.0)
```
- **Use for**: Modulation strength comparisons across neurons
- **Strengths**: True equalization (doubling = doubling regardless of baseline)
- **Pitfalls**: Explodes for low baselines, use clamp `max(baseline, 1.0)`

#### Critical Pitfalls to Avoid

1. **Circular baseline** (CRITICAL ERROR):
   ```python
   # WRONG: Each condition normalized to its own baseline
   hit_z = (hit - hit_baseline.mean()) / hit_baseline.std()
   fa_z = (fa - fa_baseline.mean()) / fa_baseline.std()
   # This inflates FA's z-score because FA has low activity → low SD → inflated z

   # CORRECT: Shared baseline computed once
   all_baseline = tensor[:, baseline_mask, :].ravel()
   baseline_mean = all_baseline.mean()
   baseline_std = all_baseline.std()
   hit_z = (hit - baseline_mean) / baseline_std
   fa_z = (fa - baseline_mean) / baseline_std
   ```

2. **Average-then-normalize** (WRONG ORDER):
   ```python
   # WRONG: High-FR neurons dominate the average
   pop_avg = np.mean([unit1_rate, unit2_rate, ...], axis=0)
   normalized = (pop_avg - pop_avg.mean()) / pop_avg.std()

   # CORRECT: Normalize each unit, then average
   unit1_z = (unit1_rate - baseline_mean) / baseline_std
   unit2_z = (unit2_rate - baseline_mean) / baseline_std
   pop_avg = np.mean([unit1_z, unit2_z, ...], axis=0)
   ```

3. **Division by zero**:
   ```python
   # Guard against zero variance
   if baseline_std < 1e-6:
       baseline_std = 1.0  # or skip this unit, or use Δrate instead
   z_score = (rate - baseline_mean) / baseline_std
   ```

4. **Inconsistent baseline windows**:
   - Always import baseline windows from `constants.py` (e.g., `TF_PULSE_PRE_WINDOW`)
   - Use the **same** baseline definition across all conditions in a comparison
   - Common baseline: `(-0.5, -0.05)` s before Change_ON

5. **Wrong method for task**:
   - Decoding on **raw rates** biases toward high-FR units → use z-score or Δrate first
   - Heatmaps without normalization are uninterpretable → always z-score per unit
   - Grand averages across sessions need **shared baseline** to preserve relative magnitude

#### Where Normalization Lives

| Module | Function | Purpose |
|--------|----------|---------|
| `analysis_suite/utils.py` | `compute_zscore_normalized()` | Per-unit z-score with shared baseline |
| `analysis_suite/utils.py` | `compute_baseline_subtracted()` | Per-unit Δrate (Hz units preserved) |
| `src/visdetect/analysis/tf_pulse.py` | `_zscore_trace()` | TF pulse z-scoring (single trace) |

#### Recent Fixes (March 2026)
- **Scripts 03a, 03d, 03e**: Now use shared baseline normalization for grand averages (preserves Hit-FA relative magnitude)
- **Scripts 04a, 04b, 04c**: Updated to normalize before decoding (removes baseline confounds)

#### Quick Reference Card

| Goal | Code Snippet |
|------|--------------|
| Z-score tensor | `tensor_z = compute_zscore_normalized(tensor, bin_centers, baseline_window)` |
| Δrate tensor | `tensor_delta = compute_baseline_subtracted(tensor, bin_centers, baseline_window)` |
| Single-unit z-score | `z = (rate - pre_mean) / pre_std` with `if pre_std < 1e-6: pre_std = 1.0` |
| Check baseline | `print(f"Baseline window: {baseline_window}, bins: {baseline_mask.sum()}")` |

See `analysis_suite/NORMALIZATION_AUDIT_MARCH2026.md` for full audit report.

### Consistency Checks (Do These for Every New Script)
1. **Are event alignment outcomes filtered correctly?** Check against `EVENT_VALID_OUTCOMES`.
2. **Are constants imported from the canonical source?** Never hardcode thresholds that exist in `constants.py`.
3. **Is the session filter consistent?** Use `load_staging_manifest()` — don't manually filter.
4. **Are units selected by the standard QC pipeline?** Use `get_good_cluster_ids()` or `load_kept_ids()`.
5. **Are existing utility functions used?** Search the codebase before writing new code.
6. **Are color palettes consistent?** Use `STAGE_COLORS`, `HMM_STATE_COLORS`, `OUTCOME_COLORS`, `CELLTYPE_COLORS` from config.
7. **Is normalization correct?** Shared baseline, normalize-then-average, division-by-zero guards.

---

## Architecture

### Library: `src/visdetect/`

The core Python package. All reusable logic lives here.

```
visdetect/
  core/
    session.py     SessionData dataclass (Trial, Cluster, Session), load_session(), save_session()
    io.py          .mat file loading (scipy + h5py fallback), parse_good_cluster_ids()
    qc.py          Unit selection pipeline, QC profiles (YAML), firing rate / ISI filtering
    kilosort.py    Attaches Kilosort waveforms to Cluster objects
  analysis/
    config.py      SINGLE SOURCE OF TRUTH: paths, colors, stages, manifest loading
    constants.py   All numeric thresholds, event windows, change sizes
    behavior.py    SDT metrics, trial classification, psychometrics, manifest filtering
    align.py       Spike-event alignment, PETH computation, HDF5 caching
    tf_pulse.py    TF pulse responsiveness screening, z-scored traces, grid plots
    lick.py        FA lick-responsive neuron detection (MATLAB-style SALT approach)
    optotagging.py SALT test for D1/D2 optogenetic identification
    su_analysis.py Single-unit: QC tables, raster/PSTH, population heatmaps
    hmm.py         Bernoulli GLM-HMM: fitting, Viterbi decoding, model selection
    hmm_downstream.py  State-conditioned analysis, transition dynamics, online prediction
  viz/plotting.py  set_style(), despine()
  utils/
    synthetic.py   make_synthetic_session() for testing
    progress.py    Progress bar wrapper (tqdm or fallback)
```

### Analysis Suite: `analysis_suite/`

Self-contained publication figure pipeline. 29+ figures across 9 modules.

**Shared infrastructure** (imported as flat modules by all scripts):
| File | Role |
|------|------|
| `config.py` | Re-exports `visdetect.analysis.config.*` + `FIGURE_DIR`, `CACHE_DIR` |
| `loader.py` | Session loading, manifest access, GLT, HMM, lick, TF traces, waveform labels, `build_unit_table()` |
| `utils.py` | `build_population_tensor()`, `smooth_psth()`, `compute_zscore_normalized()`, `get_good_cluster_ids()`, `bootstrap_ci()`, `permutation_test()`, `fdr_correct()`, `compute_auroc()` |
| `plotting.py` | `setup_style()`, `save_figure()`, `add_stage_background()`, `plot_significance_stars()`, `multi_panel_figure()` |
| `run_all.py` | Sequential runner for the 29 core scripts (30-min timeout each) |

**Modules**: `01_behavior/` (Figs 1-7), `02_single_unit/` (Figs 8-12), `03_population/` (Figs 13-17), `04_decoding/` (Figs 18-20), `05_longitudinal/` (Figs 21-23), `06_lick_motor/` (Figs 24-26), `07_advanced/` (Figs 27-34), `08_tf_pulse/` (Figs 35-42), `09_optotagging/` (Fig 43)

### Scripts: `scripts/`

Standalone utilities organized by domain:
- `scripts/analysis/behavior/` — Behavioral pipelines, HMM fitting
- `scripts/analysis/lick/` — Lick-responsive neuron detection
- `scripts/analysis/tf_response/` — TF pulse analysis
- `scripts/analysis/stage_sessions.py` — **KEY**: Generates the staging manifest
- `scripts/batch_processing/batch_convert_MatToPkl.py` — Primary .mat→.pkl converter
- `scripts/pipelines/concat_sort/` — Concatenated Kilosort4 sorting pipeline
- `scripts/QC_CHECKS/` — Session and unit QC diagnostics

---

## How to Write a New Analysis Script

### Template (analysis_suite)

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

### Naming conventions
- Script: `{letter}_{descriptive_name}.py` (e.g., `d_post_error_psychometric.py`)
- Figure output: `figures/{module}/fig{NN}_{name}.png`
- Cache: `cache/{descriptive_name}.csv`
- Stats: `figures/{module}/{name}_stats.csv`

### Checklist before finalizing
- [ ] Imports constants from canonical source (not hardcoded)
- [ ] Uses `load_staging_manifest()` for session selection
- [ ] Uses `get_good_cluster_ids()` or `load_kept_ids()` for unit selection
- [ ] Filters event alignments by `EVENT_VALID_OUTCOMES`
- [ ] Calls `setup_style()` before plotting
- [ ] Uses `save_figure()` for output
- [ ] Cleans up sessions with `del sess; gc.collect()`
- [ ] Color palette matches project conventions
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

---

## Gotchas and Pitfalls

| Gotcha | Detail |
|--------|--------|
| `py` not `python` | Windows + Git Bash requires `py` to invoke Python |
| Legacy pickle paths | `RenamingUnpickler` handles 10+ historical module paths. Don't panic about import errors on load. |
| pre-TPrime = stale | Files in `preTprime/` directories are from before spike time correction. Do NOT use for new analyses. |
| Session name format | DDMMYYYY as integer (e.g., `7072025` = July 7, 2025). Use `parse_session_date()` and `chronological_sort()`. |
| `change_size` determines trial type | Go vs catch is from `change_size`, NOT from the `trialoutcome` label. |
| `fa` ≠ SDT false alarm | The `fa` behavioral label means early/anticipatory lick. SDT FAs are `hit` outcomes on catch trials. |
| Memory management | Always `del sess; gc.collect()` after processing each session in loops. Sessions are large (~100+ MB). |
| Search before writing | **Always search the codebase for existing functions before writing new ones.** |

## Environment

- Windows 10, Git Bash shell
- Python in `.venv`: `.venv\Scripts\python.exe` (invoke via `py`)
- Run analysis_suite scripts: `cd analysis_suite && py 01_behavior/a_learning_curve.py`

## Skills

Six specialized skills in `.claude/skills/`:

### Scientific Workflow
| Skill | When to Use |
|-------|-------------|
| **Research Visualizer** | Figure design, color choices, layout, multi-option proposals |
| **Research Statistician** | Test selection, effect sizes, multiple comparisons, results tables |
| **Research Notes Summarizer** | Methods documentation, results summaries, scientific writing |

### Engineering Workflow
| Skill | When to Use |
|-------|-------------|
| **Codebase Auditor** | Systematic quality audit: alignment safety, constants, naming, figures, dependencies |
| **Analysis Runner** | Execute scripts, diagnose failures, interpret output, manage caches |
| **Pre-Commit Checker** | Fast quality gate on changed files before committing |

Skills activate automatically based on context. For full analysis workflow: Statistician → Visualizer → Summarizer. For code changes: Pre-Commit Checker → Auditor.

See also: `RUNNING.md` for a complete pipeline execution manual (Steps 1-8, prerequisites, troubleshooting).
