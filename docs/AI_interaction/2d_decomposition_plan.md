# Plan: 2D Decomposition — Task-State x Sensory Coding Directions

## Context

Lohse et al. showed that in expert mice, striatal population activity has two orthogonal dimensions: a **task-state coding direction** (predicts Hit vs Miss from pre-change baseline activity, controlled by atMOs) and a **sensory input coding direction** (tracks TF fluctuations). These form an AND-gate for action initiation. This project's unique longitudinal data (Naive→Expert) enables asking: **does this orthogonal geometry emerge with learning?**

All ingredients exist in the current codebase — the coding direction computation (`_compute_cd_shrinkage`), TF pulse traces (NPZ cache), and population tensor infrastructure. This plan connects them into a single analysis.

## New File

**`analysis_suite/03_population/f_2d_decomposition.py`** (Fig 17b)

Figure output: `figures/03_population/fig17b_2d_decomposition.png`
Stats output: `figures/03_population/2d_decomposition_stats.csv`
Cache: `analysis_suite/cache/2d_decomposition/` (per-session NPZ + summary CSV)

## Algorithm (per session)

### Step 1: Unit selection
- `good_ids = get_good_cluster_ids(sess)` from `utils.py:174`
- `tf_data = load_tf_traces_npz(session_name)` from `loader.py:203`
- `common_ids = sorted(set(good_ids) & set(tf_data['cluster_ids']))` — must have ≥10

### Step 2: Task-state CD (CD_task)
- Build population tensor aligned to `Change_ON`, window `(-1.5, 1.0)s`, bin_size=0.025
- Select **go-trial** Hits and Misses only (change_size > 1.01, outcomes 'hit'/'miss')
- Average each trial in **pre-change baseline** `(-1.0, 0.0)s` → (n_trials, n_units)
- Shrinkage LDA: `w = (Cov + reg*I)^{-1} * (mu_hit - mu_miss)`, normalized
- Reuses `_compute_cd_shrinkage()` from `a_coding_direction.py:73` (copied locally, same pattern as `d_state_matched_cd.py:132`)

### Step 3: Sensory CD (CD_sensory)
- From NPZ cache, for each common unit: extract signed peak of `fast_z` trace in (0.0, 0.2)s post-pulse window
- Form vector of peak amplitudes, normalize to unit length
- No recomputation of pulse alignments needed — reuses pre-computed cache

### Step 4: Orthogonality test
- Cosine similarity: `cos_sim = dot(CD_task, CD_sensory)`
- Null distribution: 1000 permutations of unit identity → p-value
- Track cos_sim across sessions and learning stages

### Step 5: 2D projections
- Project time-resolved population activity onto **both** axes simultaneously
- Trial types: Hit, Miss, SDT-FA (catch-trial hit), CR (catch-trial miss)
- All aligned to Change_ON (valid for all these SDT categories)
- Compute mean 2D trajectories per trial type, per stage

### Step 6: Pre-change 2D position
- Average projection in baseline window (-1.0, 0.0)s per trial type
- This shows where each trial type "starts" in the 2D space before change onset

## Figure Layout (4 rows × 3 columns)

| Panel | Content |
|-------|---------|
| A | Cosine similarity per session, colored by stage, dashed line at 0 |
| B | Cosine similarity vs session index + Spearman trend |
| C | Cosine similarity by stage (box + swarm), Kruskal-Wallis |
| D (wide) | 2D trajectories — Expert: Hit, Miss, FA, CR with time as color gradient |
| E | 2D trajectories — Learning sessions |
| F | Pre-change 2D position scatter (session dots by trial type) |
| G | Pre-change task-state projection: Hit vs Miss by stage |
| H | Pre-change sensory projection: Hit vs Miss by stage |
| I (wide) | 2D density at lick time for Hit vs SDT-FA — "action zone" |
| J | Summary stats table |

## Statistics

| Test | Method |
|------|--------|
| Orthogonality (all sessions) | Wilcoxon signed-rank vs 0 |
| Orthogonality by stage | Wilcoxon per stage |
| cos_sim trend | Spearman vs session_idx |
| cos_sim Learning vs Expert | Mann-Whitney U + Kruskal-Wallis |
| Baseline task-state: Hit vs Miss | Permutation test (1000 shuffles), per stage |
| Baseline sensory: Hit vs Miss | Permutation test (1000 shuffles), per stage |
| Per-session orthogonality | Bootstrap p-value (1000 permutations of unit identity) |

## Reused Infrastructure

| Component | Source |
|-----------|--------|
| `_compute_cd_shrinkage(X, y, reg)` | `03_population/a_coding_direction.py:73` (copy locally) |
| `build_population_tensor()` | `analysis_suite/utils.py:29` |
| `get_good_cluster_ids()` | `analysis_suite/utils.py:174` |
| `load_tf_traces_npz()` | `analysis_suite/loader.py:203` |
| `load_staging_manifest()` | `analysis_suite/loader.py` |
| `load_hmm_assignments()` | `analysis_suite/loader.py` |
| `bootstrap_ci()`, `permutation_test()` | `analysis_suite/utils.py` |
| `setup_style()`, `save_figure()` | `analysis_suite/plotting.py` |
| Constants | `visdetect.analysis.constants` |

## Key Parameters

```
TASK_CD_WINDOW = (-1.5, 1.0)       # Full tensor window (for time-resolved trajectory visualization)
BASELINE_AVG_WINDOW = (-1.0, 0.0)  # Pre-change averaging for CD_task computation (NO motor contamination)
TF_PEAK_WINDOW = (0.0, 0.2)        # Post-pulse window for CD_sensory
MIN_UNITS = 10
MIN_TRIALS_PER_CLASS = 8
N_PERM_ORTHO = 1000
```

**Important**: CD_task is computed ONLY from pre-change baseline (-1.0 to 0.0s) — no motor confound.
The wider tensor window (-1.5 to +1.0s) is loaded solely so we can project the time-resolved
trajectory through the 2D space, showing the full sensory→action path (as in Lohse et al. Fig 3i,k).
Post-change activity (which includes motor components after ~200-250ms) is visualized but never
enters the CD definition.

## Registration

Add to `analysis_suite/run_all.py` after `e_sensory_dose_response.py`:
```python
("03_population/f_2d_decomposition.py", "Fig17b 2D Decomposition"),
```

## Verification

1. Run: `cd analysis_suite && py 03_population/f_2d_decomposition.py`
2. Check per-session cosine similarity values (expect near 0 in Expert if Lohse framework holds)
3. Check that figure saves to `figures/03_population/fig17b_2d_decomposition.png`
4. Check stats CSV for all reported tests
5. Sanity: cos_sim null distribution should be centered at 0
6. Sanity: CD_task should have nonzero baseline separation (Hit > Miss)
7. Verify session count ≥ 20 (expect ~26 minus any without TF traces)
