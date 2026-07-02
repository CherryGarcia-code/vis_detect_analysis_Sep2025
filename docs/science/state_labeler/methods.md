# Behavioral-state labeler — methods (exact computation & statistics)

A precise, presentation-ready description of how the tagger works, for answering
"how exactly is this computed?" Companion to the results writeup
[`calibration_results_2026-06-11.md`](calibration_results_2026-06-11.md). All
numbers are the current 4-state model (BG_046).

**One-line summary.** From an experimenter's *sparse* labels on the trial-outcome
raster, we fit a *shallow, interpretable* decision tree over *local outcome-composition
features*, then apply that fixed rule to tag every trial of every session — no neural
data, no black box.

---

## Stage 1 — Outcome raster (lick-valence per trial)

Each trial is reduced to one categorical **lick valence** from its behavioral
outcome and trial type (`classify_lick_valence`, `build_outcome_raster`):

| lick valence | when | meaning |
|---|---|---|
| `appropriate_lick` | `outcome=hit` AND go trial (`change_size > 1`) | licked to a real change |
| `inappropriate_lick` | `outcome=fa` (any trial) **or** `outcome=hit` on a catch trial | early lick, or a catch-trial SDT false alarm |
| `nolick` | `outcome=miss` (covers go-miss and catch correct-rejection) | withheld |
| `abort` | `outcome=abort` | trial terminated before the change |
| `ref` | `outcome=ref` (reflex lick) | excluded from all feature fractions |

Go vs catch is decided by `change_size` (> 1 = go), never by the outcome label.

## Stage 2 — Sparse expert labels

In a raster GUI (`run_state_labeler.py`) the experimenter drag-paints contiguous
**spans** over runs they are confident about, choosing one of four states
(`Impulsive`, `StimSens`, `Disengaged`, `Abort`). Ambiguous stretches are left
**blank on purpose** — the model is trained only on high-confidence exemplars;
ambiguity is resolved later at tag time, not label time. Spans are stored as
`StateEpisode(session, start_trial, end_trial, state_label)` rows in
`state_episodes.csv`. Current training set: **27 sessions, 216 spans**
(Abort 64, Impulsive 73, StimSens 57, Disengaged 22).

## Stage 3 — Local features (`extract_state_features`)

Per trial we compute **six local outcome-composition fractions** over a symmetric,
centered window of width **W** trials (`rolling(W, center=True, min_periods=1)`, so
windows shrink at session edges). The denominator is the number of **non-`ref`**
trials in the window (a window of all `ref` → fraction 0):

```
denom            = Σ_window ( outcome ≠ ref )
f_applick        = Σ_window ( appropriate_lick )   / denom
f_inapplick      = Σ_window ( inappropriate_lick ) / denom
f_nolick         = Σ_window ( nolick )             / denom
f_abort          = Σ_window ( abort )              / denom
f_miss_easy      = Σ_window ( nolick AND go AND change_size ≥ 2.0 ) / denom
f_hit_hard       = Σ_window ( appropriate_lick AND change_size < 2.0 ) / denom
```

`f_miss_easy` / `f_hit_hard` are *difficulty-aware* features (misses on easy changes,
hits on hard changes); `STATE_EASY_CHANGE_THRESH = 2.0` (the TF-ratio easy/hard split).
The feature set is `STATE_FEATURE_COLS`; in practice the fitted tree keys almost
entirely on `f_nolick, f_abort, f_inapplick, f_applick`.

## Stage 4 — Calibrate the rule (`calibrate_states`)

**Model.** `sklearn.tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
class_weight="balanced", random_state=42)` — deliberately shallow so the rule is a
readable set of thresholds; `class_weight="balanced"` compensates for the rare
`Disengaged`/`Abort` classes.

**Window selection (the only hyper-parameter).** `W` is chosen by
**leave-one-session-out (LOSO)** cross-validation over the grid
`STATE_LABEL_W_GRID = [11, 15, 21, 31, 41, 51, 61]`:

1. For each `W`: pool all labelled trials with their features.
2. LOSO — hold out each labelled session in turn, fit the tree on the other
   sessions' labelled trials, predict the held-out session's labelled trials, and
   score with **Cohen's κ** (agreement corrected for chance).
3. A fold is **skipped** (not scored) if the held-out session, or the training set,
   has fewer than 2 label classes — a single-class held-out fold makes
   `cohen_kappa_score` compute 0/0 = NaN, which would poison the mean; that session's
   labels still *train* the model in every other fold.
4. `W` = the grid value with the highest mean LOSO κ (`np.nanmean` as a backstop).

**Refit.** At the chosen `W`, the tree is **refit on all labelled trials** (LOSO is
only for choosing `W` + honest generalization estimate). The result
(`CalibrationResult`: tree, `W`, LOSO κ, human-readable `rules.md`) is pickled to
`state_rule.pkl`.

**Selected model:** `W = 15`, **LOSO Cohen's κ = 0.709** (honest, held-out),
**in-sample κ = 0.764** (resubstitution). The small gap (0.055) indicates the
depth-3 tree is not overfitting. Effective learned rule:

```
if f_nolick ≤ 0.41:                         # still responding
    if   f_abort     > 0.38  → Abort
    elif f_inapplick > 0.45  → Impulsive
    else                     → StimSens
else:                                        # withdrawn
    if f_applick ≤ 0.24 (few appropriate licks) → Disengaged  else → StimSens
```

## Stage 5 — Tag every session (`tag_features` / `decode_session_states`)

For each trial of each session: compute the same features at the fixed `W`, then

- `probs = tree.predict_proba(features)`, over the tree's classes (alphabetical:
  Abort, Disengaged, Impulsive, StimSens);
- `state_label` = argmax class; `state_confidence` = max class probability;
- `state_gated` = −1 when `state_confidence ≤ STATE_CONFIDENCE_THRESHOLD (0.8)`, else 0
  (flags low-confidence trials for optional exclusion downstream);
- `p_state_{k}` = per-class probabilities.

Output columns also include `hmm_state`/`hmm_state_label`/`hmm_state_gated` aliases and
the SDT booleans (`is_hit`/`is_fa`/`is_miss`/`is_go`/`is_catch`/`outcome`/`session_name`),
so a concatenation of the per-session CSVs is a **drop-in** for the existing
`hmm_downstream` interface. One CSV per session under
`data/cache/state_tags/{SUBJECT}/{session}.csv`.

## Stage 6 — Validate (`validate_states.py`)

On the labelled sessions, compare tagger vs experimenter labels: **Cohen's κ** +
**confusion matrix**, and a per-session 3-track re-shade figure (raster / your-labels /
tagger). Current confusion (rows = labels, cols = tagger; 6 560 labelled trials):
per-class recall/precision — Abort 97%/61%, Disengaged 88%/86%, Impulsive 88%/94%,
StimSens 78%/81%.

---

## Cross-subject application

The rule is **subject-agnostic** (a tree over outcome-composition fractions), so the
BG_046-trained `state_rule.pkl` is applied unchanged to BG_031/038/039 (set
`VISDETECT_SUBJECT`). There are **no ground-truth labels** for the other subjects, so
this is a **face-validity** check (no κ): each tagged state still carries its defining
outcome signature across all mice (Impulsive→high `f_inapplick`, Disengaged→high
`f_nolick`, Abort→high `f_abort`, StimSens→highest `f_applick`).

## Reading the figures

- **Occupancy** = fraction of a session's/subject's trials tagged a given state.
- **Signature** (the transfer heatmap) = mean *defining-outcome fraction WITHIN* trials
  of that state (e.g. "of Disengaged-tagged trials, 68% of local outcomes are no-licks").
  This is high by construction and is NOT occupancy — the two are different quantities.
- **Pipeline slide** dashed lines = the tree's cut thresholds on each feature (the same
  values as the flowchart nodes).

## Caveats / honest limitations

- **Labels are the experimenter's judgments, not ground truth** — κ measures
  reproducibility of *that* labeling, not correctness of the ontology.
- **Partly self-referential.** States are *defined* by these outcome features, so the
  signature/definition consistency is expected; it shows the states are *cleanly
  separable*, not that they are behaviorally meaningful (that would need
  state-conditioned d′/RT).
- **In-sample κ (0.764) is optimistic;** the LOSO κ (0.709) is the honest number.
- **Rare states rest on fewer exemplars** (Disengaged 22 spans) — add confident spans
  rather than labeling ambiguous trials to strengthen them.
- **Cross-subject = face validity only** (no ground-truth κ). A real cross-subject κ
  needs labeling a few of that subject's sessions.
- **Pickle is version-coupled** to scikit-learn — regenerate via `calibrate_states`
  after an sklearn upgrade.

## Key constants (`visdetect.analysis.constants`)

| constant | value |
|---|---|
| `STATE_LABELS` | Impulsive, StimSens, Disengaged, Abort |
| `STATE_FEATURE_COLS` | f_applick, f_inapplick, f_nolick, f_abort, f_miss_easy, f_hit_hard |
| `STATE_LABEL_W_GRID` | 11, 15, 21, 31, 41, 51, 61 |
| selected `W` | 15 |
| `STATE_EASY_CHANGE_THRESH` | 2.0 (TF ratio) |
| `STATE_CONFIDENCE_THRESHOLD` | 0.8 (gating) |
| tree | depth 3, min_samples_leaf 5, class_weight balanced, seed 42 |
| LOSO κ / in-sample κ | 0.709 / 0.764 |

## Reproduce

```
py scripts/state_labeling/run_state_labeler.py      # 1. label   -> state_episodes.csv
py scripts/state_labeling/calibrate_states.py       # 2. fit     -> state_rule.pkl + rules.md
py scripts/state_labeling/tag_sessions.py           # 3. tag     -> data/cache/state_tags/{SUBJECT}/
py scripts/state_labeling/validate_states.py        # 4. validate-> kappa, confusion, figures
# slides:
py scripts/state_labeling/make_pipeline_figure.py --subject BG_046 --session 19082025
py scripts/state_labeling/make_learning_arc.py --subject BG_046 \
   --sessions 15072025 19082025 11092025 --labels "Early|naïve (QC-fail)" "Middle|Learning" "Late|Expert" --equal-length
```
