# Behavioral State Labeler

A human-in-the-loop tool that learns **interpretable** behavioral-state thresholds
from your sparse labels on the outcome raster, then tags every session in a form
**drop-in compatible** with the existing GLM-HMM downstream interface.

States: `Impulsive`, `StimSens`, `Disengaged` (see `STATE_LABELS` in
`visdetect.analysis.constants`).

- **Library:** `src/visdetect/analysis/state_labeling.py` (data model, raster, queue,
  rendering) and `src/visdetect/analysis/state_calibration.py` (features, decision-tree
  calibration, tagging).
- **Design:** `docs/superpowers/specs/2026-06-10-behavioral-state-labeler-design.md`.

## Prerequisites

- Run from the **repo root** with the project Python (`py` on Windows + Git Bash, or
  `.venv\Scripts\python.exe`). Each script puts `src/` on `sys.path` itself.
- The **GUI uses TkAgg** — launch it from a real desktop terminal, not a headless/SSH
  session.
- Requires the usual data inputs: the QC staging manifest and the session `.pkl`s
  (session loading goes through `visdetect.suite.loader.load_session`).

Every script supports `--help`.

## Workflow (label → calibrate → tag → validate)

### 1. Label — `run_state_labeler.py`

```
py scripts/state_labeling/run_state_labeler.py
```

Shows each session's per-trial **outcome raster** in the Expert→Naive queue. You only
need **sparse** labels — drag across runs you're confident about.

Raster colors encode the **lick decision's valence**:

| Color | Meaning |
|-------|---------|
| 🟢 green  | appropriate lick — go-trial hit |
| 🔴 red    | inappropriate lick — early lick, or catch-trial false alarm |
| 🟣 purple | no-lick — miss or correct rejection |
| grey      | abort |
| muted tan | reflex lick (excluded from feature fractions) |

Catch trials are outlined in black. Press `c` to shade go-trial hits/misses by
change-size difficulty.

**Keys:** `1`=Impulsive · `2`=StimSens · `3`=Disengaged (active label) ·
**drag = paint a span** (saved on release) · `c`=toggle difficulty shading ·
`←`/`→`=prev/next session · `q`=quit. Previously-saved spans reappear (tinted) when you
revisit a session.

Spans append to `data/state_labels/state_episodes.csv` (git-diffable; hand-edit to fix a
mislabel). **Label ≥ 2 sessions before calibrating** — cross-validation needs more than one.

### 2. Calibrate — `calibrate_states.py`

```
py scripts/state_labeling/calibrate_states.py
```

Selects the window width `W` by leave-one-session-out (LOSO) Cohen's κ, then refits a
shallow `DecisionTreeClassifier` on all labeled trials. Writes:

- `data/state_labels/state_rule.pkl` — the fitted `CalibrationResult`.
- `data/state_labels/rules.md` — **read this**: chosen `W`, LOSO κ, and the
  human-readable decision-tree rule (the thresholds it learned).

A `NaN`/"unvalidated" warning means every LOSO fold was degenerate (e.g. only one labeled
session) — label more and re-run.

### 3. Tag — `tag_sessions.py`

```
py scripts/state_labeling/tag_sessions.py
```

Tags every QC-filtered manifest session → one CSV per session in
`data/cache/state_tags/{session}.csv`.

### 4. Validate & refine — `validate_states.py`

```
py scripts/state_labeling/validate_states.py
```

Prints Cohen's κ + a confusion matrix (tagger vs your labels) and saves re-shade PNGs to
`figures/state_labeler/`. Where the tagger disagrees, relabel that region (Step 1) and
re-run 2→4. This refinement loop is the point of the tool.

## Output columns (downstream compatibility)

`decode_session_states` / `tag_sessions.py` emit, per trial:

- `state` (int argmax), `state_label` (str), `state_confidence` (max class prob),
  `state_gated` (−1 below the confidence threshold), `p_state_{k}`.
- `hmm_state` / `hmm_state_label` / `hmm_state_gated` — aliases so the frame is drop-in
  for `hmm_downstream` (which reads `hmm_state`).
- `is_hit` / `is_fa` / `is_miss` / `is_go` / `is_catch` / `outcome` / `session_name` —
  so `hmm_downstream.compute_state_behavioral_metrics` and `compute_learning_trajectory`
  run unchanged on a concatenation of the per-session tag CSVs.

> Note: `p_state_{k}` is keyed to `tree.classes_[k]` (alphabetical class order), which is
> **not** the HMM's Viterbi-state numbering. Use the string `state_label` /
> `hmm_state_label` columns as the stable interface.

## Common flags

| Flag | Scripts | Default |
|------|---------|---------|
| `--labels` | label, calibrate, validate | `data/state_labels/state_episodes.csv` |
| `--out-model` / `--model` | calibrate / tag, validate | `data/state_labels/state_rule.pkl` |
| `--out-rules` | calibrate | `data/state_labels/rules.md` |
| `--out-dir` | tag | `data/cache/state_tags` |
| `--fig-dir` | validate | `figures/state_labeler` |
| `--confidence` | tag | `0.8` |
| `--seed` | calibrate | `42` |
| `--labeler` | label | `$USERNAME` |

## Caveats

- The engine is unit-tested, but the learned **rule is only as good as your labels** —
  nothing about real BG_046 behavior is validated until you label and check κ.
- To use this from another branch/worktree or `main`, merge or push
  `feature/behavioral-state-labeler` first.
