# Behavioral State Labeler

A human-in-the-loop tool that learns **interpretable** behavioral-state thresholds
from your sparse labels on the outcome raster, then tags every session in a form
**drop-in compatible** with the existing GLM-HMM downstream interface.

States: `Impulsive`, `StimSens`, `Disengaged`, `Abort` (see `STATE_LABELS` in
`visdetect.analysis.constants`).

All outputs are **subject-scoped**: the active subject comes from the
`VISDETECT_SUBJECT` env var (default `BG_046`), and tags/figures nest under
`…/state_tags/{SUBJECT}/` and `…/state_labeler/{SUBJECT}/` so multi-subject runs
never collide. See [Cross-subject tagging](#cross-subject-tagging).

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
| green   | appropriate lick — go-trial hit |
| coral   | inappropriate lick — early lick, or catch-trial false alarm |
| lavender| no-lick — miss or correct rejection |
| grey    | abort |
| tan     | reflex lick (excluded from feature fractions) |

(Softened palette; an **outcome legend** is drawn to the left of the raster.)
Catch trials are outlined in black. Press `c` to shade go-trial hits/misses by
change-size difficulty.

**Keys:** `1`=Impulsive · `2`=StimSens · `3`=Disengaged · `4`=Abort (active label) ·
**drag = paint a span** (saved on release) · `c`=toggle difficulty shading ·
`←`/`→`=prev/next session · `q`=quit. (Number keys derive from `STATE_LABELS`.)
The GUI shows a live **"your labels"** strip under the raster so prior spans
appear on revisit.

Spans append to `data/state_labels/state_episodes.csv` (git-diffable; hand-edit to fix a
mislabel). **Label ≥ 2 sessions before calibrating** — cross-validation needs more than one.
To review what you've already labeled, use the Step-4 re-shade figures (the "your labels"
strip), or inspect the CSV directly.

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

Tags sessions → one CSV per session in `data/cache/state_tags/{SUBJECT}/{session}.csv`,
plus `_tag_summary.csv` (state occupancy + mean outcome composition per tagged state).
Session source: `--sessions` if given, else the staging manifest (BG_046), else every
pkl on disk. Flags: `--limit N` (evenly-spread subset), `--figures` (also write a
2-track raster+tagger PNG per session), `--sessions A B C`.

### 4. Validate & refine — `validate_states.py`

```
py scripts/state_labeling/validate_states.py
```

Prints Cohen's κ + a confusion matrix (tagger vs your labels) and saves a **3-track
re-shade PNG** per session to `figures/state_labeler/{SUBJECT}/`: the outcome raster on
top (with the outcome legend at left), a **your-labels** strip, and a **tagger** strip,
all vertically aligned so disagreements show as a colour mismatch between the two strips.
State colours (warm→cool arousal ramp): Impulsive `#ef6548`, StimSens `#6baed6`,
Disengaged `#3474ae`, Abort `#bdbdbd`. Low-confidence/`state_gated` cells are dimmed
(`--confidence` sets the threshold; an italic caption explains the fade — no grey swatch,
since grey now means Abort). Where the tagger disagrees, relabel that region (Step 1) and
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
| `--out-dir` | tag | `data/cache/state_tags/{SUBJECT}` |
| `--fig-dir` | tag, validate | `figures/state_labeler/{SUBJECT}` |
| `--confidence` | tag, validate | `0.8` |
| `--limit` / `--sessions` / `--figures` | tag | — |
| `--seed` | calibrate | `42` |
| `--labeler` | label | `$USERNAME` |

## Cross-subject tagging

The rule is subject-agnostic (a decision tree over outcome-composition fractions), so
you can apply BG_046's rule to another subject as a **face-validity** check. Set the
subject via the env var; outputs auto-nest under it:

```
VISDETECT_SUBJECT=BG_031 py scripts/state_labeling/tag_sessions.py --limit 5 --figures
```

Subjects without a staging manifest (BG_031/038/039) fall back to enumerating their pkls
on disk (`list_pkl_sessions`). There are **no ground-truth labels** for other subjects, so
`validate_states.py` (κ) doesn't apply — judge by the 2-track figures and `_tag_summary.csv`
(each state should still carry its defining outcome signature: Impulsive→high `f_inapplick`,
Disengaged→high `f_nolick`, Abort→high `f_abort`, StimSens→highest `f_applick`).

## Caveats

- The engine is unit-tested, but the learned **rule is only as good as your labels** —
  nothing about real BG_046 behavior is validated until you label and check κ.
- To use this from another branch/worktree or `main`, merge or push
  `feature/behavioral-state-labeler` first.
