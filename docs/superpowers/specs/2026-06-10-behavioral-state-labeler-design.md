# Design Spec — User-Defined Behavioral State Labeler

- **Date:** 2026-06-10
- **Status:** Approved (design); ready for implementation plan
- **Author:** b.gonzales@ucl.ac.uk (with Claude Code)
- **Related:** GLM-HMM (`src/visdetect/analysis/hmm.py`); legacy `identify_session_state` (`src/visdetect/analysis/behavior.py`); question-landscape theme **D** (behavioral state) and **B1** (integration timescale as a learned quantity); literature synthesis `synthesis-phase3-behavioral-state` (Ashwood, Calhoun, Miller).
- **Presentation assets:** `docs/science/state_labeler/design_mockups.html` (self-contained; print-to-PDF for slides).

---

## 1 · Motivation

The project currently has two ways to assign behavioral states, and neither matches where the experimenter would draw state boundaries by eye:

1. **GLM-HMM (Ashwood).** States are *latent* and defined only indirectly through GLM weights that maximize trial-by-trial lick likelihood. Boundaries land wherever the likelihood surface puts them; there is no place for the experimenter's judgment. This is the structural reason the inferred states diverge from the experimenter's reading.
2. **`identify_session_state` (legacy).** A rolling-rate threshold classifier (30-trial window; FA > 0.48 → impulsive, miss > 0.35 → disengaged, else balanced). Closer in spirit, but the window and cutoffs are arbitrary (quantile-derived), not calibrated to the experimenter.

We want a **data-driven, human-in-the-loop** system: the experimenter marks the states they are confident about by reading the trial-by-trial outcome raster; we then **reverse-engineer the interpretable thresholds** that reproduce that judgment, and build a **reliable tagger** that applies it across all sessions (and ideally other subjects), validated against the experimenter's own labels.

This complements rather than deletes the GLM-HMM — the HMM remains as a comparison baseline.

## 2 · Goals / Non-goals

**Goals**
- Let the experimenter sparsely label confident behavioral-state episodes on the outcome raster.
- Learn an **interpretable threshold rule** (human-readable cutoffs + confidence) that reproduces those labels.
- Tag every trial of every session with a state + confidence, in a form **drop-in compatible** with the existing GLM-HMM downstream interface.
- Be **subject-agnostic**: calibrate on BG_046, test transfer to BG_031/038/039.
- Quantify agreement with the experimenter's labels, with the GLM-HMM, and with the legacy classifier.

**Non-goals (this iteration)**
- Not replacing/removing the GLM-HMM from the codebase.
- Not online/real-time tagging (offline, acausal windows are fine).
- Not pooled multi-subject calibration up front (only as a fallback if transfer fails).
- Not a soft probabilistic classifier as the primary rule (kept as a documented comparison only).

## 3 · Locked design decisions

| # | Decision | Choice |
|---|---|---|
| Signal | What the experimenter reads | Trial-by-trial **outcome raster** (local composition), not smoothed rate curves |
| Vocabulary | State label set | Draft, refinable: **Impulsive / StimSens / Disengaged** |
| Data model | What a label commits to | **Sparse confident episodes**; ambiguous trials left unlabeled |
| Color | Tick color semantics | **Lick-valence**: appropriate lick = green, inappropriate lick (early lick **+** catch SDT-FA) = red, no-lick (miss **+** correct rejection) = purple, abort = grey |
| Display | Raster encoding | Go vs catch **distinguished** (outline + ▾); rolling-rate overlay **off** by default (avoids circularity); optional **change-size shading** toggle (off by default) |
| Calibration | Rule form | **Shallow decision tree** (+ leaf-probability confidence); multinomial-logistic kept as comparison |
| Features | Difficulty | Difficulty-aware: go outcomes split easy (`change_size ≥ 2.0`) vs hard |
| Window | `W` | **Single fitted value** (detection resolution, *not* a cap on state duration), grid-searched by LOSO agreement; per-stage re-fit as a validation check |
| Cross-subject | Strategy | **Calibrate on BG_046, test transfer**; pool only if transfer fails |
| Order | Labeling order | **Expert → Naive** (states clearest where formed) |
| Output | Tag columns | **`hmm.decode_session`-compatible** columns so downstream is drop-in |

## 4 · Architecture

Four-stage human-in-the-loop pipeline (see `design_mockups.html` §0 for the data-flow diagram). Mirrors the existing **TF Manual Labeling System** (`src/visdetect/analysis/tf_labeling.py` + `scripts/tf_labeling/run_labeling_gui.py`) for architectural consistency.

```
Sessions (.pkl, Expert→Naive)
   ├─(you)→ [1 Labeling GUI] → StateEpisodes.csv
   └──────→ [2 Feature extractor: local-window lick-valence + difficulty fractions, window W]
StateEpisodes.csv (labels) + labeled-trial features → [3 Calibration: decision tree + fit W by LOSO] → Rule (thresholds + model + confidence)
Rule + all-trial features → [4 Tagger: per-trial state + confidence] → decode_session-compatible columns → per-session tag cache
Tagger output + your labels → [5 Validation: κ/confusion · re-shade review · vs GLM-HMM · cross-subject] ─(dashed)→ relabel loop
state column (gated by confidence) → downstream neural analyses
```

## 5 · Component 1 — Labeling tool

### 5.1 Library: `src/visdetect/analysis/state_labeling.py`
- **`StateEpisode`** dataclass: `session_name: str`, `start_trial: int`, `end_trial: int` (inclusive, indices into the filtered trial DataFrame), `state_label: str`, `labeler: str`, `timestamp: str`, `notes: str = ""`.
- **`save_episode(episode, path)` / `load_episodes(path) -> list[StateEpisode]`** — persisted to `data/state_labels/state_episodes.csv` (append-only; one row per episode). Round-trippable.
- **`episodes_to_trial_labels(episodes, n_trials) -> np.ndarray[object]`** — expand sparse episodes to a per-trial label array (unlabeled = `None`).
- **`get_labeling_queue(qc_only=True) -> list[str]`** — session names ordered **Expert → Naive**, reusing `load_staging_manifest()` + `chronological_sort()`/`parse_session_date()`. Within a stage, reverse-chronological.
- **`build_outcome_raster(session) -> pd.DataFrame`** — per-trial frame with columns: `trial_idx`, `outcome`, `is_go`, `is_catch`, `change_size`, `lick_valence` (one of `appropriate_lick`/`inappropriate_lick`/`nolick`/`abort`/`ref`), `color`. Built on `behavior.get_trial_dataframe()`.

### 5.2 Lick-valence classification (single source of truth)
Given `outcome` + `is_go`/`is_catch`:
- `appropriate_lick` ← `is_go AND outcome == 'hit'` → green
- `inappropriate_lick` ← `outcome == 'fa'` (early lick, any trial type) OR (`is_catch AND outcome == 'hit'`) (SDT false alarm) → red
- `nolick` ← `outcome == 'miss'` (covers go-miss **and** catch correct-rejection) → purple
- `abort` ← `outcome == 'abort'` → grey
- `ref` ← `outcome == 'ref'` (reflex lick) → minor class, **excluded from the primary fractions** by default (configurable); rendered in a muted color.

Colors live in `config` (e.g. `LICK_VALENCE_COLORS`), not hardcoded in the GUI.

### 5.3 GUI: `scripts/state_labeling/run_state_labeler.py`
matplotlib **TkAgg**, keyboard-driven (mirrors `run_labeling_gui.py`).
- Renders the raster: one tick per (filtered) trial, colored by `lick_valence`; catch trials get an outline + ▾ notch.
- **Drag** across trials to paint an episode; number keys **1**=Impulsive **2**=StimSens **3**=Disengaged, **n**=new label (prompt), **backspace**=erase the span under cursor; only painted trials are saved.
- **←/→** navigate the Expert→Naive queue; header shows `subject · session · stage · queue position`.
- **r** = toggle rolling-rate overlay (default **off**); **c** = toggle change-size shading (go-hits → 5 greens, genuine go-misses → 5 purples by `change_size`, bigger = more opaque; default **off**).
- Autosaves episodes to `data/state_labels/state_episodes.csv`.

## 6 · Component 2 — Feature extraction

`extract_state_features(raster_df, W) -> pd.DataFrame` (one row per trial).

For each trial *t*, over a **symmetric window** of `W` trials centered on *t* (acausal; edges shrink with `min_periods`), compute mutually-exclusive composition fractions (denominator = window trials excluding `ref`, so `f_applick + f_inapplick + f_nolick + f_abort = 1`) plus difficulty splits:
- `f_applick`, `f_inapplick`, `f_nolick`, `f_abort` (primary lick-valence fractions; `ref` excluded from denominator by default)
- Difficulty-aware (easy = `change_size ≥ STATE_EASY_CHANGE_THRESH = 2.0`): `f_miss_easy` (missed an *obvious* change — key disengagement signal), `f_hit_hard` (caught a *hard* change — key sensitivity signal); optionally `f_miss_hard`, `f_hit_easy`.

Fractions are bounded in [0,1] → **no z-scoring** needed and inherently subject-agnostic. `W` is a hyperparameter fit in calibration.

## 7 · Component 3 — Calibration

`calibrate_states(episodes, sessions, W_grid, seed=42) -> CalibrationResult`.
- **Training set:** only trials inside painted episodes (features from §6, label from the episode). Unlabeled trials excluded.
- **Model:** `sklearn.tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight='balanced', random_state=seed)` (`min_samples_leaf` tunable). Depth kept shallow for readability.
- **Fit `W`:** grid-search `W ∈ {11,15,21,31,41,51,61}` (odd, symmetric), selecting the `W` that maximizes **leave-one-session-out** agreement (Cohen's κ) on held-out labeled episodes.
- **Outputs (`CalibrationResult`):** fitted tree, chosen `W`, exported **human-readable rules** (`sklearn.tree.export_text` → `rules.md`), per-class confidence (leaf probabilities), and the LOSO κ. Model saved via pickle.
- **Comparison (deferred Option 2):** a `LogisticRegression(multi_class='multinomial')` fit on the same features; report its LOSO κ alongside the tree's. Not used for tagging.

## 8 · Component 4 — Tagger + downstream integration

`decode_session_states(model, session, W, state_labels, confidence_threshold=0.8) -> pd.DataFrame` — **mirrors `hmm.decode_session`'s signature and columns** so existing downstream code is drop-in:
- `state` (int), `state_label` (str), `p_state_0 … p_state_{K-1}` (leaf probabilities), `state_confidence` (max prob), `state_gated` (−1 where `state_confidence < confidence_threshold`, paralleling `hmm.assign_states_with_confidence`).
- Per-session tags cached to `data/cache/state_tags/<session>.csv`.
- A batch driver tags all manifest sessions.

## 9 · Component 5 — Validation & refinement loop

1. **Agreement vs experimenter labels:** LOSO cross-validated accuracy + **Cohen's κ** + confusion matrix on labeled episodes (consult Research Statistician for reporting).
2. **Re-shade review:** re-render each session shaded by the tagger with the experimenter's painted episodes overlaid (reuse `scripts/analysis/behavior/plot_session_behavior.py` style). The experimenter eyeballs disagreements and relabels — closes the dashed loop and is how unlabeled regions get revisited.
3. **vs GLM-HMM and vs legacy `identify_session_state`:** agreement matrices; quantify that the new rule matches the experimenter's eye better.
4. **Cross-subject transfer:** apply the BG_046 model to BG_031/038/039, re-shade, sanity-check; pooled re-calibration only if transfer fails.
5. **Per-stage W test:** re-fit `W` separately per stage; report whether the optimal timescale grows Naive→Expert (a substantive finding tied to question **B1**).

## 10 · Integration, reuse, constants

- **Reuse (search-before-writing):** `behavior.get_trial_dataframe`, `behavior.compute_rolling_performance` (overlay), `config.load_staging_manifest`, `chronological_sort`/`parse_session_date`, `utils.synthetic.make_synthetic_session` (tests), `viz.plotting.set_style`.
- **New constants** in `src/visdetect/analysis/constants.py`: `STATE_LABEL_W_GRID`, `STATE_LABEL_W_DEFAULT`, `STATE_CONFIDENCE_THRESHOLD = 0.8`, `STATE_EASY_CHANGE_THRESH = 2.0`. Reuse `CHANGE_SIZES`. Add `LICK_VALENCE_COLORS` / `STATE_COLORS` to `config`.
- **Do not duplicate** `identify_session_state` — supersede but retain for comparison.

## 11 · File layout

```
src/visdetect/analysis/state_labeling.py     # data model, raster, queue, lick-valence
src/visdetect/analysis/state_calibration.py  # features, calibrate_states, decode_session_states
scripts/state_labeling/run_state_labeler.py  # matplotlib GUI
scripts/state_labeling/calibrate_states.py   # CLI: fit rule + export rules.md
scripts/state_labeling/tag_sessions.py       # CLI: batch tagger → tag cache
scripts/state_labeling/validate_states.py    # CLI: κ/confusion, re-shade, vs HMM, transfer
data/state_labels/state_episodes.csv         # experimenter labels (append-only)
data/cache/state_tags/<session>.csv          # per-session tags
docs/science/state_labeler/                   # presentation deck (committed)
```

## 12 · Testing strategy (TDD)

Unit tests (synthetic sessions via `make_synthetic_session`):
- Lick-valence mapping: catch-`hit` → `inappropriate_lick`; catch-`miss` → `nolick`; go-`hit` → `appropriate_lick`; `fa` → `inappropriate_lick`.
- Feature extraction: a planted-composition window yields the expected fractions; edge windows shrink correctly.
- `StateEpisode` save/load round-trip; `episodes_to_trial_labels` expansion.
- Queue ordering is Expert→Naive.
- Calibration is deterministic given `seed`; recovers a planted threshold on synthetic labels; W grid-search recovers a planted scale.
- Tagger output schema matches the spec; confidence gating sets −1 below threshold; columns parallel `decode_session`.

## 13 · Literature grounding

Per `synthesis-phase3-behavioral-state`: behavioral states are different **cue→action mappings**, not just different action rates (Calhoun 2019; Ashwood 2022). Our difficulty-aware features operationalize exactly this — StimSens is defined by *responding to changes (especially hard ones)*, Impulsive by *licking regardless of stimulus*, Disengaged by *not licking even to obvious changes*. Miller 2022 ("variance is a feature") motivates the per-stage-W and slow-drift checks. This supervised, experimenter-anchored approach is complementary to the unsupervised GLM-HMM.

## 14 · Risks & open questions

- **`ref` (reflex licks):** excluded from primary fractions by default; revisit if they cluster with impulsive episodes.
- **Few labels / class imbalance:** sparse labeling may yield few exemplars per state; `class_weight='balanced'` + shallow depth + LOSO mitigate; report minimum-episode sensitivity.
- **Boundary trials:** sparse model means episode edges are the experimenter's confident cores, not exact changepoints — acceptable by design; the re-shade loop catches systematic edge errors.
- **Transfer:** if BG_046-calibrated thresholds fail on other subjects, escalate to pooled calibration (documented fallback).

## 15 · Build sequencing

Implementation on a dedicated feature branch (e.g. `feature/behavioral-state-labeler`), TDD throughout. Suggested order: (1) `state_labeling.py` lib + tests → (2) GUI → (3) feature extraction + calibration + tests → (4) tagger + downstream columns + tests → (5) validation/CLIs + figures. Detailed steps to be produced by the writing-plans skill.
