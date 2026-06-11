# Behavioral State Labeler — Calibration & Validation Results

**Date:** 2026-06-11 · **Subject:** BG_046 · **Branch:** `feature/behavioral-state-labeler`
**Tooling:** `src/visdetect/analysis/state_labeling.py`, `state_calibration.py`; CLIs/GUI in `scripts/state_labeling/` (see [README](../../../scripts/state_labeling/README.md)).
**Spec:** `docs/superpowers/specs/2026-06-10-behavioral-state-labeler-design.md`.

---

## Executive summary

A human-in-the-loop labeler learns interpretable behavioral-state thresholds from the
experimenter's sparse labels on the per-trial outcome raster and tags every session.
From **27 hand-labeled sessions** (152 episodes), a shallow decision tree over local
outcome-composition features recovers three states — **Impulsive, StimSens, Disengaged** —
with **substantial cross-session generalization (LOSO Cohen's κ = 0.731)** and an
in-sample κ = 0.775. The rare **Disengaged** state is recovered well (88% recall) despite
sparse labels; the only material confusion is the intrinsically graded **StimSens↔Impulsive**
boundary. Output columns are drop-in compatible with the existing GLM-HMM downstream
interface.

---

## Methods

### Subject & data
- Subject: BG_046 (medial striatum, chronic Neuropixels 2.0), visual change-detection task.
- Sessions labeled: **N = 27** (of 28 in the QC-filtered staging manifest).
- Labels: **152 episodes**; **6063 labeled trials** scored at validation
  (Disengaged 470, Impulsive 3410, StimSens 2183). Labels cover a sparse, high-confidence
  subset of trials per session — ambiguous stretches were deliberately left unlabeled.

### States (operational meaning learned from labels)
- **Impulsive** — early/anticipatory-lick–heavy regime (high fraction of inappropriate licks:
  behavioral `fa` plus catch-trial SDT false alarms).
- **StimSens** (stimulus-sensitive) — appropriate-lick–driven regime (go-trial hits dominate).
- **Disengaged** — withdrawn regime (high no-lick fraction: misses / correct rejections).

> Note on terminology: a trial's *lick valence* is `appropriate_lick` (go-trial hit),
> `inappropriate_lick` (early `fa` on any trial, OR a catch-trial `hit` = SDT false alarm),
> `nolick` (`miss`, covering go-miss and catch correct-rejection), `abort`, or `ref`.
> The behavioral `fa` label is an early lick, NOT an SDT false alarm.

### Labeling protocol
- Interactive raster GUI (`run_state_labeler.py`), Expert→Naive queue. The experimenter
  drags to paint contiguous spans (`StateEpisode`) only over runs they were confident about;
  ambiguous mixes were left blank by design. Episodes appended to
  `data/state_labels/state_episodes.csv`.

### Features (`extract_state_features`)
Per trial, six **local-window outcome-composition fractions** over a symmetric, centered
window of width `W` (edges shrink, `min_periods=1`), denominator = window trials excluding `ref`:
`f_applick`, `f_inapplick`, `f_nolick`, `f_abort`, `f_miss_easy` (no-lick on an easy go trial,
`change_size ≥ 2.0`), `f_hit_hard` (appropriate lick on a hard go trial, `change_size < 2.0`).

### Calibration (`calibrate_states`)
- Model: `sklearn.tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
  class_weight="balanced", random_state=42)` — deliberately shallow so the rule is readable.
- Window selection: window `W` chosen by **leave-one-session-out (LOSO)** mean Cohen's κ over
  the grid `[11, 15, 21, 31, 41, 51, 61]`, then the tree is **refit on all labeled trials** at
  the chosen `W`. Folds whose held-out session has `< 2` true classes are excluded from the κ
  average (a single-class fold cannot be scored for agreement) but **still train** the model in
  every other fold.
- **Selected `W = 11`** (shortest window — short-timescale composition is most predictive).
- Constants: `STATE_EASY_CHANGE_THRESH = 2.0`, `STATE_CONFIDENCE_THRESHOLD = 0.8`
  (from `visdetect.analysis.constants`).

### Tagging (`tag_sessions` / `decode_session_states`)
Each trial gets `state` (argmax), `state_label`, `state_confidence` (max class prob),
`state_gated` (−1 when confidence ≤ 0.8), `p_state_{k}`, plus `hmm_state`/`hmm_state_label`/
`hmm_state_gated` aliases and the SDT booleans (`is_hit`/`is_fa`/`is_miss`/`is_go`/`is_catch`/
`outcome`/`session_name`) so the frame is drop-in for `hmm_downstream`. Ambiguous trials are
handled here, not at label time: they receive a state + confidence, and `state_gated`
quarantines the genuinely uncertain ones from downstream neural analyses.

---

## Results

### Key finding
The experimenter's sparse, confident labels are sufficient to recover three interpretable
behavioral states with substantial cross-session generalization (**LOSO κ = 0.731**), with
the learned thresholds matching the intended semantics.

### Cross-validation vs in-sample agreement
- **LOSO Cohen's κ = 0.731** (honest generalization to held-out sessions).
- **In-sample Cohen's κ = 0.775** (resubstitution, all labeled trials, no confidence gating).
- The small LOSO–in-sample gap (0.044) indicates the shallow tree is **not overfitting**.

### Confusion matrix (rows = experimenter label, cols = tagger prediction)

|              | Disengaged | Impulsive | StimSens | recall |
|--------------|-----------:|----------:|---------:|-------:|
| **Disengaged** |        415 |        13 |       42 |  88.3% |
| **Impulsive**  |         16 |      3075 |      319 |  90.2% |
| **StimSens**   |         56 |       304 |     1823 |  83.5% |
| **precision**  |      85.2% |     90.7% |    83.5% |        |

Raw agreement 87.6% (5313/6063); κ = 0.775 corrects for chance.

- **Disengaged is recovered well** (88.3% recall / 85.2% precision) despite being the rarest
  class — `class_weight="balanced"` compensates for the imbalance.
- The dominant error is **StimSens↔Impulsive** (304 + 319 trials), an intrinsically graded
  boundary split at `f_inapplick ≈ 0.44`; both are "engaged" regimes differing in early-lick rate.
- **Impulsive↔Disengaged** barely confuse (16 / 13) — cleanly separated.

### Re-shade figures (`figures/state_labeler/reshade_{session}.png`)
Each session gets a **3-track** figure: the outcome raster, a **your-labels** strip, and a
**tagger** strip (low-confidence/`state_gated` cells dimmed), vertically aligned so a
tagger–label disagreement appears as a colour mismatch and you can see the tagger fill your
unlabeled gaps. State colours reuse the HMM palette (Impulsive `#fb6a4a`, StimSens `#6baed6`,
Disengaged `#bdbdbd`) — distinct from the lick-valence raster. (An earlier version drew the
state tints behind the opaque raster bars, where they were fully occluded — hence "no shading".)

### Learned rule (`rules.md`, `W = 11`)
```
|--- f_nolick <= 0.39                      # engaged (licking)
|   |--- f_inapplick <= 0.44 --> StimSens  # appropriate-lick driven
|   |--- f_inapplick >  0.44 --> Impulsive # early/inappropriate-lick driven
|--- f_nolick >  0.39                      # withdrawn (not licking)
|   |--- f_applick <= 0.24 --> Disengaged
|   |--- f_applick >  0.24
|   |   |--- f_abort <= 0.05 --> Disengaged
|   |   |--- f_abort >  0.05 --> StimSens
```
(The tree's duplicated child-splits are `max_depth=3` artifacts; the effective logic is the
collapsed form above.)

### Interpretation
The rule operationalizes the three states in behaviorally sensible terms: **Impulsive** =
high inappropriate/early-lick fraction; **StimSens** = appropriate-lick–dominated engagement;
**Disengaged** = high no-lick/withdrawn. That the thresholds emerge from a depth-3 tree and
generalize at κ = 0.73 suggests the states are separable from local outcome composition alone,
without neural data.

---

## Caveats & limitations
- **Single subject (BG_046).** Cross-subject transfer is untested (run `tag_sessions.py`
  against BG_031/038/039 to eyeball).
- **Labels are the experimenter's subjective judgments, not ground truth.** κ measures
  reproducibility of *that* labeling, not correctness of the state ontology.
- **In-sample κ (0.775) is optimistic;** the LOSO κ (0.731) is the honest generalization estimate.
- **`Disengaged` is sparse** (470/6063 labeled trials). It validates well now, but its boundary
  rests on fewer exemplars — if a downstream analysis leans on Disengaged, add a few more
  confident spans rather than labeling ambiguous trials.
- **StimSens↔Impulsive is graded;** ~10% of those trials sit near the `f_inapplick` boundary.
  For sensitive analyses, gate on `state_confidence` / use `state_gated` to drop the boundary
  cases.
- Four sessions (`09072025`, `14082025`, `28082025`, `29082025`) carry only one labeled state;
  they train the model but are excluded from the LOSO κ average (a one-class fold is unscoreable).

## Relation to prior work
Conceptually parallel to the GLM-HMM behavioral-state framework (Calhoun 2019; Ashwood;
project synthesis `synthesis-phase3-behavioral-state`), but **anchored to the experimenter's
labels rather than a latent HMM**, and emitting HMM-compatible columns so it can feed the same
state-conditioned downstream analyses.

## Tagging output (this run)
- **27 of 28** QC-manifest sessions tagged → `data/cache/state_tags/{session}.csv`.
- **`05092025` skipped**: its pkl was never generated (a pre-existing data-inventory gap,
  not a labeler issue). `tag_sessions.py` now skips an unloadable session and continues
  (earlier it aborted the whole batch on the first missing pkl).
- Each per-session CSV carries the raster columns, the six features, and the tag columns
  (`state`, `state_label`, `state_confidence`, `state_gated`, `p_state_{0..2}`,
  `hmm_state*`, `is_hit`/`is_fa`/`is_miss`/`is_go`/`is_catch`/`outcome`/`session_name`).

## Reproduce
```
py scripts/state_labeling/run_state_labeler.py      # 1. label  -> data/state_labels/state_episodes.csv
py scripts/state_labeling/calibrate_states.py       # 2. fit    -> state_rule.pkl + rules.md  (LOSO kappa)
py scripts/state_labeling/tag_sessions.py            # 3. tag    -> data/cache/state_tags/{session}.csv
py scripts/state_labeling/validate_states.py         # 4. check  -> kappa, confusion, figures/state_labeler/*.png
```
Artifacts as of this run: `data/state_labels/state_rule.pkl` (W=11, LOSO κ=0.731),
`data/state_labels/rules.md`, 27 per-session tag CSVs in `data/cache/state_tags/`,
27 re-shade PNGs in `figures/state_labeler/`.
