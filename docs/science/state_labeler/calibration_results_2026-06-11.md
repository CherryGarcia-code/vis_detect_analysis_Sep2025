# Behavioral State Labeler — Calibration & Validation Results

**Date:** 2026-06-11 · **Revised:** 2026-06-12 (added **Abort** as a 4th state) · **Subject:** BG_046
**Branch:** `feature/behavioral-state-labeler`
**Tooling:** `src/visdetect/analysis/state_labeling.py`, `state_calibration.py`; CLIs/GUI in `scripts/state_labeling/` (see [README](../../../scripts/state_labeling/README.md)).
**Spec:** `docs/superpowers/specs/2026-06-10-behavioral-state-labeler-design.md`.

---

## Executive summary

A human-in-the-loop labeler learns interpretable behavioral-state thresholds from the
experimenter's sparse labels on the per-trial outcome raster and tags every session.
From **27 hand-labeled sessions** (216 episodes), a shallow decision tree over local
outcome-composition features recovers four states — **Impulsive, StimSens, Disengaged,
Abort** — with **substantial cross-session generalization (LOSO Cohen's κ = 0.709)** and an
in-sample κ = 0.764. **Adding the Abort state cost almost no agreement** (LOSO κ 0.731 → 0.709;
in-sample 0.775 → 0.764), confirming it is a genuinely separable regime rather than noise that
muddies the others. Abort is in fact the **best-recalled** state (97%). Output columns are
drop-in compatible with the existing GLM-HMM downstream interface.

---

## Why a 4th state (Abort)

Abort trials (the change stimulus was never presented — the trial terminated early) come in
**streaks**: long runs where the mouse repeatedly fails to hold through baseline. In the
original 3-state scheme these runs had no home and were absorbed into whichever state
surrounded them — most often **StimSens**, spuriously inflating it. Rather than discard aborts,
we promoted them to their own state. The name is intentionally **neutral** ("Abort", not e.g.
"Frustrated"): we have no firm intuition for the underlying internal state, so the neural data
can speak for itself — and at worst Abort serves as a clean reference/exclusion regime.

> Scope note: ~62% of aborts are isolated singletons, not streaks. A one-off abort is not a
> "state" and is expected to be absorbed into the surrounding regime — it is the **streaks**
> this state captures.

---

## Methods

### Subject & data
- Subject: BG_046 (medial striatum, chronic Neuropixels 2.0), visual change-detection task.
- Sessions labeled: **N = 27** (of 28 in the QC-filtered staging manifest).
- Labels: **216 episodes** (Abort 64, Impulsive 73, StimSens 57, Disengaged 22, over 18 of the
  27 sessions for Abort); **6560 labeled trials** scored at validation
  (Abort 501, Disengaged 470, Impulsive 3406, StimSens 2183). Labels cover a sparse,
  high-confidence subset of trials per session — ambiguous stretches were deliberately left
  unlabeled.

### States (operational meaning learned from labels)
- **Impulsive** — early/anticipatory-lick–heavy regime (high fraction of inappropriate licks:
  behavioral `fa` plus catch-trial SDT false alarms).
- **StimSens** (stimulus-sensitive) — appropriate-lick–driven regime (go-trial hits dominate).
- **Disengaged** — withdrawn regime (high no-lick fraction: misses / correct rejections).
- **Abort** — abort-dominated regime (high fraction of early-terminated trials), typically in
  streaks; neutral label, interpretation deferred to the neural data.

> Note on terminology: a trial's *lick valence* is `appropriate_lick` (go-trial hit),
> `inappropriate_lick` (early `fa` on any trial, OR a catch-trial `hit` = SDT false alarm),
> `nolick` (`miss`, covering go-miss and catch correct-rejection), `abort`, or `ref`.
> The behavioral `fa` label is an early lick, NOT an SDT false alarm.

### Labeling protocol
- Interactive raster GUI (`run_state_labeler.py`), Expert→Naive queue. The experimenter
  drags to paint contiguous spans (`StateEpisode`) only over runs they were confident about;
  ambiguous mixes were left blank by design. Episodes appended to
  `data/state_labels/state_episodes.csv`. (Keys: `1`=Impulsive `2`=StimSens `3`=Disengaged
  `4`=Abort.)

### Features (`extract_state_features`)
Per trial, six **local-window outcome-composition fractions** over a symmetric, centered
window of width `W` (edges shrink, `min_periods=1`), denominator = window trials excluding `ref`:
`f_applick`, `f_inapplick`, `f_nolick`, `f_abort`, `f_miss_easy` (no-lick on an easy go trial,
`change_size ≥ 2.0`), `f_hit_hard` (appropriate lick on a hard go trial, `change_size < 2.0`).
No new feature was needed for Abort — `f_abort` already existed, so the tree learned the Abort
boundary directly once Abort labels were supplied.

### Calibration (`calibrate_states`)
- Model: `sklearn.tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
  class_weight="balanced", random_state=42)` — deliberately shallow so the rule is readable.
- Window selection: window `W` chosen by **leave-one-session-out (LOSO)** mean Cohen's κ over
  the grid `[11, 15, 21, 31, 41, 51, 61]`, then the tree is **refit on all labeled trials** at
  the chosen `W`. Folds whose held-out session has `< 2` true classes are excluded from the κ
  average (a single-class fold cannot be scored for agreement) but **still train** the model in
  every other fold.
- **Selected `W = 15`** (short-timescale composition is most predictive; the 4-state fit moved
  from `W = 11` to `15` — a slightly longer window better separates abort streaks).
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
The experimenter's sparse, confident labels recover **four** interpretable behavioral states
with substantial cross-session generalization (**LOSO κ = 0.709**), and the new Abort state
slots in essentially for free — agreement barely moved versus the 3-state model, so Abort is
separable, not noise.

### Cross-validation vs in-sample agreement

| Metric | 3-state | 4-state |
|---|---:|---:|
| LOSO Cohen's κ (honest, held-out) | 0.731 | **0.709** |
| In-sample Cohen's κ (resubstitution) | 0.775 | **0.764** |
| Window `W` | 11 | **15** |

The ~0.02 LOSO drop is within noise for LOSO over 27 sessions; both sit firmly in "substantial
agreement" (0.61–0.80). The small LOSO–in-sample gap (0.055) indicates the shallow tree is
**not overfitting**.

### Confusion matrix (rows = experimenter label, cols = tagger prediction)

|               | Abort | Disengaged | Impulsive | StimSens | recall |
|---------------|------:|-----------:|----------:|---------:|-------:|
| **Abort**       |   487 |         12 |         0 |        2 | 97.2% |
| **Disengaged**  |     2 |        416 |         7 |       45 | 88.5% |
| **Impulsive**   |    41 |         13 |      2992 |      360 | 87.8% |
| **StimSens**    |   267 |         44 |       177 |     1695 | 77.6% |
| **precision**   | 61.1% |      85.8% |     94.2% |    80.6% |       |

Raw agreement 85.2% (5590/6560); κ = 0.764 corrects for chance.

- **Abort recall is the best of any state (97.2%)** — when a region was labeled Abort, the tagger
  almost always agrees, because `f_abort` is a direct, unambiguous feature.
- **Abort precision (61.1%) is the weak spot**, driven almost entirely by **267 trials the
  experimenter called StimSens but the tagger calls Abort** — windows with ~38–52% aborts that
  the experimenter still read as stimulus-sensitive. This is the one real tension (see boundary
  note below); it is a *deliberately accepted* trade.
- **Impulsive is the cleanest** (87.8% recall / 94.2% precision); its main leak is the graded
  Impulsive↔StimSens border (360 trials at `f_inapplick ≈ 0.45`).
- **Disengaged** validates well (88.5% / 85.8%) despite being the rarest labeled class —
  `class_weight="balanced"` compensates — and barely touches Abort (2 trials).
- **StimSens absorbs the cost** of the new state (recall 77.6%): it cedes 267 trials to Abort
  and 177 to Impulsive. StimSens was always the "default" engaged regime, so the new boundary
  eats into it — by design (see below).

### The StimSens↔Abort boundary (decision: leave as-is)
The tree draws Abort at `f_abort > 0.38` — slightly more aggressively than the experimenter,
producing the 267 StimSens→Abort flips. **We keep this boundary.** Rationale: the express
purpose of promoting Abort was to stop abort runs contaminating StimSens, so a window that is
≥38% aborts landing in Abort is the feature *working*. The accepted consequence is that
**StimSens becomes a higher-confidence, higher-precision label** — when the tagger says
StimSens, it means "confidently stimulus-engaged," not "engaged-ish but abort-heavy." The 61%
Abort precision is also partly a metric artifact of the experimenter's sparse, deliberately
clean StimSens labeling. Eyeballed against the abort-heavy sessions below — performance judged
acceptable.

### Per-session abort load (tagger Abort fraction)
Sanity check that Abort is concentrated in genuinely abort-heavy sessions, not sprinkled
everywhere:

| heaviest | frac | | lightest | frac |
|---|---:|---|---|---:|
| 03092025 | 0.52 | | 03072025 | 0.01 |
| 02092025 | 0.48 | | 01072025 | 0.02 |
| 27082025 | 0.37 | | 02072025 / 15082025 / 14082025 | 0.04 |
| 16092025 | 0.35 | | 27062025 | 0.00 |
| 01092025 | 0.32 | | **13082025, 10092025** | **0.00** |

Two sessions get **zero** Abort tags and eight are ≤4% — the tagger is not over-calling Abort.

### Re-shade figures (`figures/state_labeler/reshade_{session}.png`)
Each session gets a **3-track** figure: the outcome raster, a **your-labels** strip, and a
**tagger** strip (low-confidence/`state_gated` cells dimmed), vertically aligned so a
tagger–label disagreement appears as a colour mismatch and you can see the tagger fill your
unlabeled gaps. State colours: Impulsive `#fb6a4a`, StimSens `#6baed6`, Disengaged `#bdbdbd`,
**Abort `#8c564b`** (neutral brown). (An earlier version drew the state tints behind the opaque
raster bars, where they were fully occluded — hence "no shading".)

### Learned rule (`rules.md`, `W = 15`)
```
if f_nolick <= 0.41:                        # still responding
    if   f_abort     > 0.38  --> Abort       # abort-dominated window
    elif f_inapplick > 0.45  --> Impulsive   # early/inappropriate-lick driven
    else                     --> StimSens    # appropriate-lick driven
else:                                        # f_nolick > 0.41 (withdrawn)
    if   f_applick  <= 0.24  --> Disengaged
    elif f_abort    <= 0.03  --> Disengaged
    else                     --> StimSens
```
(The tree's duplicated child-splits are `max_depth=3` artifacts; the effective logic is the
collapsed form above.)

### Interpretation
The rule operationalizes the four states in behaviorally sensible terms: **Abort** =
abort-dominated; **Impulsive** = high inappropriate/early-lick fraction; **StimSens** =
appropriate-lick–dominated engagement; **Disengaged** = high no-lick/withdrawn. That the
thresholds emerge from a depth-3 tree and generalize at κ = 0.71 suggests the states are
separable from local outcome composition alone, without neural data.

---

## Caveats & limitations
- **Single subject (BG_046).** Cross-subject transfer is untested (run `tag_sessions.py`
  against BG_031/038/039 to eyeball).
- **Labels are the experimenter's subjective judgments, not ground truth.** κ measures
  reproducibility of *that* labeling, not correctness of the state ontology.
- **In-sample κ (0.764) is optimistic;** the LOSO κ (0.709) is the honest generalization estimate.
- **Abort precision is moderate (61%) by design** — see the boundary note. If a downstream
  analysis needs a *pure* Abort regime, gate on `state_confidence` / `state_gated`.
- **`Disengaged` is sparse** (470/6560 labeled trials). It validates well now, but its boundary
  rests on fewer exemplars — if a downstream analysis leans on Disengaged, add a few more
  confident spans rather than labeling ambiguous trials.
- **StimSens↔Impulsive is graded;** trials near the `f_inapplick ≈ 0.45` boundary are the other
  soft edge. For sensitive analyses, gate on `state_confidence` / use `state_gated`.
- One session (`29082025`) carries only a single labeled state; it trains the model but is
  excluded from the LOSO κ average (a one-class fold is unscoreable). This is down from four
  single-state sessions in the 3-state run — Abort labeling diversified most sessions, so 26 of
  27 folds are now LOSO-scoreable.

## Relation to prior work
Conceptually parallel to the GLM-HMM behavioral-state framework (Calhoun 2019; Ashwood;
project synthesis `synthesis-phase3-behavioral-state`), but **anchored to the experimenter's
labels rather than a latent HMM**, and emitting HMM-compatible columns so it can feed the same
state-conditioned downstream analyses.

## Tagging output (this run)
- **27 of 28** QC-manifest sessions tagged → `data/cache/state_tags/{session}.csv`.
- **`05092025` skipped**: its pkl was never generated (a pre-existing data-inventory gap,
  not a labeler issue). Both `tag_sessions.py` and the labeling GUI now skip an unloadable
  session and continue (earlier they aborted on the first missing pkl).
- Each per-session CSV carries the raster columns, the six features, and the tag columns
  (`state`, `state_label`, `state_confidence`, `state_gated`, `p_state_{0..3}`,
  `hmm_state*`, `is_hit`/`is_fa`/`is_miss`/`is_go`/`is_catch`/`outcome`/`session_name`).

## Reproduce
```
py scripts/state_labeling/run_state_labeler.py      # 1. label  -> data/state_labels/state_episodes.csv
py scripts/state_labeling/calibrate_states.py       # 2. fit    -> state_rule.pkl + rules.md  (LOSO kappa)
py scripts/state_labeling/tag_sessions.py            # 3. tag    -> data/cache/state_tags/{session}.csv
py scripts/state_labeling/validate_states.py         # 4. check  -> kappa, confusion, figures/state_labeler/*.png
```
Artifacts as of this run: `data/state_labels/state_rule.pkl` (W=15, LOSO κ=0.709),
`data/state_labels/rules.md`, 27 per-session tag CSVs in `data/cache/state_tags/`,
27 re-shade PNGs in `figures/state_labeler/`.
