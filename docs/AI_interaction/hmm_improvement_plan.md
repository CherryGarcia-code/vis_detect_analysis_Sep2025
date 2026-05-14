# HMM 2.0 — Improvement Plan

**Created**: May 13, 2026  
**Reviewed by**: Research Statistician, Research Visualizer, Codebase Auditor skills  
**Goal**: Improve the GLM-HMM behavioral state analysis for rigorous multi-session, multi-subject neural analysis across learning.

---

## Scientific Rationale

The current K=3 model (Disengaged 11%, Engaged 33%, Impulsive 56%) was fit jointly on all BG_046 sessions. The heavy Impulsive fraction is dominated by early sessions. Before using HMM states as faceting variables for neural analyses — and before extending to other subjects — we need to validate that the state definitions are stable, scientifically interpretable, and generalizable.

**Two complementary state definitions serve different questions:**

| Model | Fit on | Scientific question answered |
|-------|--------|------------------------------|
| Joint (current) | All sessions | "What behavioral strategies does the animal use throughout learning?" |
| Expert-anchor | Expert sessions only, decoded everywhere | "How does each trial relate to the learned endpoint?" |

---

## Pre-Step: Bug Fixes (identified by Codebase Auditor — fix before running any new fits)

### HIGH — Fix immediately

**BUG 1: `hmm.py:399` — M-step warm-starts ALL states from state-0 weights**  
`w0 = self._weights[0]` passed to `_fit_glm_state` for every k.  
When k=1 is processed, state-0 has already been updated → warm start is the new state-0 solution.  
**Fix**: `w0 = self._weights[k]` (one-line change).  
*Impact*: Affects model quality for K≥3; partially mitigated by 20 restarts but still suboptimal.*

**BUG 2: `b_hmm_state_dynamics.py:120–138` — Panel C crosses session boundaries**  
Transitions loop over stage_trials sorted by session/trial without resetting at session edges.  
Last trial of session N is paired with trial 0 of session N+1 → spurious cross-session transitions inflate self-transitions.  
**Fix**: Add a session-boundary guard in the transition loop.

**BUG 3: `loader.py` — K parameter silently ignored in load_hmm_assignments/per_session/trajectory**  
All three functions read from paths fixed at K=3 regardless of the K argument.  
**Fix**: Parameterize path lookup by K and subject (needed for multi-K and multi-subject work).  
*Can defer to Step 4 since Step 1 uses a new dedicated script, not the loader.*

**BUG 4: `loader.py` — "BG_046" hardcoded in `load_tf_traces_npz`**  
Filename built as `f"BG_046_{sname}..."` instead of using `SUBJECT` constant.  
**Fix**: Replace with `SUBJECT` constant. *(Defer to Step 4.)*

---

## Step 1 — Expert-Only Fit Diagnostic (BG_046) ✦ START HERE

**Script**: `scripts/analysis/behavior/expert_anchor_diagnostic.py`  
**Saves to**: `data/hmm/BG_046/expert_only/` + `FIGURES/behavior/BG_046/hmm/expert_vs_joint_diagnostic.png`

### 1a. Fit Expert-only model
- Filter manifest: `load_staging_manifest(qc_only=True)` → keep `stage == "Expert"` rows
- Run `fit_best_model()` K_range=[2,3,4], 20 restarts — same config as joint model
- Save model + labels to `data/hmm/BG_046/expert_only/`

### 1b. State correspondence
- Compute P(lick) profiles (5 stim values × K states) for both models
- Find best state alignment (K! permutations) maximizing mean cosine similarity
- Report cosine similarity per matched state pair
- Null: cosine similarities of non-matched permutations (descriptive range)

### 1c. Expert-anchor decode of all sessions
- Apply Expert-only model (fixed weights) to ALL sessions via Viterbi + forward-backward
- Compute per-session: mean held-out LL per trial, state fractions, P(Engaged)
- Save full trial-level assignments to `data/hmm/BG_046/expert_only/state_assignments_all_sessions.csv`

### Statistical Tests (from Research Statistician review)

| Comparison | Test | Effect size | Notes |
|-----------|------|-------------|-------|
| ΔBIC, joint vs Expert-only | Kass-Raftery ΔBIC | — | ΔBIC > 10 = very strong evidence |
| Held-out LL: Learning vs Expert | Mann-Whitney U (per-session means) | rank-biserial r | ~12-14 vs ~10-13 sessions; n too small for Kruskal-Wallis |
| P(Engaged) trajectory | Spearman ρ vs session index + bootstrap CI (1000 resamples, seed=42) | ρ | Supplement with Mann-Whitney U (Learning vs Expert) for categorical cross-check |
| State fraction trajectory per state | Mann-Whitney U × 3 (Holm-Bonferroni corrected) | rank-biserial r | Per-session fractions, Learning vs Expert |
| State assignment agreement | Cohen's κ per session, then Mann-Whitney U (Learning vs Expert κ) | rank-biserial r | Quantifies how well Expert model degrades on early sessions |

**Sample-size caveat**: With ~12-14 sessions per stage, 80% power requires effect r ≥ 0.55. Non-significant results should report effect sizes + bootstrap CIs as primary claims, not just p-values.

### Figure layout (from Research Visualizer review)
**GridSpec 3×2, figsize=(12, 13), height_ratios=[1, 1, 1.3]**

| Panel | Row | Col | Content |
|-------|-----|-----|---------|
| A | 0 | 0 | BIC/AIC vs K — joint + Expert-only overlaid |
| D | 0 | 1 | Held-out LL per trial by stage — violin/box, both models |
| B | 1 | 0 | Psychometric curves — solid=joint, dashed=Expert-anchor, same state colors |
| C | 1 | 1 | GLM weight vectors — full-sat=joint, 40% lightness=Expert-anchor bars |
| E | 2 | 0-1 | Learning score P(Engaged\|Expert model) trajectory — full width |

**Color notes:**
- Panel B: Solid lines = joint model, dashed = Expert-anchor. Same `HMM_STATE_COLORS`. Required for colorblind safety (grey/blue similar under deuteranopia).
- Panel C: `HMM_STATE_COLORS` at full saturation for joint; 40% lightness variant for Expert-anchor.
- Panel D: X-axis label says "Learning (incl. Naive)" — `merge_naive_learning=True`.
- Panel E: Engaged line = `#6baed6`; stage fills as `axvspan` at `alpha=0.08` only.

**Save stats to**: `FIGURES/behavior/BG_046/hmm/expert_vs_joint_stats.csv`

---

## Step 2 — Cross-Validation Integration

**Goal**: Surface LOSO CV output from `hmm_downstream.loso_cross_validation()` in the fitting pipeline.

### Changes
- `fit_behavioral_hmm.py`: add `--cv` flag → run LOSO after fitting; save `cv_results_K{n}.csv`
- `loader.py`: add `load_hmm_cv(K=3, subject="BG_046")` loader
- Report: per-stage mean ± SEM test LL per trial; accuracy vs session index

---

## Step 3 — Robust State Labeling (`auto_label_states()`)

**Current fragility**: Absolute P(lick) thresholds (p_catch > 0.65 → "Biased", p_high < 0.40 → "Disengaged") will silently mislabel states when base lick rates differ across subjects.

### Fix (rank-based labeling)
1. States are already sorted by ascending bias (`sort_states_by_bias()`)
2. Compute sensitivity slope: `P(lick | stim=max) − P(lick | catch)` per state
3. Assign by rank:
   - Lowest sensitivity AND lowest baseline → "Disengaged"
   - High baseline (top 1/K by catch P(lick)) → "Impulsive"
   - Remaining (high sensitivity, moderate baseline) → "Engaged"
4. Add `n_states` branching (K=2: Disengaged+Engaged; K=3: standard; K=4+: Engaged_low/Engaged_high split)
5. Add explicit K>3 fallback (currently documented but not implemented — Auditor finding #6)
6. Fix `HMM_LABEL_RENAME` in `config.py` to be derived from labeling function, not hardcoded (Auditor finding #8)

---

## Step 4 — Subject-Parameterized Loader

**Goal**: Allow `loader.py` functions to work across subjects.

### Changes
- Fix K-parameter ignored bug (Auditor finding #3)
- Add `subject` parameter to `load_hmm_assignments/per_session/trajectory` (default `"BG_046"`)
- Fix `load_tf_traces_npz` hardcoded subject name (Auditor finding #4)
- Add `load_hmm_cv(K=3, subject="BG_046")`

---

## Step 5 — Multi-Subject Hierarchical Fitting (FUTURE — blocked on .pkl availability)

**Prerequisite**: BG_031/038/039 .pkl files (KS4 + TPrime + raw_to_pkl pipeline).

### Design
- Joint multi-subject EM: pass all subjects' sessions to `fit_best_model()` (already supported)
- Shared: GLM weights, transition matrix → common state vocabulary
- Subject-specific: state sequences → subject-specific fractions and trajectories
- Diagnostic: per-subject psychometric curves under shared model — if Engaged curves align, shared-weight assumption holds
- Fallback: per-subject GLM weights + shared transition matrix if base rates diverge too much

---

## Pre-Commit Checklist (applied before any commit)

- [ ] Constants imported from `visdetect.analysis.constants`
- [ ] Session filter uses `load_staging_manifest(qc_only=True)`
- [ ] No parametric tests on behavioral/neural data without justification
- [ ] Effect sizes alongside every p-value
- [ ] Figures use canonical `HMM_STATE_COLORS`, `STAGE_COLORS`
- [ ] `del sess; gc.collect()` in session loops
- [ ] `save_figure()` for analysis suite output

---

## Priority Order

1. **Pre-step bug fixes** (warm-start bug + cross-session transition — now)
2. **Step 1** — Expert-only fit diagnostic (answers the key design question)
3. **Step 3** — Robust labeling (needed before multi-subject, low code cost)
4. **Step 2** — CV integration (validates model)
5. **Step 4** — Subject loader (infrastructure)
6. **Step 5** — Multi-subject (blocked on data pipeline)
