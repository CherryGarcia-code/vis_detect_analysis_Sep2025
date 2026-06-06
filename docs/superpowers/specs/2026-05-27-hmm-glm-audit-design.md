# GLM-HMM Audit and Improvement — Design

**Date:** 2026-05-27
**Owner:** Ben (UCL)
**Status:** Design
**Subject of audit:** `src/visdetect/analysis/hmm.py`, `src/visdetect/analysis/hmm_downstream.py`, and the six dependent scripts in `scripts/analysis/behavior/hmm_*.py`.

## 1. Purpose

Audit the project's GLM-HMM implementation against Ashwood et al. (2022, *Nature Neuroscience*) and against its three intended scientific roles in the project:

1. **Behavioral characterization** — states themselves are a finding (state fractions, dwell times, per-state psychometrics).
2. **Gating neural analyses** — states are used to subset trials before computing PSTHs, decoders, encoding models. Mixing or mislabeling between states would directly confound neural results.
3. **Learning-trajectory tracking** — state occupancies across the 42 BG_046 sessions characterize Naive→Learning→Expert progression.

The overarching scientific question this audit serves is:

> *How do we correctly define a behavioral state — one that is statistically warranted, persistent with high posterior confidence, consistent across mice, and maps to behavioral observables outside the model — so that neural activity conditioned on that state can be interpreted as reflecting a genuine behavioral regime?*

The audit answers this for the current single-subject (BG_046) pipeline AND flags what changes when the pipeline scales to a multi-subject cohort, since both directions are in the project roadmap.

### 1.1 Project's a priori state hypothesis

Independent of the GLM-HMM fit, the project has a scientifically grounded 3-state expectation derived from rolling-average behavior across learning:

| State | Defining behavioral signature |
|---|---|
| **Impulsive** | High lick rate on any trial — both response-window hits and early licks (fa). Stimulus-independent. |
| **Stimulus-sensitive** | High lick rate to actual Change trials, low early-lick rate. Lick driven by evidence: large TF changes + occasional fa from sub-threshold TF outliers that mistakenly "triggered" the mouse. |
| **Disengaged** | Reduced responsiveness across the board, possibly with residual scaling at the largest change sizes (mouse "can be bothered" for obvious changes only). |

Learning involves reducing Impulsive occupancy and increasing Stimulus-sensitive occupancy. This is the project's working definition of "what learning looks like behaviorally" and the audit treats it as the reference frame against which GLM-HMM states are validated.

**Important caveat — within-trial time expectancy.** Baseline TF is noisy (fluctuates around 1 Hz, not exactly 1 Hz). Sub-threshold TF outliers can drive licking and are useful for studying neuronal evidence-responsiveness. Separately, mice develop a within-trial time-expectancy — the likelihood of licking grows during the baseline period and can sometimes dominate TF-driven licking. This is a confound for the Impulsive state (long-baseline trials accumulate more time-expectancy, which could be misidentified as cognitive impulsivity). The audit must flag this.

## 2. Scope

**In-scope.** Six audit dimensions:

| § | Dimension | What it asks |
|---|---|---|
| 4.1 | Model selection method | Is K chosen the way Ashwood does it (CV LL in bits/trial), or by BIC? |
| 4.2 | Lapse-model baseline | Is Ashwood's "L" point in K-comparison plots present? |
| 4.3 | Task encoding choices | Are `y`, `X`, history features defined defensibly for change-detection? |
| 4.4 | Priors and regularization | Are weight, transition, and initial-state priors set appropriately? |
| 4.5 | Gating-safety for neural analyses | Are state labels robust enough to subset trials reliably? |
| 4.6 | Learning-trajectory stability | Does the single global fit do justice to a heterogeneous 42-session arc? |
| 4.7 | External state validation | Do inferred states map to behavioral observables outside the model? |

Plus two cross-cutting findings:

- **CC-1** — Architectural readiness for multi-subject (interface, two-stage fit).
- **CC-2** — Auto-labeling robustness across animals (state correspondence).

**Out-of-scope** (noted at the end with rationale):

- PsyTrack continuous-state comparison (Ashwood Fig 4f).
- Dwell-time geometric-distribution check (Ext Fig 2).
- Choice run-length statistics (Ext Fig 8).
- Parameter recovery on simulated data (Ext Figs 9–10).

## 3. Ranked improvement list

Findings sorted by impact ÷ effort. Impact judged against the three use cases AND the million-dollar question (does it sharpen state definitions?).

| ID | Title | Impact | Effort | Use cases |
|---|---|---|---|---|
| F1 | Migrate K selection from BIC to cross-validated test LL in bits/trial | H | S | All |
| F14 | Add posterior-confidence gating helper (γ > threshold, else "unassigned") | H | S | Gating |
| F22 | External behavioral validation per state (RT, lick latency, TF-pulse responsiveness) | H | M | All |
| F25 | Replace rank-based auto-labeling with explicit a priori criteria | M | S | Gating, Multi-subject |
| F3 | Add lapse-model baseline ("L" point) to K-comparison | M–H | S | Behavioral char. |
| F18 | Single global fit vs per-stage fits — pick principled answer | H | M | Trajectory |
| F24 | Add `baseline_duration` covariate — control for within-trial time expectancy | M | S | All |
| F4 | Document `y = is_hit\|is_fa` commitment (state structure depends on it) | M | S | All |
| F10–F13 | Add Gaussian/Dirichlet priors + hyperparameter selection by CV | M | M | All |
| F16 | State-label stability check across CV folds | M | S | Gating |
| F8 | Outcome-history encoding — verify aversive signals are captured | M | S | All |
| F9 | Audit collinearity between `prev_choice` and `prev_early_lick` | M | S | All |
| F15 | Session-boundary sentinel-zero in history features | M | S | All |
| CC-1 | Multi-subject architectural readiness (interface, two-stage fit) | H (later) | M | Multi-subject |
| CC-2 | State correspondence across animals (label robustness) | H (later) | M | Multi-subject |
| F21 | Frozen-Expert decoding alternative for trajectory | M | M | Trajectory |
| F23 | Neural validation — carefully, to avoid circularity | M | M | Gating |
| F5 | Robustness test: `change_size` retained on fa trials | L | S | All |
| F6 | Stim encoding: `log2` vs raw ratio vs z-scored | L | S | All |
| F7 | History encoding `{0,1}` vs `{-1,+1}` | L | S | All |
| F17 | Min-run threshold for state assignment | L | S | Gating |
| F19 | Per-stage K-comparison diagnostic | L | S | Trajectory |
| F20 | Posterior-confidence calibration across stages (diagnostic) | L | S | Trajectory |
| F2 | K range default `(2,3,4,5)` — matches Ashwood, no action | L | trivial | — |
| F12 | Initial-state Dirichlet prior — small effect | L | trivial | — |

## 4. Findings by dimension

### 4.1 — Model selection method

#### F1 — Migrate K selection from BIC to cross-validated test LL in bits/trial   [Impact: H, Effort: S]

**Current state.** `fit_best_model` selects K by minimum BIC ([hmm.py:842](src/visdetect/analysis/hmm.py#L842)). BIC requires `n_params * log(n_trials)`, computed in [hmm.py:574-578](src/visdetect/analysis/hmm.py#L574-L578).

**Ashwood reference.** Methods, "Assessing model performance — Cross-validation" and Figs 2b, 4a, 4f, 5b, 7b: five-fold CV with log-likelihood reported as **bits per trial relative to a Bernoulli coin-flip null** (Methods Eq. 22).

**Why it matters.** BIC is asymptotically consistent for *iid* observations and a fixed model class; for HMMs with multiple states whose parameters trade off against the transition matrix, BIC is known to under- or over-penalize depending on dwell-time regime. Ashwood's choice of held-out-session CV is the field standard and the right metric for comparing K=2, 3, 4, 5 because it directly measures generalization to unseen behavior — which is precisely what gating neural analyses requires.

**Multi-subject note.** Bits/trial normalization is essential for cross-animal model comparison; raw LL is not comparable across mice with different trial counts. This must be addressed before any cohort fit.

**Recommendation.** Substantial change. Wire the existing `loso_cross_validation` in [hmm_downstream.py:37](src/visdetect/analysis/hmm_downstream.py#L37) into `fit_best_model`. Report bits/trial. Default selection on highest mean CV-LL; BIC can remain in the `selection_df` for reference.

**Sketch.**
```python
# In fit_best_model:
records.append({
    "K": K,
    "train_ll": best_ll_K,
    "cv_ll_bits_per_trial": loso_mean,
    "cv_ll_std": loso_std,
    "bic": bic_val,
    "aic": aic_val,
})
best_K = selection_df.loc[selection_df["cv_ll_bits_per_trial"].idxmax(), "K"]
```

---

#### F2 — Default `K_range=(2,3,4,5)`   [Impact: L, Effort: trivial]

Matches Ashwood. **No action.** Keep as default but make it overridable (already is).

---

### 4.2 — Lapse-model baseline

#### F3 — Add lapse-model baseline as the "L" point in K-comparison   [Impact: M–H, Effort: S]

**Current state.** No lapse model is fit. K-comparison plots start at K=1 (basic GLM).

**Ashwood reference.** Figs 2b, 4a, 4b, 5b — the "L" point sits between K=1 and K=2 and represents a *restricted* 2-state GLM-HMM whose state-2 weights are all zero except for the bias, and whose transition matrix has identical rows (so the lapse probability is stimulus-independent and time-independent). Methods, "Classic lapse model for sensory decision-making" (Eq. 1, 2).

**Why it matters.** The lapse model is the dominant *prior* model in the field. If a multi-state GLM-HMM does not beat it on bits/trial, the multi-state story is not warranted. Without "L" in your comparison plots, an external reviewer cannot tell whether your K=3 fit is genuinely better than a single-strategy + lapse explanation.

**Adaptation to lick/no-lick.** Ashwood has separate γ_l, γ_r for left/right lapse. For binary lick/no-lick the natural analog is a single γ controlling P(spontaneous lick), with the unconstrained sigmoid in the engaged state:

$$P(y_t = 1 | x_t) = (1 - \gamma) \cdot \sigma(w \cdot x_t) + \gamma \cdot \gamma_{\mathrm{lick}}$$

where γ is the lapse probability and γ_lick is P(lick | in lapse state).

**Multi-subject note.** Lapse baseline is computed per-animal; aggregation to cohort matches what Ashwood does (their Fig 4a shows individual lines).

**Recommendation.** Substantial addition (one new file or extension to `hmm.py`). Fit lapse model as restricted 2-state GLM-HMM; include in K-comparison plots and `selection_df`.

**Sketch.**
```python
def fit_lapse_model(sessions_data, n_features, config) -> GLMHMM:
    model = GLMHMM(n_states=2, n_features=n_features, config=config)
    model._init_params(seed=0)
    # Constrain state-2 to have only bias non-zero, transition rows identical.
    # Run EM with constrained M-step (project weights[1, 1:] = 0 each iteration).
    ...
```

---

### 4.3 — Task encoding choices

#### F4 — Document the `y = is_hit | is_fa` commitment   [Impact: M, Effort: S]

**Current state.** [hmm.py:132](src/visdetect/analysis/hmm.py#L132): `y = (df["is_hit"] | df["is_fa"]).astype(float).values`.

**Why it matters.** This single line defines what "lick" means to the model and therefore what a state is. With current encoding, an "Impulsive" state is one with high P(lick) regardless of stimulus — fa-dominated. With `y = is_hit` only, fa trials become "no-lick" observations and the Impulsive state disappears (or is captured implicitly via the bias).

**The project's a priori 3-state hypothesis (§1.1) explicitly requires the Impulsive state as a distinct cognitive regime.** Mice need to learn to control impulsivity; Impulsive behavior is part of the science, not noise. The current encoding (a) is therefore the correct commitment for this project — *not* a hyperparameter to optimize.

This means F4 is **not** a sensitivity analysis (test alternatives, pick the best). It is a **commitment to document**, with one short confirmatory check that the commitment behaves as expected.

**Multi-subject note.** All mice in the cohort should use the same encoding — `y = is_hit | is_fa` — so that the Impulsive state is identifiable in every animal that has impulsive trials.

**Recommendation.** Minor. Add a docstring paragraph to `prepare_session_data` explaining the encoding choice and citing the §1.1 hypothesis. Add one diagnostic plot: per-state P(lick|catch) vs P(lick|large-go) across the fitted model — confirms the three states fall into the expected three corners of this 2D space (Impulsive = both high; Stim-sensitive = catch low, go high; Disengaged = both low).

**Sketch.**
```python
# In prepare_session_data docstring:
"""
Choice variable y = is_hit | is_fa.

Rationale: the project's a priori state structure (see specs/...hmm-glm-audit)
requires an Impulsive state distinguishable from Stimulus-sensitive and
Disengaged. Treating fa (early licks) as licks lets the model learn an
Impulsive state with high P(lick) regardless of stimulus. Treating fa as
no-lick (the alternative) would collapse the K=3 structure to K=2.
"""
```

---

#### F5 — `change_size` retained on fa trials — robustness check   [Impact: L, Effort: S]

**Current state.** [hmm.py:137-144](src/visdetect/analysis/hmm.py#L137-L144) retains scheduled `change_size` even on fa trials, with a docstring rationale that the scheduled value is "still an unconditional trial property." Reasonable.

**Why it matters.** The defense is correct in expectation, but worth a sanity check: do `is_fa` trials cluster at any particular `change_size`? If fa-rate is uniform across scheduled change sizes, there's no issue. If fa correlates with change_size (e.g., mice anticipating large changes), the encoding bakes that correlation into the stim weight.

**Multi-subject note.** None specific; same test applies per animal.

**Recommendation.** Diagnostic check. Plot fa-rate vs `change_size` for each session. If approximately flat, document and keep current encoding. If not, consider alternative.

---

#### F6 — Stim encoding: `log2` vs raw ratio vs z-scored   [Impact: L, Effort: S]

**Current state.** [hmm.py:143](src/visdetect/analysis/hmm.py#L143): `np.log2(np.clip(change_size, 1.0, None))`. Catch → 0; go change_sizes → 0.32, 0.43, 0.58, 1.0, 2.0.

**Ashwood reference.** Methods, "Forming the design matrix": IBL uses z-scored signed contrast.

**Why it matters.** `log2` compresses the high end (4.0×) closer to the rest, which is appropriate if perceptual discriminability scales with log-frequency-ratio (likely true for TF). z-scoring across trials would also work and matches Ashwood's recipe more directly.

**Multi-subject note.** z-scoring should be **per-animal**, not pooled, to avoid one mouse's distribution dominating.

**Recommendation.** Compare CV LL across encodings. If equivalent (likely), document `log2` choice with TF-perception justification. If z-scored wins, switch.

---

#### F7 — History encoding `{0,1}` vs `{-1,+1}`   [Impact: L, Effort: trivial]

**Current state.** [hmm.py:147-152](src/visdetect/analysis/hmm.py#L147-L152): `prev_choice` and `prev_reward` are `{0, 1}`.

**Ashwood reference.** Methods: `{-1, +1}` for symmetry with their L/R encoding.

**Why it matters.** The two encodings produce the same model up to a bias shift; with `{0,1}`, the bias term absorbs the mean of the history features and is therefore less directly interpretable as "innate lick propensity." With `{-1, +1}`, bias is the lick propensity when history is "average."

**Multi-subject note.** None.

**Recommendation.** Minor. Switch to `{-1, +1}` for symmetry with Ashwood and cleaner bias interpretation. Cosmetic; doesn't change predictions.

---

#### F8 — Outcome-history encoding — verify aversive signals are captured   [Impact: M, Effort: S]

**Current state.** [hmm.py:150-152](src/visdetect/analysis/hmm.py#L150-L152): `prev_reward = (is_hit & is_go).astype(float)` — 1 if previous trial was rewarded hit-on-go, 0 otherwise. The `prev_early_lick` regressor ([hmm.py:159-160](src/visdetect/analysis/hmm.py#L159-L160)) separately codes whether the previous trial was an fa (early lick → punished by noise burst).

**Why it matters.** Three outcome categories matter for state transitions:
1. **Rewarded** — hit-on-go (water delivered). Current `prev_reward` captures this.
2. **Punished** — fa (noise-burst + timeout). Currently captured by `prev_early_lick`, which is structurally redundant with `prev_choice` (see F9).
3. **Neutral/non-rewarded** — miss-on-go, miss-on-catch (correct reject), catch-hit (false alarm but not actively punished per project task design). All currently coded as 0.

The split is: `prev_reward=1` ⟺ rewarded; everything else is 0. The punishment signal lives only in `prev_early_lick`, which has the collinearity issue from F9. There is no single signed `prev_outcome` axis.

This is workable if F9 is resolved cleanly (i.e., decide whether fa-history lives in `prev_choice` or `prev_early_lick`, not both), but the audit should confirm that:
- Any post-trial state shift induced by noise-burst punishment is identifiable from the current encoding.
- No additional aversive cue exists (e.g., catch-hit) that the current encoding misses.

**Multi-subject note.** Mice differ in sensitivity to noise burst; whichever encoding survives F9 should let per-animal fits learn this.

**Recommendation.** Run alongside F9. After F9 decides which feature codes fa-history, fit and compare CV LL between:
- Current encoding (`prev_reward` + `prev_early_lick`).
- A signed `prev_outcome` ∈ {+1 rewarded, −1 fa, 0 else} replacing both.
- A more granular three-feature encoding (`prev_reward`, `prev_punishment_fa`, `prev_choice_other`).

**Sketch.**
```python
# Variant: explicit signed outcome (replaces prev_reward and prev_early_lick).
prev_outcome = np.zeros(len(df))
rewarded = (df["is_hit"] & df["is_go"]).astype(int).values   # +1: water
punished = df["is_fa"].astype(int).values                    # -1: noise burst
prev_outcome[1:] = (rewarded - punished)[:-1]                # {-1, 0, +1}
```

---

#### F9 — Audit collinearity between `prev_choice` and `prev_early_lick`   [Impact: M, Effort: S]

**Current state.** Both features fire when previous trial was fa:
- `prev_choice[t] = y[t-1] = 1` when prev was fa (since `y = is_hit | is_fa`).
- `prev_early_lick[t] = is_fa[t-1] = 1`.

These overlap. The model can still fit, but with inflated variance on the two weights and potentially unstable estimates.

**Why it matters.** Collinearity inflates standard errors and makes coefficient interpretation unreliable. If you want to claim "the impulsive state has high `prev_early_lick` weight," you need to know that weight is identifiable independently of `prev_choice`.

**Multi-subject note.** Same per animal.

**Recommendation.** Compute the empirical correlation across all trials. With F4 committed (`y = is_hit | is_fa` stays), option (ii) from earlier framings — drop fa from `y` — is off the table. So the realistic choices are:

- **(i)** Drop `prev_early_lick`; rely on the fact that `prev_choice = 1` already captures any prev-trial lick.
- **(iii)** Keep both, accepting some variance inflation, because they encode different information: `prev_choice` is *any* prev-lick (hit OR fa); `prev_early_lick` is specifically *aversive* prev-fa. The two features carry distinct signals when the previous trial was a hit (prev_choice=1, prev_early_lick=0) vs fa (both 1).

Decide by running both and comparing CV LL plus weight identifiability (variance of weights across CV folds).

---

#### F24 — Add `baseline_duration` covariate — control for within-trial time expectancy   [Impact: M, Effort: S]

**Current state.** No within-trial timing information enters the GLM. Every trial is reduced to a single binary `y` and a covariate vector that doesn't depend on the baseline period's duration.

**Why it matters (see §1.1 caveat).** Mice develop within-trial time-expectancy: the longer the baseline runs, the more likely a lick becomes — independently of the stimulus and possibly dominating it. This creates a confound for the Impulsive state:

- Long-baseline trials accumulate more time-expectancy → higher fa-rate AND higher late-hit-rate.
- The model, lacking a time covariate, has no way to attribute these licks to time expectancy.
- It will instead attribute them to the bias term — and trials with long baselines will look "Impulsive" by accident rather than by cognitive regime.

The risk: a state labeled "Impulsive" might actually be "trials that happened to have long baselines." Same risk for an Engaged state with elevated late-hit rate.

**Multi-subject note.** Time-expectancy strength likely varies per animal (some mice are more time-anticipatory than others). Per-animal fits would absorb this individual variation into per-animal weights; pooled fits would average over it.

**Recommendation.** Substantial. Add `baseline_duration` (or `time_to_change` for go trials, `total_trial_duration` for fa trials) as a continuous covariate. Two acceptance checks:

1. **Reduces confound:** after fitting with `baseline_duration` included, the per-state distributions of `baseline_duration` should be more similar across states (i.e., the model is no longer using baseline duration to classify state).
2. **Improves CV LL:** marginal LL gain from the new feature, evaluated by CV with and without.

If both pass, adopt. If neither passes, document that time-expectancy is not a meaningful confound at the population level (mice may have learned to suppress it).

**Sketch.**
```python
# In prepare_session_data, add to the design matrix:
baseline_duration = df["change_time_s"].values.astype(float)   # or trial duration for fa
# z-score within session to keep weights interpretable
baseline_duration = (baseline_duration - np.nanmean(baseline_duration)) / np.nanstd(baseline_duration)
baseline_duration = np.nan_to_num(baseline_duration, nan=0.0)
X = np.column_stack([X, baseline_duration])
```

**Diagnostic plot.** Per-state distribution of `baseline_duration` (violin) — if the Impulsive state's distribution shifts toward longer baselines, confound is real and confirmed.

---

### 4.4 — Priors and regularization

#### F10 — Add Gaussian prior on GLM weights (l2 regularization)   [Impact: M, Effort: S]

**Current state.** [hmm.py:191](src/visdetect/analysis/hmm.py#L191): `l2_penalty: float = 0.0` — flat prior.

**Ashwood reference.** Methods, "GLM-HMM objective function" Eq. 6: zero-mean Gaussian prior with σ² selected by validation grid over {0.5, 0.75, 1, 2, 3}. For IBL: σ=2.

**Why it matters.** With a flat prior, weights for low-occupancy states (e.g., a state with γ summing to a few dozen trials) are essentially unregularized and can blow up. This destabilizes EM and contributes to the "all restarts failed" path. A mild Gaussian prior buys numerical stability at almost no cost.

**Multi-subject note.** σ should be selected by CV across mice, not hard-coded; per-animal small-sample fits especially benefit from regularization.

**Recommendation.** Add support (`l2_penalty` is already wired into `_nll_and_grad`, just needs to be non-zero by default). Default σ=2 to match Ashwood; verify via grid search.

---

#### F11 — Add Dirichlet pseudocount on transition counts   [Impact: M, Effort: S]

**Current state.** [hmm.py:423-426](src/visdetect/analysis/hmm.py#L423-L426): `A = total_xi / row_sums` — MLE, no smoothing.

**Ashwood reference.** Methods Eq. 20: `A_jk = (α - 1 + Σξ) / (K(α-1) + ΣΣξ)`. α=2 for IBL.

**Why it matters.** Without smoothing, a state that never transitions to another state in the data assigns it probability zero forever. This can lock in pathological transition structure during EM. The Dirichlet pseudocount (α-1 ≥ 0) injects a small amount of probability into every off-diagonal entry.

**Multi-subject note.** Same value of α appropriate per animal.

**Recommendation.** Minor. Add `dirichlet_alpha: float = 2.0` to config, modify M-step.

**Sketch.**
```python
A = (total_xi + (alpha - 1)) / (total_xi.sum(axis=1, keepdims=True) + K * (alpha - 1))
```

---

#### F12 — Initial-state Dirichlet prior   [Impact: L, Effort: trivial]

**Current state.** [hmm.py:429](src/visdetect/analysis/hmm.py#L429): MLE.

**Ashwood reference.** Methods: α_π = 1 (flat). Essentially no prior.

**Recommendation.** No action needed. Document that initial-state prior is flat.

---

#### F13 — Hyperparameter selection by CV grid search   [Impact: M, Effort: M]

**Depends on F10, F11.** Once F10 and F11 are in, σ and α should not be hardcoded but selected by grid search on validation LL.

**Recommendation.** Implement after F10/F11. Grid: σ ∈ {0.5, 1, 2, 3}, α ∈ {1.5, 2, 3}. Select on mean CV LL.

---

### 4.5 — Gating-safety for neural analyses

#### F14 — Add posterior-confidence gating helper   [Impact: H, Effort: S]

**Current state.** [hmm.py:929](src/visdetect/analysis/hmm.py#L929): `decode_session` returns Viterbi hard assignment. Posteriors `p_state_k` are also returned but downstream scripts typically use `hmm_state` directly.

**Why it matters.** This is the single biggest gating-safety improvement. Mixed-confidence trials (e.g., γ = [0.45, 0.55, 0.0]) are currently labeled as state 1 with full confidence, even though the model is essentially undecided. Including those trials in "state-1 PSTHs" injects noise from state-0 trials. Ashwood notes (Fig 3a) that posteriors are usually close to 1, meaning a confidence threshold throws away few trials but removes the ambiguous cases that confound neural analyses.

**Multi-subject note.** Threshold should be the same across mice for consistency.

**Recommendation.** Add a helper:
```python
def assign_states_with_confidence(
    posteriors: np.ndarray,  # (T, K)
    threshold: float = 0.8,
) -> np.ndarray:  # (T,) with -1 = unassigned
    max_prob = posteriors.max(axis=1)
    assigned = posteriors.argmax(axis=1)
    assigned[max_prob < threshold] = -1
    return assigned
```
Plumb through `decode_session` as an option. Update downstream scripts to use the helper for neural-conditioning calls; document that behavioral-characterization figures can still use Viterbi.

---

#### F15 — Session-boundary sentinel-zero in history features   [Impact: M, Effort: S]

**Current state.** [hmm.py:148-152](src/visdetect/analysis/hmm.py#L148-L152): `prev_choice[0] = 0`, `prev_reward[0] = 0` at session start.

**Why it matters.** A first-trial-of-session looks identical to a within-session trial preceded by a no-lick / no-reward outcome. The model has no way to know the session just started, so it conflates the two. For mice that warm up over the first ~50 trials (Ashwood Fig 5g), this systematically biases the initial state inference.

**Multi-subject note.** Same per animal; matters more in tasks with strong warm-up.

**Recommendation.** Substantial. Add an explicit `is_session_start` feature OR drop the first trial of each session from training. Decision depends on whether warm-up is a state the model should learn (then keep first trials with the indicator) or noise (drop them).

---

#### F16 — State-label stability across CV folds   [Impact: M, Effort: S]

**Current state.** `auto_label_states` ([hmm.py:855](src/visdetect/analysis/hmm.py#L855)) uses rank-based criteria within a single fit (argmax p_catch → Impulsive, argmin sensitivity → Disengaged, remainder → Engaged). It runs *after* fitting on a per-model basis.

**Why it matters.** For gating, you need to trust that "state 0 = Disengaged" means the same thing across CV folds AND across mice. If LOSO-fold-1's "Disengaged" is fold-2's "Engaged_low," every aggregated analysis is corrupted.

**Multi-subject note.** This is the same problem CC-2 tackles for multi-subject. The diagnostic is the same.

**Recommendation.** Diagnostic. For each LOSO fold, fit, label, and check correspondence with the global-fit labels (e.g., by GLM-weight cosine similarity). Report a confusion matrix of label agreement. If labels are unstable, augment `auto_label_states` with anchoring (match to global-fit weights by minimum-cost assignment).

---

#### F17 — Min-run threshold for state assignment   [Impact: L, Effort: S]

**Current state.** Viterbi can produce single-trial state excursions. No min-run enforcement.

**Why it matters.** A single-trial state assignment is unlikely to reflect a genuine cognitive transition; it's more likely an HMM artifact at a noisy trial. For gating, dropping single-trial runs (or runs < 3 trials) sharpens state definitions at the cost of losing a few trials.

**Multi-subject note.** None.

**Recommendation.** Optional. Add a post-processing helper `enforce_min_run(states, min_run=3, fill_value=-1)`. Use for gating, NOT for behavioral characterization (the raw Viterbi sequence is closer to the model).

---

#### F25 — Replace rank-based auto-labeling with explicit a priori criteria   [Impact: M, Effort: S]

**Current state.** `auto_label_states` ([hmm.py:855](src/visdetect/analysis/hmm.py#L855)) uses generic rank-based logic: argmax(p_catch) → Impulsive; argmin(sensitivity = p_high − p_catch) → Disengaged; remainder → Engaged / Engaged_low / Engaged_high. The criteria are heuristic and could in principle assign labels in ways that disagree with §1.1's a priori definitions.

**Why it matters.** §1.1 specifies the three states by their joint signature on two axes — P(lick | catch) and P(lick | large go). The rank-based logic only uses one ranking per assignment step; it doesn't enforce the full joint signature. For example:

- A state with moderate p_catch AND moderate p_high could be labeled Impulsive (by argmax(p_catch) if no other state has higher p_catch), even though it doesn't match the "high on both" Impulsive signature.
- A state with low p_catch AND low p_high (truly Disengaged) could in principle be labeled Stimulus-sensitive by elimination if the Impulsive criterion fires on the wrong state.

Explicit criteria remove these failure modes and make labels reproducible across mice and across CV folds.

**Multi-subject note.** This is foundational for CC-2. If labels are assigned by explicit criteria rather than rank, two mice's "Impulsive" states are guaranteed to match the same signature, removing the need for a separate state-correspondence step (or reducing it to a sanity check).

**Recommendation.** Substantial reframing of `auto_label_states`. Replace rank-based logic with thresholded joint criteria in the (p_catch, p_high) plane:

| Region | Label |
|---|---|
| p_catch ≥ τ_high AND p_high ≥ τ_high | Impulsive |
| p_catch < τ_low AND p_high ≥ τ_high | Stimulus-sensitive |
| p_catch < τ_low AND p_high < τ_high | Disengaged |
| Anything else | "Unlabeled" or "Intermediate_k" (flagged for inspection) |

Thresholds τ_low and τ_high chosen empirically (e.g., τ_low = 0.2, τ_high = 0.5; tunable). For K > 3, multiple states may match the same region; suffix with `_1, _2, …` by ascending sensitivity.

**Sketch.**
```python
def auto_label_states_explicit(
    model: GLMHMM,
    tau_low: float = 0.2,
    tau_high: float = 0.5,
    stim_high: float = 2.0,    # log2(4.0)
) -> List[str]:
    K, D = model.n_states, model.n_features
    x_catch = np.zeros(D); x_catch[0] = 1.0
    x_high  = np.zeros(D); x_high[0]  = 1.0; x_high[1] = stim_high
    p_catch = np.array([float(expit(model.weights[k] @ x_catch)) for k in range(K)])
    p_high  = np.array([float(expit(model.weights[k] @ x_high))  for k in range(K)])

    labels = []
    for k in range(K):
        if p_catch[k] >= tau_high and p_high[k] >= tau_high:
            labels.append("Impulsive")
        elif p_catch[k] < tau_low and p_high[k] >= tau_high:
            labels.append("Stimulus_sensitive")
        elif p_catch[k] < tau_low and p_high[k] < tau_high:
            labels.append("Disengaged")
        else:
            labels.append(f"Intermediate_{k}")
    return labels
```

Keep `auto_label_states` (rank-based) as a fallback for when explicit criteria produce all-Unlabeled (e.g., a K=4 fit with two intermediate states).

---

### 4.6 — Learning-trajectory stability

#### F18 — Single global fit vs per-stage fits — pick a principled answer   [Impact: H, Effort: M]

**Current state.** Pipeline fits ONE GLM-HMM on all sessions concatenated (the natural reading of `fit_best_model(sessions_data, ...)` called with all sessions). State fractions across sessions are then computed by decoding each session.

**Why it matters.** A 42-session arc spanning Naive→Expert is heterogeneous: GLM weights that fit a Naive mouse poorly may dominate Expert sessions if Expert trial count is higher, or vice versa. If you fit globally, the inferred states are a compromise that may not best describe either end. Alternatives:

1. **Global fit** (current) — single state set across all stages; trajectory = state-fraction shifts.
2. **Per-stage fits** — separate K and weights for Learning vs Expert; need to match states across stages.
3. **Frozen-Expert decoding** (= F21) — fit on Expert only (cleanest behavior); decode Learning sessions with the frozen model. Trajectory = how Learning sessions navigate Expert's state space.

**Multi-subject note.** Per-stage fits get even more complicated with multi-subject; frozen-Expert (option 3) is the cleanest if it works.

**Recommendation.** Run all three and compare on (i) CV LL within each stage, (ii) neural separability between states within each stage, (iii) interpretability of the trajectory plot. Decide based on results.

---

#### F19 — Per-stage K-comparison diagnostic   [Impact: L, Effort: S]

**Why it matters.** If optimal K differs for Learning (e.g., K=2: engaged/disengaged) vs Expert (K=3 with impulsive), the global fit may be forcing a single K on heterogeneous regimes.

**Recommendation.** Diagnostic only. Fit K=2..5 separately on Learning-only and Expert-only subsets; report whether optimal K differs.

---

#### F20 — Posterior-confidence calibration across stages   [Impact: L, Effort: S]

**Why it matters.** Are Learning-stage trials systematically lower-confidence (γ_max < 0.8 more often) than Expert? If so, F14's confidence threshold will throw away a larger fraction of Learning trials — a stage-confounded gating issue.

**Recommendation.** Diagnostic. Plot distribution of max-posterior per trial, split by stage.

---

#### F21 — Frozen-Expert decoding alternative   [Impact: M, Effort: M]

**See F18.** This is the implementation of option 3 in F18.

**Recommendation.** Implement as part of F18's comparison. If it wins on the three criteria, make it the default and deprecate the global fit.

---

### 4.7 — External state validation

#### F22 — External behavioral validation per state (RT, lick latency, TF-pulse responsiveness)   [Impact: H, Effort: M]

**Current state.** None. Per-state behavioral metrics in `hmm_downstream.py` cover d′, criterion, hit rate, FA rate — all derived from the same outcomes the model was fit on.

**Ashwood reference.** Fig 6: Q-Q plots of response-time distributions per state (engaged vs disengaged); violation-rate difference per state. Critical evidence that states are not just statistical artifacts.

**Why it matters.** This is the direct answer to "is a state real?" States are validated when behavioral observables *not used in fitting* differ across states in the directions predicted by §1.1.

**Predicted per-state signatures.** Given the project's three a priori states (§1.1), each external observable should produce a specific, falsifiable pattern:

| Observable | Impulsive | Stim-sensitive | Disengaged |
|---|---|---|---|
| **Lick latency** (change-onset → first lick, on hits) | Short, low variance — early-anticipatory licks bleed into "hits" | Short, low variance — evidence-driven | Long, high variance — sluggish |
| **RT distribution shape** (Q-Q vs all-trials) | Heavy left tail | Tight, mode at ~250-350 ms | Heavy right tail (Ashwood Fig 6 pattern) |
| **TF-pulse responsiveness** (lick rate driven by sub-threshold TF outliers) | Low/noisy — licks are stim-independent | **High** — pulses drive evidence accumulation, occasional fa from misattributed outliers | Low across the board, perhaps residual scaling at the largest outliers |
| **Psychometric slope** (P(lick) vs change_size on go trials) | Shallow — licks regardless of change size | **Steep** — sensitive across change sizes | Possibly steep at the largest sizes only ("can be bothered" effect) |
| **Inter-trial interval** | Short — repeated initiation | Moderate | Long |

**TF-pulse responsiveness is the strongest discriminator.** Impulsive licks are stimulus-independent by definition, so sub-threshold TF outliers should NOT preferentially drive them. Stimulus-sensitive licks ARE evidence-driven, so TF outliers should occasionally trigger them. This is a mechanistically specific test that the existing `tf_pulse` analysis in `src/visdetect/analysis/` can deliver.

**Multi-subject note.** External validation per animal first; then aggregate. Mice that show clear state separation on external observables are evidence the framework is sound; mice that don't may need a different K or feature set. Pre-register: a successful audit requires per-state differences on at least 2 of the 5 observables for ≥80% of mice.

**Recommendation.** Substantial. Build a `validate_states_externally(model, sessions, observables)` helper. Produce a figure analogous to Ashwood Fig 6 per animal: per-state RT distribution, per-state lick latency, per-state TF-pulse responsiveness, per-state psychometric slope. This figure becomes the "are the states real?" evidence in the manuscript.

---

#### F23 — Neural validation — carefully, to avoid circularity   [Impact: M, Effort: M]

**Why it matters.** The strongest evidence for "real states" is that neurons differ between them. BUT if you later use neural activity as a feature in the model (you don't currently, but it's a natural extension), neural validation becomes circular.

**Recommendation.** OK to validate states with neural activity as long as you commit to NOT using neural activity in the model. Document this. Pre-register the test: e.g., D1-SPN baseline firing rate should differ between Engaged and Disengaged states; if it does, that's confirmation. If you later add neural features to the GLM-HMM (a future extension), you lose this validation lever.

**Implementation.** Per-state baseline PSTHs across the population; per-state cell-type-specific firing-rate distributions; per-state encoding model fits.

---

### 4.8 — Cross-cutting findings

#### CC-1 — Architectural readiness for multi-subject   [Impact: H (later), Effort: M]

**Current state.** `fit_best_model(sessions_data, ...)` treats all sessions as fungible; no `animal_id` concept. `sessions_data` is `List[Dict]` with no grouping field.

**Ashwood reference.** Algorithm 1, Methods, "Comparing states across animals":
1. Fit one GLM (single-state model) to all data from all animals (concatenated).
2. Fit K-state GLM-HMM to all animals together to get global parameters (n_restarts=20).
3. For each animal, initialize K-state GLM-HMM with global parameters and re-fit per animal.

This achieves state correspondence by initializing each per-animal fit at the global solution.

**Why it matters.** Without this, every per-animal fit can find its own state labeling, and "Engaged" for mouse A may not correspond to "Engaged" for mouse B.

**Multi-subject note.** This *is* the multi-subject finding.

**Recommendation.** Substantial. Refactor `prepare_session_data` to include `animal_id`. Add `fit_global_then_per_animal(all_sessions_data, ...)` that implements Algorithm 1. Keep current single-animal `fit_best_model` as the inner per-animal step. Roll out only when multi-subject is actively needed; design now to avoid rework later.

**Sketch.**
```python
def fit_global_then_per_animal(
    all_sessions: List[Dict],   # each dict has animal_id
    K: int,
    n_restarts: int = 20,
) -> Dict[str, GLMHMM]:
    # Stage 1: global GLM fit
    global_glm = fit_glm_pooled(all_sessions, n_restarts=n_restarts)
    # Stage 2: global GLM-HMM init from GLM weights
    global_hmm = fit_glmhmm_pooled(
        all_sessions, K=K, n_restarts=n_restarts, init_weights=global_glm.weights
    )
    # Stage 3: per-animal fits initialized from global HMM
    per_animal = {}
    for animal_id in unique_animals(all_sessions):
        animal_data = [s for s in all_sessions if s["animal_id"] == animal_id]
        model = GLMHMM(K=K, ...)
        model.init_from(global_hmm)
        model.fit(animal_data)
        per_animal[animal_id] = model
    return per_animal
```

---

#### CC-2 — State correspondence across animals (label robustness)   [Impact: H (later), Effort: M]

**Current state.** `auto_label_states` uses rank-based labeling on a single model. No mechanism to ensure animal A's "Disengaged" matches animal B's.

**Why it matters.** If multi-subject pooled analyses compare "all Engaged trials across mice," every mouse must agree on what each label means. Algorithm 1 (CC-1) helps via shared init, but doesn't guarantee labels match — the per-animal EM can still permute states.

**Relation to F25.** F25 (explicit a priori labeling) is the foundation for CC-2. If every per-animal fit is labeled by the same explicit criteria over (p_catch, p_high), then mice's "Impulsive" states are guaranteed to match the same signature by construction — and the correspondence problem reduces to "which animal's fit didn't produce a state in the Impulsive region?" rather than "do these two states correspond?"

**Recommendation.** Substantial, but lighter once F25 is in place. After per-animal fits:

1. Apply F25's explicit-criterion labels per animal.
2. For each label (Impulsive / Stimulus_sensitive / Disengaged), tabulate which animals have a state matching that label.
3. For animals where a label is missing (e.g., no Impulsive-region state), flag for inspection — either the animal lacks that regime, or the fit failed, or thresholds need tuning.
4. For "Intermediate" states (animal-specific extra states beyond the canonical three), report separately rather than force-aligning.

If F25 is not adopted (criteria fail to cover the data well), fall back to cosine-similarity assignment against a global-fit reference (Hungarian algorithm), as originally planned. Report assignment confidence and any animal whose states don't align.

---

## 5. Out-of-scope items

These were considered but excluded by user decision. Documented here so they can be revisited.

| Item | Why excluded | When to revisit |
|---|---|---|
| **PsyTrack continuous-state comparison** (Ashwood Fig 4f) | Requires extra dependency; not central to defining states for neural conditioning | If reviewers challenge "discrete vs continuous" framing in a manuscript |
| **Dwell-time geometric check** (Ext Fig 2) | Diagnostic only; HMM dwell times *must* be geometric by construction | If dwell-time deviations become a finding worth defending |
| **Choice run-length statistics** (Ext Fig 8) | Diagnostic of model goodness-of-fit; CV LL already covers this | If you need an additional sanity check before a high-stakes claim |
| **Parameter recovery on simulated data** (Ext Figs 9–10) | Sanity check on fitting code; the implementation here is a reasonable port of standard EM | If results become unstable or hard to explain |

## 6. Suggested implementation tracks

Three coherent bundles. Pick one to start; later tracks build on earlier.

### Track A — Minimum to publish (~1 week)

The bare minimum to defend single-subject BG_046 GLM-HMM results in a manuscript:

- **F1** — CV-based K selection in bits/trial.
- **F3** — Lapse baseline.
- **F14** — Posterior-confidence gating helper.
- **F22** — External behavioral validation per state (with TF-pulse responsiveness as the key discriminator).
- **F25** — Explicit a priori labeling criteria (replaces rank-based logic).
- **F4** — Document the `y = is_hit | is_fa` commitment (1-paragraph docstring + diagnostic plot).

After Track A: results are Ashwood-aligned, gating is defensible, labels are scientifically grounded in the §1.1 hypothesis, and states have external evidence. Sufficient for a single-mouse story.

### Track B — Defensible audit response (~2–3 weeks)

Track A plus:

- **F24** — Add `baseline_duration` covariate; test whether Impulsive state is confounded with within-trial time-expectancy.
- **F8, F9** — Outcome-history encoding + collinearity audit.
- **F10, F11, F13** — Priors and hyperparameter selection.
- **F16** — State-label stability across CV folds.
- **F18** — Single global vs per-stage fit decision.

After Track B: methodology fully audited, the Impulsive state is shown to be a cognitive regime (not a time-expectancy artifact), and the state definition is the result of explicit comparisons rather than defaults.

### Track C — Multi-subject + full Ashwood (~1–2 months)

Track B plus:

- **CC-1** — Multi-subject architecture (Algorithm 1).
- **CC-2** — State correspondence across animals (lighter now that F25 is the labeling foundation).
- **F15** — Session-boundary handling.
- **F21** — Frozen-Expert decoding (if F18 selects it).
- Out-of-scope items as they become relevant.

After Track C: cohort-ready pipeline; defensible against Ashwood-paper-level scrutiny.

---

## 7. Acceptance criteria

The audit deliverable (this document) is accepted when:

1. Each of the 27 findings (F1–F25 plus CC-1, CC-2) has a clear current-state citation, recommendation, and rationale.
2. The ranked list at §3 is justified by §4 detail.
3. Multi-subject implications are surfaced wherever they change the recommendation.
4. The three implementation tracks are coherent and progressive.

The **implementation phase** (not this spec) is accepted when:

- Track A items reproduce Ashwood's K-comparison figure style for BG_046.
- External behavioral validation (F22) shows significant per-state differences in at least one observable not used in fitting.
- LOSO CV LL improves or matches the current BIC-selected model.
- Gating helper (F14) is wired into at least one downstream neural script.
