---
name: harden-result
description: Use BEFORE claiming any neural or behavioral result is real - before writing a results doc, captioning a figure claim, saving a memory note, opening a PR with a finding, or telling the user "we found X". Runs this lab's mandatory verification battery (FR-normalization, circularity, pseudoreplication/mixedlm, per-region DMS-vs-VMS breakdown, trial-count matching, lick/movement leakage controls, yield-bias caveats) and a final adversarial refutation pass with Opus 4.8 subagents. Triggers on "is this result solid", "verify this finding", "harden", "write up the result", "can I present this", "adversarial check", "sanity check the effect", or any headline p-value about to leave the repo.
---

# Harden a result before you claim it

**This battery is not hypothetical.** On 2026-07-02 this lab RETRACTED a result already
presented as solid ("sustained TF cells carry the task-state offset more, |task_load| 3.65
vs 2.24 Hz, p=4.9e-3"). It was a raw-Hz firing-rate artifact. On the FR-normalized column —
which the script had **already computed** and which I failed to use — it is a clean NULL
(p=0.37). See `tf_transient_sustained_state_jul2026` + `feedback_verify_important_results_adversarially`.

Run the gates **in order**. Any gate that fails **kills the claim** — you do not get to
"note it as a caveat" and publish anyway. Downgrade to null/exploratory and say so.

---

## Red flags that mean STOP (scan this first — 30 seconds)

| Red flag | The incident it caused |
|---|---|
| My effect is in **raw Hz** and compares **different neurons** | THE RETRACTION. Sustained cells fire faster (14.9 vs 12.8 Hz); ρ(\|task_load\|, base_hz)=+0.59. The whole effect was firing rate. |
| A **`_z` / normalized column exists** in my own cache and I used the raw one | Exactly what happened. `task_load_z` was sitting right there. |
| My **grouping variable was derived from the same trials** as my DV | State-labeler circularity: states are defined from `f_inapplick`/`f_hit_hard`/`f_miss_easy`, so "Impulsive = more FAs" is definitional, not a finding. |
| **n is neurons/cell-sessions** but the variance that matters is **across sessions/mice** | Pooled MWU over 3 mice with no random effect = pseudoreplication. Also B10: "VMS engagement-gated" was per-trial pseudoreplication; n.s. with session-level CIs. |
| I **pooled DMS + VMS** (or all 3 mice) into one p-value | Region confound (DMS \|task_load\| 1.51 vs VMS 3.47 Hz). Latency×outcome: BG_046 ρ=+0.20, BG_031 ρ=−0.10 → cancel to a fake null. Simpson's paradox both directions. |
| My two conditions have **different trial counts** and my metric is an **encoding strength / fit quality** | B9: `c1_r` is trial-count **attenuated** — an apparent state or stage difference in encoding strength is a power artifact. |
| My window could contain **a lick** (or a lick's motor prep) | N1: within-FA timing decode r=**0.56** → **0.027** once leakage-filtered. StimSens-vs-Impulsive change response: p=4e-3 → 0.067 censored → **0.33** on the clean RT>0.25 s subset. |
| The effect rests on **one mouse / one session / cherry-picked exemplars** | Rep-cells exemplars were cherry-picked VMS extremes, exactly where the population trend is flat. B10's "VMS Naive ramp" was a 1-session date-parsing artifact. |
| I used **GMM ΔBIC** to argue bimodality on a skewed variable | ΔBIC=+242 fired on right-skew alone; collapsed to +2.4 after log-transform. Use `silverman_bootstrap` + a matched unimodal null. |

If any row matches, you are at Gate 1, not at "write-up".

---

## Gate 1 — FR-NORMALIZE every cross-neuron magnitude test ⛔ THE #1 ARTIFACT

**WHAT.** Any comparison of *magnitude* (Δrate, |loading|, response size, coupling strength)
between *different neurons* or *different groups of neurons* must be on a firing-rate-normalized
quantity — never raw Hz.

**WHY.** The retraction. Raw-Hz magnitude scales with baseline firing rate; any grouping variable
that correlates with FR (width, cell type, region, yield) will manufacture an effect.
CLAUDE.md golden rule: *"Never average raw firing rates across units without normalization."*

**HOW.**
- Canonical helpers — `src/visdetect/analysis/utils.py`:
  - `compute_zscore_normalized(tensor, bin_centers, baseline_window)` — per-unit z, **shared baseline**
  - `compute_baseline_subtracted(tensor, bin_centers, baseline_window)` — per-unit Δrate (Hz preserved)
- Or per-session z of the metric column (the `task_load_z` pattern), or **rate-matching**
  (`waveform_celltype_join.py` — resample so groups have matched FR distributions), or
  **partial Spearman controlling `base_hz`**, or add `base_hz` as a covariate:
  `width_vs_waveform.py` shows width surviving `outcome ~ w + t2p` and `~ w + base_hz`.
- Guards: shared baseline across ALL conditions compared (**per-condition baselines = CRITICAL
  ERROR**, inflates the low-activity condition); normalize-then-average, never average-then-normalize;
  `if baseline_std < 1e-6: baseline_std = 1.0`.

**KILL CRITERION.** The effect does not survive normalization / rate-matching / the `base_hz`
covariate → **the claim is dead.** It was firing rate. (If a normalized column already exists
in your cache, it is the headline. Full stop.)

---

## Gate 2 — CIRCULARITY

**WHAT.** Is the grouping variable, class label, or state defined from the same data/features as
the dependent variable?

**WHY.** `state_labeler_circularity_caveat`: `STATE_FEATURE_COLS = [f_applick, f_inapplick,
f_nolick, f_abort, f_miss_easy, f_hit_hard]`. So "Impulsive has more FAs / a more liberal
criterion" is a **sanity check that the labeler works**, not a discovery. d′/sharpness claims are
*partially* entangled too (`f_miss_easy`, `f_hit_hard`).

**HOW.**
- List your DV's inputs and your grouping variable's inputs. Any overlap → confirmatory, not discovery.
- Labeler-**independent** readouts: timing / lick-hazard shapes, RT distributions, and **neural** measures.
- If the class comes from a fit, test it on data the fit never saw: the B10 held-out-sign control
  (signs from ODD trials, tested on EVEN → peak r=0.050 vs shuffle 0.007) is the template.
- Independent-criterion check: the transient/sustained → outcome coupling is *non*-circular because
  the outcome metrics come from Change_ON/Hit/FA alignment, which the GLM TF kernel never saw.

**KILL CRITERION.** DV and grouping variable share inputs and no independent readout reproduces
it → **downgrade to "confirmatory / definitional"**; it is not a finding.

---

## Gate 3 — PSEUDOREPLICATION

**WHAT.** Your n is cell-sessions. Your independent replicate is the **session** (units don't repeat
within a session, but they do across sessions; and sessions/mice are the real sampling unit).

**WHY.** The retracted result was *also* a single pooled MWU over 3 mice with no random effect.
B10's "VMS engagement-gated tracking" evaporated once CIs were bootstrapped over **sessions**
instead of trials.

**HOW.** Copy `scripts/tf_responsiveness/state_conditioned/hardening_continuum.py` (continuous
predictor) or `hardening_pseudoreplication.py` (class contrast). Three complementary controls —
none is both high-confidence and high-coverage, so run several:

- **A. Session random-intercept mixedlm** (all mice, full coverage):
  `smf.mixedlm(f"{col} ~ w + C(region)", d, groups=d["session"]).fit()`
- **A′. Cluster-robust OLS, ALWAYS fit alongside it** — statsmodels' mixedlm can silently fail to
  converge when the RE variance is ~0:
  `smf.ols(f"{col} ~ w + C(region)", d).fit(cov_type="cluster", cov_kwds={"groups": d["session"]})`
  **Write BOTH coefficients + the convergence flag** (`getattr(fit, "converged", True)`); prefer the
  cluster-robust OLS when the flag fires. Never silently drop a non-converged model.
- **B. Per-session sign test** (session = replication unit): per-session Spearman ρ or median Δ,
  then `wilcoxon()` across sessions.
- **C. Tracked-unit collapse** (BG_046 only, cleanest, lowest n): collapse cell-sessions to unique
  `um_uid` via `data/cache/tracking_consensus/BG_046/consensus_members.csv` (312-unit UM∩DANT
  consensus), then re-test. Coverage is thin — treat as directional support, not the primary.

**KILL CRITERION.** The effect exists **only** in the pooled cell-session test and dies under the
session random effect / cluster-robust OLS / per-session sign test → **dead**. (Passing looks like
the width→coupling result: mixedlm *stronger* than raw, per-session sign test significant across
24 sessions.)

---

## Gate 4 — PER-REGION / PER-SUBJECT BREAKDOWN — NEVER POOL DMS WITH VMS

**WHAT.** Report the effect **separately** for each region and each mouse, always, even when
the pooled p is beautiful.

**WHY.** Regions differ in magnitude (DMS |task_load| 1.51 vs VMS 3.47 Hz) → pooling makes a
region-composition difference look like an effect. And the reverse: the latency→outcome test
was BG_046 ρ=+0.18..+0.20 (p<0.02) vs BG_031 ρ=−0.10..−0.01 — **they cancelled into a fake pooled
null.** Simpson's paradox runs both ways here.

**HOW.**
- Mapping: **BG_046 = DMS, BG_039 = DMS, BG_031 = VMS, BG_038 = cortex (M1/S1)**.
  `df["region"] = df.subject.map({"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS"})`
- `region_bank_confirmed` is FALSE across the whole `tf_responsive` registry → **cannot gate on it**;
  pool by SUBJECT and treat region as provisional.
- Always carry `+ C(region)` in the model AND print the per-region, per-subject rows
  (`robustness_width_coupling.py` prints `pooled ρ | DMS ρ | VMS ρ` on every line).

**KILL CRITERION.** The effect lives in exactly one region/one mouse and reverses or vanishes in
the other → it is **not** a striatal result. Report it as a single-subject observation or drop it.
(Passing looks like width→coupling: holds independently in DMS-only, VMS-only, and all 3 mice.)

---

## Gate 5 — TRIAL-COUNT MATCHING

**WHAT.** Before comparing encoding *strength* / fit quality / selectivity between two conditions
(states, stages, outcomes), the conditions must have the **same number of trials**.

**WHY.** **B9**: `c1_r` (GLM TF encoding strength) is **trial-count attenuated** — fewer trials
→ noisier fit → lower measured encoding. An "engaged encodes TF better than disengaged" effect
is then guaranteed by the fact that Disengaged is under-sampled. (Same hazard flagged in
`state_tf_encoding_population_geometry`: BG_031's 2-3× point estimate came with
"disengaged under-sampled/noisy".) CLAUDE.md: *"Trial-match conditions when comparing population
responses. Unequal trial counts bias variance estimates."*

**HOW.** The canonical implementation is `scripts/state_tf_learning/b9_engagement.py`:
subsample **each state to the same per-session N** (`FLOOR=80` min trials, `N_MAX=200` cap,
`K_DRAWS` repeats averaged), refit on the **same units in the same session**, and **pair within
session** — so the trial-count attenuation cancels in the paired difference.
Related precedents: `state_conditioned/cluster/tf_glm_state_task.py` (size-matched random control =
the power control); `ws1_changesize_validation.py` §1.3 (RT-matched subsampling);
`ws2_celltype_validity.py` (matched-n subsample panel + Wilson binomial CIs so the smaller group
honestly gets a wider band).

**KILL CRITERION.** The difference disappears at matched N → it was **power, not biology**.
Also: if you *cannot* match (one condition is too thin), the comparison is **not reportable** —
say "under-powered", don't report the point estimate as a trend.

---

## Gate 6 — LEAKAGE / MOVEMENT-MATCHED CONTROLS

**WHAT.** Could the lick (or its ~150 ms motor prep) be inside your analysis window? Could the
"neural" signal simply be movement?

**WHY.** **N1 (the leakage lesson).** A within-FA striatal decode of self-timed lick timing gave
r = **0.559**. After requiring the lick to fall ≥0.25 s *after* the readout window: r = **0.027**
(bootstrap CI over sessions spans 0). Movement-matched partial Spearman: **0.010**. The entire
result was the lick leaking into the window. The ramp-slope readout replicated the collapse
(0.332 → 0.023). Independently, the StimSens-vs-Impulsive change-response "gain effect" was pure
RT leakage — Impulsive hits are ~120 ms faster (RT 0.50 vs 0.63 s, p=3e-11), so the lick leaked
into the (0, 0.25) window.

**HOW.**
- **Any** condition split that differs in RT is leakage-prone. Run the cascade in
  `statesplit_rt_leakage_control.py`: `resp_full` (uncensored) → **`resp_censored`** (per-trial
  window is `(0, min(0.25, RT))` — spikes after the lick excluded) → **`resp_clean`**
  (only trials with RT > 0.25 s, so the whole window is pre-lick in both conditions).
  Report all three; the **clean subset is the verdict**.
- Timing decodes: **leakage-filter first** (`decision_time >= window_hi + 0.25 s`), **then**
  movement-match (partial Spearman as primary; motor-subspace projection secondary).
  Lead with the earliest motor-CD-free window (`scripts/neural_latents/n1_c1b_within_fa.py`,
  `n1_c1c_ramp_slope.py`).
- Divergence-onset control: does the neural split precede the **earliest lick minus ~150 ms**
  motor-prep bound? (`ws1_changesize_validation.py` §1.1, cluster-based permutation.)
- Sensory-vs-motor dissociation: the **stimulus-matched control** —
  compare FA epochs to withholds carrying the SAME stimulus trajectory
  (`visdetect.analysis.psychophysical_kernel.stimulus_matched_control`). B10: sensory component
  flat, gain ramps to the lick → the pre-FA signal is **motor/gain, not sensory**.
- Negative control you should have: an exogenously-timed condition where the answer must be ~0
  (N1's within-HIT r≈0.05 — licks come >6 s later, so a nonzero value would have exposed leakage).

**KILL CRITERION.** Effect collapses once lick-censored / RT-matched / leakage-filtered →
**dead, and it was movement.** Report the honest negative + the collapse magnitude (0.56 → 0.03
is itself a publishable methods lesson).

---

## Gate 7 — YIELD-BIAS + POOLING CAVEATS, WRITTEN INTO THE `_stats` SIDECAR

**WHAT.** Every deliverable writes its own caveats next to its own numbers, so the CSV/TXT sidecar
cannot be read without the caveat.

**WHY.** Yield bias is real and quantified here: narrow/FSI units are **over-sampled** (FSI fraction
BG_046 84% / BG_031 71% / BG_039 43%, vs ~90-95% SPN true composition) because FSIs fire 15.9 vs
6.1 Hz and are easier to sort. Chronic-probe drift moves broad/SPN% from 89% → 15% across stages
(`qc_celltype_yield_jun2026`) → **cross-stage cell-type comparisons are confounded**. A figure that
travels without this caveat will be over-read.

**HOW.** Follow `width_vs_waveform.py` (~L145-149) — append caveat lines into the same `lines`
list that becomes the stats sidecar:

```python
# ── caveats (make the deliverable self-documenting) ──
lines.append(f"YIELD-BIAS CAVEAT: FSI:SPN = {n_fsi}:{n_spn} in the labeled sample — narrow cells are")
lines.append("  OVER-SAMPLED; do NOT read population fractions as biology. The within-sample t2p<->width")
lines.append("  relationship + the independence test (which don't depend on the FSI/SPN marginal) are what matter.")
```

Also self-document: the mixedlm **convergence flag** (`M`/`C` tag, `hardening_continuum.py`), the
**pooling gate** (which sessions/units entered, e.g. `<50% Disengaged`, QC-pass), and **join coverage
asserts** — `assert len(dd), "... matched 0/N consensus members — date_key mismatch?"`. A silent
0-row join from the leading-zero-DAY session-id footgun looks exactly like a null. Canonicalize
through `config.canonical_session_id()` (or `config.session_date_key()` for the subject-prefixed
`tf_responsive` registry — `canonical_session_id` gives **0 overlap** there).

**KILL CRITERION.** Not a kill — a **blocker**: the write-up does not ship until the caveats are in
the sidecar and the joins have coverage asserts.

---

## Gate 8 — ADVERSARIAL REFUTATION PASS

**WHAT.** Spawn **N independent subagents (default 5-6)**, each told to **REFUTE** the claim, each
with a **different lens**, each **independently reproducing the headline numbers from the caches**.

**WHY.** This is what caught the retraction — after I had already reported the result to the user as
solid. A single skeptical self-review did not catch it. The user's standing rule:
*"if so, it is a very important result, so I can't afford to get this wrong."*

> ⚠️ **ALL subagents MUST run on Opus 4.8** — `model: 'opus'` (`claude-opus-4-8`) explicitly on every
> `Agent` call. This is a hard user rule (`feedback_subagent_model_opus`). Never let a subagent
> default to Haiku/Sonnet (the `Explore` agent type defaults to Haiku — override it).

**HOW.** One agent per lens; give each the claim, the exact cache paths, and the script. Lenses:

| # | Lens | Its job |
|---|---|---|
| 1 | **Normalization artifact** | Is the headline raw-Hz? Does a `_z`/normalized column exist? Re-run on it. |
| 2 | **Statistical validity** | Is the test right for the quantity the hypothesis is about (magnitude vs signed)? Effect size? Multiple comparisons? |
| 3 | **Pseudoreplication / pooling** | Refit with session RE + cluster-robust OLS; break out per region and per mouse. |
| 4 | **Does it reproduce** | Independently re-derive every headline number from the CSV caches, from scratch. |
| 5 | **Circularity / leakage / provenance** | Class definition, metric windows, lick in the window, join integrity. |
| 6 | **Alternative explanation** | Yield bias, drift, trial counts, composition/Simpson, exemplar cherry-picking. |

**Decision rule: default to REFUTED if uncertain.** The claim survives only if a **majority fail to
refute** — and any single refuter landing a normalization, circularity, or leakage hit **kills it
outright** regardless of the vote (those are the three that have actually burned this project).

**Record the outcome in the project's existing notation** — `"adversarially verified 0/6 refuted"`
(as in `tf_spectrum_celltype_orthogonality_jul2026`), and carry the refuters' **mandatory writeup
caveats** into the doc (that pass produced: "never cite GMM ΔBIC as pro-bimodality", "FR-control the
coupling metrics", "specify *waveform*-t2p"). Not: "I checked and it's fine."

**KILL CRITERION.** Majority refuted, or any successful normalization/circularity/leakage
refutation → **retract before you ship, not after.** Be transparent and fast about it.

---

## Gate 9 — ONLY NOW: write it up

Once (and only once) Gates 1-8 pass:

1. **Results doc** → `docs/science/YYYY-MM-DD-<topic>-results.md` (or the spec's paired results file;
   one question = one spec + plan + results, per `science_spec_corpus_convention`; index it in
   `docs/science/QUESTION_INDEX.md`).
2. **Figures** → `FIGURES/<topic>/<SUBJECT>/` with the `_stats` sidecar from Gate 7 beside them.
3. **Memory note** → record the verification verdict (`0/6 refuted`), the **mandatory writeup
   caveats**, the kill criteria that were tested, and the reusable methods lesson. If something was
   retracted, say so in the note **at the top**, loudly — that is how the retraction note reads now.
4. **State the claim at the altitude the evidence supports.** "Holds in both regions, all 3 mice,
   survives session RE + rate-matching" is a claim. "Pooled p<0.05" is not.

**An honest, fully-controlled negative is a result** (N1 shipped as one, and its leakage lesson is
more reusable than the effect would have been). A p<0.05 that dies at Gate 1 is not.
