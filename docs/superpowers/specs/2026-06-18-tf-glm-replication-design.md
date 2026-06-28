# Per-neuron Poisson GLM replication (Khilkevich-Lohse 2024) — design

**Date:** 2026-06-18
**Status:** spec (awaiting user review → implementation plan)
**Goal:** Replicate the Khilkevich-Lohse 2024 per-neuron Poisson encoding GLM and its TF-responsive identification criteria as faithfully as the BG_046/BG_039 data allow, to test whether DMS (and cortex) carry graded baseline-TF coding that the earlier *instantaneous single-pulse-triggered* method (≈0% across 4 regions) was too blunt to detect.

**Provenance:** `paper_references/Brain-wide dynamics linking sensation to action during decision-making.pdf`, Methods, **p16 "GLM of neural activity"** (and p15 QC criteria, p14 event/recording context). Companion: memory `tf_responsiveness_null_finding_jun2026`, `paper-khilkevich-lohse-2024-brainwide`.

---

## 1. Why this, why now

The triggered |z|>3 fast-vs-slow-pulse metric returned ≈0% TF-responsive across BG_046 DMS / BG_031 striatum / BG_039 cortex / BG_038 GPe — **four regions including cortex at the floor**, which indicts the *metric*, not the biology. Khilkevich-Lohse find **5–45% TF-responsive in nearly every region including basal ganglia** using a per-neuron Poisson GLM that (a) uses the **graded** per-50 ms TF as a regressor (not binarized ±SD tails), (b) reads responsiveness from a **model-prediction correlation + ablation test** (not a raw |z| threshold), and crucially (c) **regresses out lick preparation and lick execution** — and striatal firing is lick-dominated, so an uncontrolled TF signal is buried. This spec rebuilds their method so a TF result here is believable either way.

## 2. The exact paper specification (verbatim-faithful)

### 2.1 Model (paper p16)
- Neural activity binned in **50-ms bins** (= one TF pulse), aligned to trial start. Target = spike **count** per bin.
- **Poisson GLM**, log link, predicting trial-to-trial activity from temporally-unfolded task predictors.
- Each predictor is **"temporally unfolded"**: an independent weight per 50-ms lag across a kernel window prior to and/or post the predictor's timing (a finite-impulse-response / lagged / Toeplitz design). Continuous predictors (TF, motion energy, wheel, pupil) enter as the signal shifted by each lag; event predictors (trial start, change, lick, reward, abort) enter as delta-at-lag indicators; phase enters as 12 phase-bin indicators.
- **19 predictors** with kernel windows:

| # | Predictor | Kernel window | Type |
|---|-----------|---------------|------|
| 1 | Baseline-period TF fluctuations | 0 → 1.5 s | continuous |
| 2 | Trial start | 0 → 1 s | event |
| 3 | Time since baseline start | 1 s-from-trial-start → change onset | ramp |
| 4–9 | Six change onsets (one per change size) | 0 → 2 s | event ×6 |
| 10 | Lick preparation | −1.25 → 0 s (pre-lick) | event |
| 11 | Lick execution | 0 → 0.5 s (post-lick) | event |
| 12 | Air-puff | 0 → 0.25 s | event |
| 13 | Reward | 0 → 0.4 s | event |
| 14 | Abort | −1.25 → 0.25 s | event |
| 15 | Grating phase, upward drift | 12 phase bins (0–360°) | categorical |
| 16 | Grating phase, downward drift | 12 phase bins (0–360°) | categorical |
| 17 | Video motion energy | −0.05 → 0.8 s | continuous |
| 18 | Running-wheel movement | −0.05 → 0.8 s | continuous |
| 19 | Pupil diameter | −0.75 → 0.75 s | continuous |

- **Regularization/fit:** L2 (ridge), GLMnet cyclical coordinate descent, **α = 0**. Per neuron: train on 90%, predict held-out 10%, iterate over **10-fold CV**. The L2 term λ is tuned by an inner **10-fold CV** within each training set.

### 2.2 TF-responsive identification (paper p16, both criteria required)
1. **Refit a reduced model** identical to the full model but with the TF predictor (#1) removed (90% train, 10-fold CV).
2. For each 10% test fold, compute the **mean actual PETH** and **mean predicted PETH** (full *and* reduced) over **−0.15 → 0.75 s around fast and slow TF pulses**, where **fast/slow pulses = TF values ≥/≤ ±0.5 SD of the mean baseline TF**.
3. A unit **significantly encodes baseline TF** iff **both**:
   - **(C1) Shape:** mean (across k-folds) **Pearson r between the full-model-predicted and the actual *fast-minus-slow* TF-pulse response > 0.2**.
   - **(C2) Necessity:** the cross-validated **residual prediction** (full-model TF response minus reduced-model TF response) is **significant: P < 0.01, two-sided t-test, n = 10 independent CV folds**.

(The same template defines lick-prep- and lick-exec-responsive units by ablating #10 or #11 respectively. Note the paper's lick-exec criterion text reuses "no lick preparation kernel" — a typo; the intended ablation is the lick-execution kernel.)

**Note on the confirmation method (re: held-out vs circular-shift):** the paper's significance test is itself **held-out** — both C1 (prediction-vs-actual correlation) and C2 (ablation t-test) are computed on the **10-fold cross-validated** held-out predictions. The paper does **not** use a circular-shift null for TF identification. So replication uses the paper's CV-based C1/C2 verbatim (a held-out-half test is just a coarser k=2 version of this). A circular time-shift of the TF regressor is retained **only as an optional internal pipeline sanity check** (§6.2), to confirm the responsive fraction collapses to the C2 false-positive rate under a null — not as part of the replication.

### 2.3 Secondary GLM-derived measures (optional, paper p16)
- **Focality index** F = Σpₐ² / (Σpₐ)² over areas (pₐ = fraction TF-encoding in area a). Not applicable to our single-region-per-subject design; skip unless pooling regions.
- **TF-kernel peak time & FWHM** (sign-flipped if peak weight negative) — cheap to report once a unit set exists; informative for "sustained vs transient" TF coding. **Include as a secondary readout.**
- **Change-kernel ramp slope vs change size** — the evidence-accumulation ramp; defer (decision-spine territory).

## 3. What we implement on BG_046 / BG_039 (faithful-reduced)

User decision (2026-06-18): **Option 1 — licks + running wheel as the motor control**, build now, note the video gap as a limitation. Confirmed data facts drive the reduced set:

| # | Predictor | Status | Source / note |
|---|-----------|--------|---------------|
| 1 | **Baseline TF** | ✅ include | `trial.baseline_values` (`St1TrialVector`, 1800-buf, **stride 3** → 50 ms; sample_period 0.05 s), truncated by outcome. **Upgrade option (preferred if clock-aligned):** the raw `Session/*trials.json` also stores the **actually-displayed** per-frame `TF` (787 frames/trial) with `vbl` flip timestamps — more faithful than the planned buffer. Use displayed-TF if vbl→nidaq alignment is clean; else planned buffer (they should agree closely). |
| 2 | Trial start | ✅ | `ni_events.Baseline_ON` |
| 3 | Time-since-baseline ramp | ✅ | derived; **window longer than paper** (BG_046 baseline min 6.0 s / median 7.3 s / max 18.2 s vs their ~3 s) |
| 4–9 | **Six change onsets** | ✅ | `change_size ∈ {1.0, 1.25, 1.35, 1.5, 2, 4}` (catch=1.0 is the 6th, matching the paper). **Onset time = `ni_events.Change_ON`** (accurate nidaq timestamp). Verified: `Change_ON` is finite exactly for **Hit/Miss/Ref** (271/634, change reached — incl. completed catch trials, e.g. 59 catch with finite `Change_ON`) and **NaN for FA/abort** (363, truncated before the change). So the regressor fires on finite-`Change_ON` trials only, grouped by `change_size`; FA/abort are **N/A** (no change regressor). (Cross-check: `Change_ON` − (`Baseline_ON`+`change_time`) median 5 ms.) |
| 10 | **Lick preparation** | ✅ | `ni_events.Piezo_1` → lick-**bout onsets** (gap-split); −1.25→0 s kernel |
| 11 | **Lick execution** | ✅ | `Piezo_1` licks; 0→0.5 s kernel |
| 13 | Reward | ✅ include | `ni_events.Valve_L` finite entries (~171 ≈ hit count); 0→0.4 s |
| 14 | Abort | ✅ | `abort`-outcome trial end times; −1.25→0.25 s |
| 18 | Running wheel | ✅ include | `ni_events.Rot_enc_A/B` quadrature → speed signal (continuous); −0.05→0.8 s |
| 12 | Air-puff | ❌ drop | not used in BG_046 training (empty in pkls) — confirmed |
| 15–16 | **Grating phase** | ✅ include via raw extraction | **Confirmed present in raw** `Session/*trials.json` as per-frame `phase` (787×2) with `vbl` timestamps (vbl = Psychtoolbox vertical-blank flip times) — extract directly (user preference over computing the integral). Requires an **ingest enhancement + pkl remake**, done **upfront** (user decision 2026-06-18: cheap, no reason to postpone), so **both DMS and cortex runs include phase from the start**. |
| 17 | Video motion energy | ❌ omit | cameras unprocessed (no ME); **the main deviation** — partially proxied by lick prep/exec + wheel |
| 19 | Pupil | ❌ omit | no video |

**Resulting design:** ~9 predictor groups (the 6 change sizes count as 6 event regressors), each unfolded over its lag window → a few-hundred-column design matrix per session. TF-responsive identification (C1+C2) is applied **exactly** as in §2.2.

## 4. Python implementation approach

- **Library module** `src/visdetect/analysis/tf_glm.py` (new), CLI/driver `scripts/tf_responsiveness/run_tf_glm.py`. Co-locate with the existing `tf_selectivity.py` / batch tooling. Build in the **`vd_tf_phase0` worktree** (branch `feature/tf-responsiveness-labeler`) to stay isolated from parallel chats; set `PYTHONPATH=<worktree>/src`.
- **Design matrix:** one row per 50-ms bin spanning each trial (trial start → trial end, truncated by outcome). Build per-predictor lagged columns; concatenate trials within a session. Standardize continuous regressors (TF, wheel) to unit variance; leave indicators as 0/1.
- **Fitter — faithful ridge-Poisson:** `sklearn.linear_model.PoissonRegressor(alpha=λ, fit_intercept=True, max_iter=…)` is L2-penalized Poisson (log link), the convex equivalent of GLMnet α=0. Outer **10-fold CV** for prediction; inner **10-fold CV** grid-search over λ (e.g. logspace) per training fold. (Fallback if convergence/scale issues: statsmodels GLM Poisson with L2, or a small IRLS-ridge.) Penalize coefficients but **not** the intercept (matches glmnet).
- **PETH evaluation:** for each test fold, build actual & predicted spike-count PETHs on −0.15→0.75 s windows around fast/slow pulses (±0.5 SD), average within fold; compute the fast−slow differential PETH; **C1** = mean-over-folds Pearson r(predicted_diff, actual_diff) > 0.2; **C2** = t-test over 10 folds of (full−reduced) residual TF prediction, P<0.01.
- **Outputs:** per-unit table (`is_tf_responsive`, C1 r, C2 p, kernel peak-time, FWHM, full-model CV pseudo-R²/deviance), per-session/region TF-responsive fraction, exemplar kernel + actual-vs-predicted PETH figures. Cache to `data/cache/tf_glm/`, figs to `figures/tf_responsiveness/glm_*`.
- **Units:** `get_good_cluster_ids` / `good_and_stable_ids`; the paper's QC (0.5 Hz floor + stability + ISI) is essentially `find_good_stable_units`, already applied to the pkl pool.

## 5. Deviations from "exact" and their scientific implications (believability)

| Deviation | Direction of risk | Mitigation / interpretation |
|-----------|-------------------|------------------------------|
| **No video motion-energy / pupil** (#17, #19) | Less complete motor/arousal control → could leave residual movement variance that *inflates* apparent TF (false positives) OR adds noise that *reduces* power | Lick prep/exec + wheel capture the dominant motor confound; if TF is **null even with these**, the null is strong (motor control only gets stricter with ME). If TF is **positive**, follow up by adding ME (process video for those sessions) before claiming a fraction. |
| **Grating phase** (#15–16) — *included* via raw extraction | Was the main remaining gap; now closed (phase stored per-frame in raw) | Extracted directly from raw `phase`+`vbl` (not reconstructed). Only deviation is the DMS-first build may run before the pkl remake; phase added before/for the cortex run where it matters. |
| **Earliest change ~6 s vs ~3 s** | None for model structure | Longer baseline = *more* TF pulses/trial → **better** TF estimation; ramp predictor window simply longer. |
| **GLMnet → PoissonRegressor** | Same convex objective (ridge Poisson); solver differs | Verify on a held-out unit that λ-path behaves sensibly; report CV deviance. |
| Single region per subject | No focality index | Report per-region fractions directly; compare DMS vs cortex. |

**Headline guardrail:** a positive TF fraction with licks+wheel control is *suggestive*; a fraction that **survives adding video motion-energy** is *confirmatory*. A null that holds *with* lick control is a **strong** null (unlike the triggered-metric null). Either outcome is publishable-grade evidence about the original question.

## 6. Validation / sanity checks (before trusting fractions)

**Primary positive control — reproduce the paper on the paper's own data (gold standard).** The Khilkevich-Lohse brain-wide dataset is pre-converted to Python-friendly form on **ceph = `X:/public/`** at:

```
X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted/
  <animal_id>/<session_name>/
      trials.parquet    # FSM behavioural data (incl. per-frame TF/phase/vbl)
      daq.parquet       # nidaq events
      neural.parquet    # spike data
      movement.pkl      # motion energy, pupil size
```

Crucially, **`movement.pkl` provides motion energy + pupil**, so on *their* data we can run the **fully faithful 19-regressor model** (all motor controls present) — not the reduced set. (Verified the dir is even richer: also `stim.csv` = per-frame TF/phase/vbl, `running.csv` = wheel, `licks.csv`, `clusters.csv` = region labels, and CSV mirrors of the parquets — every regressor is available.) Plan: write a thin `pandas.read_parquet` loader adapter, run **this exact GLM** on a few of their regions (a visual-cortical area + a basal-ganglia area), and confirm we **recover their reported 5–45% TF-responsive fractions**. This validates the *implementation* independent of BG_046 — the most convincing "what we built works" check — and de-risks the reduced-set interpretation on our data. **Recommended order: validate on their data first, then apply to BG_046/BG_039.**

**Internal checks:**
1. **Lick-prep/exec recovery:** the same pipeline should flag the known ~72–83% lick/motor-responsive units (sanity that the GLM + ablation test work).
2. **Optional shuffle null:** circularly shift the TF regressor relative to spikes → TF-responsive fraction should collapse to ≈ the C2 false-positive rate (~1%). (Pipeline sanity only; *not* part of the paper's identification — see §2.2 note. The paper's own confirmation is the held-out 10-fold CV in C1/C2.)
3. **Cortex expectation:** BG_039 visual/association cortex should show a **higher** TF fraction than DMS if the method has power (the paper finds cortex high); if cortex is *also* ≈0% with this method, suspect an implementation bug, not biology. (Subsumed by the ceph positive control if that runs first.)
4. **Pulse-count adequacy:** with ~6 s baselines, confirm ≥ a few-thousand fast and slow pulses per session feed the PETH evaluation.

## 7. Deliverables
- `tf_glm.py` (model + identification) with unit tests (synthetic neuron with injected TF kernel must be recovered; lick-only neuron must be TF-negative).
- `run_tf_glm.py` batch driver (per-subject, reverse-chrono sessions).
- Per-unit / per-region TF-responsiveness tables + exemplar figures for BG_046 (DMS) and BG_039 (cortex).
- A short results note + memory update; if positive, this **revives the TF direction**; if null-with-lick-control, it **closes the TF door cleanly**.

## 8. Open items / prerequisites
- **Phase extraction (ingest enhancement + pkl remake) — DONE UPFRONT (Task 1):** add per-frame `phase` (787×2), `vbl` flip times, and displayed per-frame `TF` from `Session/*trials.json` to the Trial object (currently dropped by `ingest.py`, which reads only `St1TrialVector`/`Stim2Ori`/`stimD`/`stimT`). Needs `vbl` (Psychtoolbox vertical-blank clock) → nidaq/spike clock alignment (anchor on the trial-start frame; cross-check against `Baseline_ON`). Remake the analysis-set pkls (BG_046 manifest sessions + BG_039) **before** any GLM run so both DMS and cortex include phase. **Shared-data safety (parallel chats):** the change is purely *additive* (new Trial fields; old code ignores them) — but to avoid disturbing other chats that read the live pkls, re-ingest to a **staging dir**, validate additive-only against reference pkls with `scripts/conversion/validate_pkl.py`, then swap in. Do **not** delete/junction-overwrite blindly (cf. the June-2026 data-loss incident).
- **Khilkevich-Lohse positive-control dataset (path resolved):** `X:/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted/<animal>/<session>/{trials,daq,neural}.parquet + movement.pkl`. Write a `pandas.read_parquet` loader adapter; full 19-regressor model feasible (movement.pkl has ME+pupil). See §6.
- Decide lick-bout-onset gap threshold for lick-prep events (inspect Piezo ISI distribution).
- Rot_enc A/B → speed decoding (quadrature decode + smoothing kernel).
