# Normalization — Audit & Fix History

This is the **audit record and fix changelog** for normalization across the
analysis code. For the *how-to* — when and how to normalize — see the
"Normalization Best Practices" section of `CLAUDE.md`, which is the canonical
guide. This document records how the codebase was brought into compliance.

Consolidates four March 2026 reports (`NORMALIZATION_AUDIT_MARCH2026.md`,
`NORMALIZATION_FIX_SUMMARY.md`, `NORMALIZATION_FIX_IMPACT.md`,
`NORMALIZATION_DECODING_FIX_MARCH30.md`), previously in `analysis_suite/`.

## Current status (as of 2026-03-30)

**Grade: A+ — 0 normalization issues.** All active analysis scripts in
`analysis_suite/` and `src/visdetect/analysis/` follow correct shared-baseline,
normalize-then-average practice with division-by-zero guards.

## The golden rules

1. Normalize each unit separately, using a **shared baseline** across all
   compared conditions.
2. Normalize **then** average — never average then normalize.
3. Guard against division by zero (`if sd < 1e-6: sd = 1.0`).
4. Use consistent baseline windows, imported from `constants.py`.
5. Match the method to the task (z-score for heatmaps, Δrate for coding
   directions, etc.).

## Fix history

### 2026-03-23 — Population CD circular-baseline bug (3 scripts)

**Bug**: each outcome type was z-scored to *its own* baseline —
`hit → hit baseline`, `fa → fa baseline`, `cr → cr baseline`. This forces every
baseline to zero, erasing biologically real pre-stimulus differences between
trial types and inflating low-activity conditions (FA's low SD → inflated z).

**Fix**: compute the baseline once from the highest-signal condition
(Hit / Hit_big / Go_big) and normalize *all* conditions to that shared baseline,
preserving relative magnitude.

Scripts fixed:
- `03_population/a_coding_direction.py` — Panel D (change-aligned), Panel F (lick-aligned)
- `03_population/d_state_matched_cd.py` — two-pass shared baseline (pass 1 pools
  Hit traces for mu/sd; pass 2 normalizes fa / hit_small / hit_big)
- `03_population/e_sensory_dose_response.py` — same two-pass pattern (Go_big ref)

A per-outcome fallback (with warning) is retained if shared-baseline computation
fails (insufficient samples or zero SD).

### 2026-03-23 — Validation of the CD fix

Re-ran `a_coding_direction.py` (25/26 manifest sessions). The core finding
survived the fix: **CD strength increases with learning** — Spearman ρ = 0.515,
p = 0.0085; Kruskal-Wallis by stage p = 0.012. That the learning effect remains
significant after correcting the bug confirms the result is robust, not a
normalization artifact.

### 2026-03-30 — Full audit (43 active scripts)

Audited all active analysis scripts in `analysis_suite/` and
`src/visdetect/analysis/` (archive/legacy excluded). Result: **0 critical
issues, 1 moderate**. Core utilities (`compute_zscore_normalized`,
`compute_baseline_subtracted`, `tf_pulse._zscore_trace`) verified sound —
baseline pooled across all trials (no circular baseline), per-unit stats,
division-by-zero guards present. Grade: A−.

The one moderate issue: the decoding scripts trained on raw firing rates with a
per-time-bin `StandardScaler`, which can inflate decoder performance in baseline
periods.

### 2026-03-30 — Decoding scripts fix → A+

Resolved the moderate issue by normalizing to a shared baseline before decoding:
- `04_decoding/a_hit_miss_decoding.py` — z-score Hit/Miss (and FA transfer)
  tensors to baseline `(-0.5, -0.05)` s
- `04_decoding/b_change_size_decoding.py` — z-score Big/Small tensor, same window
- `04_decoding/c_state_decoding.py` — Δrate (`compute_baseline_subtracted`) to an
  early pre-trial window `(-1.5, -1.0)` s; Δrate chosen because pre-trial
  activity is tonic, not stimulus-evoked

No breaking changes — `StandardScaler` still runs, now scaling Δz values. The
Codebase Auditor skill gained a 5-point normalization checklist; `CLAUDE.md`
gained the "Normalization Best Practices" section. Grade: A− → **A+**.

## Audit findings — per-file status (2026-03-30)

| Status | Files |
|--------|-------|
| ✅ Clean (9) | `analysis_suite/utils.py`, `src/visdetect/analysis/tf_pulse.py`, `03_population/a_coding_direction.py`, `03_population/d_state_matched_cd.py`, `03_population/e_sensory_dose_response.py`, `03_population/b_population_psth_heatmap.py`, `02_single_unit/a_responsiveness_screen.py`, `06_lick_motor/a_fa_neural_signatures.py`, `07_advanced/f_fa_subtype_lick_triggered_tf.py` |
| ⚠️ Fixed 2026-03-30 (3) | `04_decoding/a_hit_miss_decoding.py`, `04_decoding/b_change_size_decoding.py`, `04_decoding/c_state_decoding.py` |
| 🔴 Critical | none |

Notes: `a_responsiveness_screen.py` correctly uses per-trial paired differences
with a paired sign-flip permutation test (no circular baseline — each trial's
baseline is independent). `f_fa_subtype_lick_triggered_tf.py` applies no
normalization by design — it averages a stimulus feature (TF in log2 units), not
neural activity.

## Method quick reference

| Method | Formula | Use for | Pitfall |
|--------|---------|---------|---------|
| Z-score | `(r − μ_bl) / σ_bl` | single-unit significance, heatmaps, population comparisons | inflates low-FR units if σ_bl tiny |
| Δrate | `r − μ_bl` | population averages, coding directions, decoding | still biased toward high-FR units |
| Percent change | `100·(r − μ_bl) / μ_bl` | modulation-strength comparison | explodes for low baselines |
| Raw rates | `r` | within-unit, single-trial decoding | cannot compare across units |

## Pre-publication checklist

- [ ] All compared conditions use a **shared baseline**.
- [ ] Baseline windows imported from `constants.py` (not hardcoded).
- [ ] Units normalized **before** averaging.
- [ ] Division-by-zero guarded (`if sd < 1e-6: sd = 1.0`).
- [ ] Normalization method matches the analysis goal.
- [ ] Decoding uses baseline-normalized features, not raw rates.
- [ ] Cross-session grand averages use a shared baseline (preserve relative magnitude).
