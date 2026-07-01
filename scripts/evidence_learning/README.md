# B10 — Impulsivity kernel across learning (evidence_learning)

Behavioral (I1) + neural (N-B) Orsolic-style reverse-correlation kernel, 3 mice
(BG_046/039 DMS, BG_031 VMS). What stimulus pattern of baseline TF fluctuation
does the mouse mistake for a real change (→ an impulsive early lick), and how
does it — and its neural echo in TF-responsive striatal cells — change with
learning and behavioral state?

Spec: `docs/superpowers/specs/2026-07-01-B10-impulsivity-kernel-learning-design.md`
Plan: `docs/superpowers/plans/2026-07-01-B10-impulsivity-kernel-learning-plan.md`

## Layout
- Library: `visdetect.analysis.psychophysical_kernel` (pure estimators) +
  `visdetect.analysis.evidence_learning_io` (multi-subject loaders).
- Scripts (this dir): coverage gate + 3 figure builders.
- Tests: `tests/analysis/test_psychophysical_kernel.py` (13 incl. synthetic
  recovery) + `tests/scripts/test_b10_scripts.py` (path-loaded smoke tests).

## Run order (real data — needs local `data/pkls`, `data/cache/tf_responsive`,
## `data/cache/state_tags`, `data/<subj>_staging_manifest.csv`)
```
py scripts/evidence_learning/b10_phase0_coverage.py    # coverage/usable gate
py scripts/evidence_learning/b10_phase1_behavioral.py  # Fig B10.1
py scripts/evidence_learning/b10_phase1_neural.py      # Fig B10.2
py scripts/evidence_learning/b10_phase2_state.py       # Fig B10.3
```
Outputs: `FIGURES/evidence_learning/<pool>/*.png`, `data/cache/evidence_learning/*.csv`.

**Worktree note:** to run against the primary checkout's real data without
junctions, run from the primary repo root with the worktree's src on the path:
`PYTHONPATH=<worktree>/src <primary>/.venv/Scripts/python.exe scripts/evidence_learning/<script>.py`.
Never `git worktree remove` while data junctions are live (primary-data-loss risk).

## Method (one line each)
- **Kernel** = FA-triggered log2-TF (stride-3, 50 ms grid) minus time-in-trial-
  matched no-lick withhold; bootstrap CI; n-matched Naive-vs-Expert; shape
  (half-width/peak-lag) reported separately from amplitude.
- **Neural** = signed population TF signal on TF-responsive cells
  (`sign(c1_r_log2)·z`), FA vs withhold, with a stimulus-matched sensory-vs-gain
  decomposition. Pooled by subject (DMS 046+039 / VMS 031), per-session then
  aggregated. Region labels provisional (`region_bank_confirmed` False registry-wide).
- **State** = same kernel split StimSens vs Impulsive (conf ≥ 0.8). NON-CIRCULAR:
  labels use lick rates/outcomes; the kernel shape is independent.

## Honest limitations (printed on figures)
- No video → behavioral kernel = "stimulus history preceding impulsive licks,"
  not pure sensory evidence.
- VMS is n=1 region; BG_039 Learning = 1 session (Naive/Expert only).
- Naive-StimSens is the thinnest Phase-2 cell (neural especially, ~2-3 TF cells/session).
- Lag axis is "time before RECORDED lick" (no calibrated hardware delay; a
  constant shift cancels in every learning/state contrast).
- Nulls (flat kernel / no learning or state change) are pre-registered as reportable.
