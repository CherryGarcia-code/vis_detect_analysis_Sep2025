# Handoff — Generalize the track-curation → QC-sheet pipeline to multi-subject

**Date:** 2026-06-21
**For:** the chat that built the track-curation pipeline (spec `docs/superpowers/specs/2026-06-07-track-curation-design.md`, plan `…/plans/2026-06-07-track-curation-plan.md`).
**From:** the multi-subject UnitMatch chat.
**Branch:** `main` (all UM work below is merged/landed on `main`).

---

## Why this handoff

UnitMatch now has **full, clean results for four more subjects** (not just BG_046).
The user wants **per-subject curation sheets** (the `trusted`/`review`/`suspect`
QC PDFs from `render_curation_sheets.py`) for these subjects so they can eyeball
track quality. The curation→sheet pipeline you built is exactly the right tool,
but **every CLI is hardcoded to BG_046** — so this is a generalization pass, not
a re-run. Please pick it up and make the modifications.

### UM status (all complete, on ceph + local)
| Subject | UM job | sessions | tracked IDs | notes |
|---------|--------|----------|-------------|-------|
| BG_031  | 3180734 | 43 | 4309 | re-sorted/re-extracted 8 March sessions; clean |
| BG_038  | (earlier) | 43 | 1739 | |
| BG_039  | (earlier) | 32 | 887 | |
| BG_049  | 3180822 | 9 | 163 | 8-digit session names (see gotcha) |
| BG_041  | *pending* | 15 | — | staging→sbatch in progress; same prep applies once done |
| BG_040  | *pending* | — | — | KS4 retry running; UM prep still to come |

UM outputs live at ceph `wEPhys/<SUB>/unit_match/output/all_sessions/`
(`unit_index.csv`, `batch0/output_prob_matrix.npy`, `cell_registry.csv`,
`run_summary.json`). **Note the dir is `all_sessions`, NOT BG_046's `all42`.**

---

## Inputs already in place (local repo, `data/`)

| Subject | `data/pkls/<SUB>` | `data/unit_match/input/<SUB>` (raw wf) | UM output (ceph) |
|---------|-------------------|----------------------------------------|------------------|
| BG_031  | 43 pkls ✅ | 43 dirs ✅ | ✅ |
| BG_038  | 43 pkls ✅ | 43 dirs ✅ | ✅ |
| BG_039  | 32 pkls ✅ | 32 dirs ✅ | ✅ |
| BG_049  | 9 pkls ✅ | 9 dirs ✅ | ✅ |

Raw waveforms are uniform width **within** each subject (BG_049 = 383ch,
BG_041 = 384ch, etc. — a benign per-subject chanmap difference; UM only needs
within-subject consistency, which holds). `cluster_group.tsv` per session lists
exactly the extracted good_and_stable units (curated, no orphans).

**Missing for all four (must be generated or handled):**
- `data/<SUB>_staging_manifest.csv` — none exist (BG_046-only). Without it, sheets
  render sessions as light-grey "Unknown" stage (cosmetic only; the curation
  signal is biophysical/ISI and is unaffected).
- `data/cache/states/<SUB>/{session}_states.csv` — none exist. Generate via
  `make_state_tables.py --provider uniform` (all trials → `in_zone`; no fitted
  HMM needed). Functional corroborator then runs un-engagement-conditioned —
  acceptable for a first look.
- per-subject drift CSV (BG_046 has `FIGURES/tracking_qc/intersession_drift.csv`).
  Run `diagnose_intersession_drift.py` per subject for the drift-corrected depth
  gate, or make `curate_tracks` tolerate its absence.

---

## The generalization work (the actual asks)

All four CLIs default to BG_046 and need a `--subject` path through. Concrete
hardcoded spots found:

### `scripts/pipelines/tracking/render_curation_sheets.py`
- `UM_ROOT` → `.../BG_046/unit_match/output/**all42**` (L43-44) — needs subject + `all_sessions`.
- `DEFAULT_RAW_WF_ROOT`/`DEFAULT_PKL_DIR`/`DEFAULT_OUT_DIR` → `…/BG_046` (L49-51).
- `_session_pkl()` builds `f"**BG_046**_{s}.pkl"` (L64).
- `_session_date()` = `strptime(zfill(8), "%d%m%Y")` (L58-59) — **breaks on 6-digit names** (see gotcha).

### `scripts/pipelines/tracking/curate_tracks.py`
- `UM_ROOT` `all42` (L41-42); `DEFAULT_PKL_DIR`/`DEFAULT_STATES_DIR`/`DEFAULT_DRIFT_CSV`/`DEFAULT_RAW_WF_ROOT` → BG_046 (L45-50).
- `_session_pkl()` `f"BG_046_{s}.pkl"` (L70); `_date_key()` `zfill(8)` slicing (L53-55) — **same 6-digit issue**.
- Uses `load_filtered_manifest` — **check it's subject-parameterized** (it may assume BG_046).

### `scripts/pipelines/tracking/make_state_tables.py`
- `DEFAULT_PKL_DIR`/`DEFAULT_STATES_DIR`/`DEFAULT_TAGS_DIR` → BG_046 (L24-26); `_session_pkl()` `f"BG_046_{s}.pkl"` (L31). Has the `--provider uniform` bootstrap already. Uses `load_filtered_manifest`.

### `scripts/pipelines/tracking/validate_curation.py`
- Not audited line-by-line but almost certainly the same BG_046 defaults — generalize alongside.

### Library modules (likely fine)
`src/visdetect/analysis/track_curation.py` and `state_provider.py` are "pure
functions" per the spec (take data, not paths) — probably subject-agnostic, but
confirm no `BG_046` literals slipped in.

### ⚠️ Gotcha #1 — session-id date format (the one that will bite)
BG_046 sessions are all **8-digit DDMMYYYY** (`01072025`). BG_031/038/039 have a
**mix** including **6-digit DDMMYY** (`BG_031_050325`). The current parsers
`str(s).zfill(8)` then `strptime("%d%m%Y")`, so `050325` → `00050325` →
**invalid date → crash**. Need format-aware parsing (6-digit `DDMMYY` → `20YY`;
8-digit `DDMMYYYY` as-is). **BG_049 is all 8-digit, so it sidesteps this.**

### ⚠️ Gotcha #2 — UM output dir name
BG_046 = `output/all42`; the new subjects = `output/all_sessions`. The registry
is `global_uid` in `unit_index.csv` and pair scores in
`batch0/output_prob_matrix.npy` (+ row-aligned `batch0/unit_index.csv`) — same
schema, just the parent dir name differs.

---

## Suggested order

1. **Pilot on BG_049** — smallest (9 sessions) **and all 8-digit names**, so it
   exercises the `--subject` plumbing without needing the date-format fix.
   Per subject: `make_state_tables --provider uniform` → `diagnose_intersession_drift`
   → `curate_tracks` → `render_curation_sheets --tier trusted` (and `review`).
   Sanity-check the PDFs render.
2. **Fix Gotcha #1 (6-digit dates)**, then run BG_031 / BG_038 / BG_039.
3. **BG_040 / BG_041** later — same recipe once their UM lands.

## First-pass caveats (fine to state in the sheets/readme)
- **Uniform state** (not engagement-conditioned) → the functional corroborator
  runs on all trials, slightly less precise than BG_046's HMM-gated version. A
  per-subject state model can replace it later (the provider is pluggable; spec §4).
- **No staging** → sessions show "Unknown" stage coloring (cosmetic).
- Held-out-ISI AUC per tier (`validate_curation.py`) is the quantitative check —
  worth running per subject to confirm `trusted` separates from `suspect`.

## Pointers
- Spec: `docs/superpowers/specs/2026-06-07-track-curation-design.md` (§3 inputs, §7 outputs, §9 module layout).
- Renderer reuses `qc_sheet_figures.write_uid_pdf` + `build_qc_sheets.compute_uid_metrics`.
- Curation output schema: `curated_tracks.csv` (`curated_uid, kept_sessions, dropped_sessions, confidence_tier`, …) + `curated_links.csv` audit trail.
- Memory: `[[unitmatch-multisubject-jun2026]]`, `[[bg031-reextract-inprogress-jun2026]]`, `[[bg041-049-um-prep-jun2026]]`, `[[neuron_tracking_may2026]]`.
