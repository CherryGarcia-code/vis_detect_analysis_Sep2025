# DANT track curation + QC-sheet rendering on BG_046 — Design

**Date:** 2026-06-25
**Branch / worktree:** `feature/dant-tracking` @ `E:/python_analysis/git_repos/vd_dant` (continues the DANT integration)
**Status:** approved design → spec review → writing-plans

---

## 1. Goal

Run DANT's BG_046 cross-session tracks (the `dant_registry.csv` from the prior work) through the project's **existing, registry-agnostic** track-curation + QC-sheet pipeline so we can **visually inspect what DANT actually matched** — per-track 2-page QC sheets (cross-session waveform footprint + ISI + task PSTHs + badges) — organized by a trusted/review/suspect tier, with a held-out ISI AUC per tier as the independent quality signal. Primary deliverable: the rendered sheets. The tiering + AUC are referenced against the existing UnitMatch curation numbers as a light yardstick (no UM re-run).

## 2. What we're reusing (unchanged)

The mapping confirmed the curation/QC pipeline is already registry-agnostic. We drive these **existing** CLIs unchanged (no edits to `scripts/pipelines/tracking/` or `visdetect`):
- `scripts/pipelines/tracking/curate_tracks.py` — Expert→Naive sweep → `curated_tracks.csv` (tiers) + `curated_links.csv`.
- `scripts/pipelines/tracking/validate_curation.py` — held-out ISI AUC by tier → `curation_validation.json`.
- `scripts/pipelines/tracking/render_curation_sheets.py` — per-track 2-page QC PDFs (`qc_sheet_figures.write_uid_pdf`).
All three accept `--registry` + `--liberal-col`; our registry is a near drop-in with `--liberal-col dant_uid`.

## 3. Scope

**In scope:**
- A curation-ready registry (drop `dant_uid <= 0`).
- Curate DANT tracks **biophysical-only** (waveform + depth + ISI; corroborator OFF), into a DANT-specific output dir + cache.
- Validate: held-out ISI AUC per tier.
- Render per-track QC sheets: **trusted** tier in full + a capped sample (`--max-uids 25`) of **review** for spot-checking.
- A small summary (tier counts + AUC-by-tier) that references the existing UM curation numbers as a yardstick.

**Out of scope (explicit):**
- Matched UM re-curation (we only reference existing UM numbers).
- Any functional/PETH feature in identity or curation (corroborator OFF).
- Multi-subject; `--subject` generalization beyond BG_046.
- Edits to the shared pipeline (`scripts/pipelines/tracking/*`, `visdetect/*`), incl. the `build_qc_sheets.py`/`validate_long_tracks.py` `global_uid`-hardcoded path (we don't use it).
- Any `X:`/Samba compute.

## 4. Standing instructions (apply throughout)

1. **Opus 4.8 for every subagent/dispatch.** Never downgrade.
2. **Presentation-ready visualizations.** The rendered sheets + the tier/AUC summary figure are saved under `FIGURES/tracking_dant/BG_046/`.
(Refs: memory `feedback_subagent_model_opus`, `feedback_plain_language_and_save_figures`, `feedback_repo_structure_scripts_figures`.)

## 5. Approach

**Thin orchestration runner, zero changes to the existing pipeline.** A single runner (`scripts/tracking_dant/curate_dant.py`) writes the curation-ready registry and drives the three existing CLIs via **subprocess** (each CLI sets `VISDETECT_SUBJECT` from argv at import, so subprocess with explicit flags is the robust call). (Rejected: adding flags to `curate_tracks.py` — touches shared code; a `_subject_paths` DANT branch — invasive coupling.)

## 6. Architecture & components — `scripts/tracking_dant/curate_dant.py`

All steps run from the worktree root with the analysis interpreter (`E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe`); subprocesses use the same interpreter.

### 6.1 Curation-ready registry (pure, unit-tested)
Read `data/cache/dant/BG_046/dant_registry.csv` with `dtype={"session": str}`, keep rows with `dant_uid > 0`, write `data/cache/dant/BG_046/dant_registry_curation.csv` (columns unchanged: `session, ks_unit_id, dant_uid`). This drops the 1630 untracked/−1 rows so they can't collapse into one bogus mega-track (the pipeline has no uid-value filter, only `--min-span`). Helper: `write_curation_registry(in_csv, out_csv) -> n_kept_rows, n_uids`.

### 6.2 Curate (biophysical-only)
Subprocess `scripts/pipelines/tracking/curate_tracks.py` with:
`--subject BG_046 --registry data/cache/dant/BG_046/dant_registry_curation.csv --liberal-col dant_uid --raw-wf-root <PRIMARY>/data/unit_match/input/BG_046 --pkl-dir <PRIMARY>/data/pkls/BG_046 --states-dir data/cache/dant/BG_046/states_empty --out-dir FIGURES/tracking_dant/BG_046/curation --cache-path data/cache/dant/BG_046/curation_features_dant.pkl --drift-source none --min-span 2 --rebuild-cache`
where `<PRIMARY>` = `E:/python_analysis/git_repos/vis_detect_analysis_Sep2025`. The `states_empty` dir is created empty → `in_zone_trial_indices` returns `[]` → the functional corroborator abstains everywhere → sweep runs on waveform + depth + ISI only. → `curated_tracks.csv`, `curated_links.csv`.

### 6.3 Validate (in-process, NOT via the CLI — clobber-safe)
⚠️ `validate_curation.py` has **no `--out-dir`**: it hardcodes `sjp.curation_out_dir(subj)` (the UM dir `FIGURES/tracking_qc/BG_046/curation/`) and would **overwrite the UnitMatch `curation_validation.json`**. So the runner does NOT call that CLI. Instead it **replicates the small validation loop in-process** (≈25 lines, all public functions): read `curated_tracks.csv`; re-key the curation registry on `dant_uid`; map each kept `(uid, session)` → `ks_unit_id`; load each kept session's pkl once and build the **odd-partition (held-out)** ISI hist via `tc.partitioned_isi_hists(spike_times)`; call `tc.held_out_isi_auc_by_tier(tracks, holdout)`; write the result to `FIGURES/tracking_dant/BG_046/curation/curation_validation.json`. (`curate_tracks.py` and `render_curation_sheets.py` both DO have `--out-dir`, verified, so they're driven via subprocess and write only to the DANT dir.) Output: held-out odd-partition ISI AUC per tier (matched cross-session same-uid vs within-session different-uid).

### 6.4 Render QC sheets
Subprocess `render_curation_sheets.py` per tier with:
`--subject BG_046 --tracks .../curated_tracks.csv --registry .../dant_registry_curation.csv --liberal-col dant_uid --raw-wf-root <PRIMARY>/data/unit_match/input/BG_046 --pkl-dir <PRIMARY>/data/pkls/BG_046 --out-dir FIGURES/tracking_dant/BG_046/curation/sheets --no-pair-scores`
- `--tier trusted` (all trusted tracks).
- `--tier review --max-uids 25` (capped spot-check sample).
→ `FIGURES/tracking_dant/BG_046/curation/sheets/{tier}_uid{u}_span{N}.pdf`.

### 6.5 Summary (light)
A small step prints + saves a tier histogram (trusted/review/suspect counts) and the held-out ISI AUC per tier, and writes a one-figure/CSV summary `FIGURES/tracking_dant/BG_046/curation/dant_curation_summary.{png,csv}` annotated with the existing UM curation yardstick (22 trusted / 567 review / 160 suspect; trusted AUC ≈ 0.96 — referenced from project records, not re-run).

## 7. Critical details / gotchas (from the pipeline mapping)
- **Raw waveforms + pkls live in the PRIMARY repo** (the worktree's `data/` is gitignored/empty). Always pass `--raw-wf-root`/`--pkl-dir` at `<PRIMARY>` — same as the DANT build. No `X:`, no junctions.
- **Biophysical-only = empty states dir.** Point `--states-dir` at a fresh empty dir so the corroborator abstains; do NOT point it at the existing UM `data/cache/states/BG_046` (which would activate the corroborator).
- **Separate out-dir + cache** (`FIGURES/tracking_dant/...` + `curation_features_dant.pkl`) so the UM curation under `FIGURES/tracking_qc/BG_046/curation/` is never clobbered.
- **`--liberal-col dant_uid` on ALL three CLIs** (curate, validate, render) — must be identical (curated_uid keys back to the registry by it). Forgetting it curates the absent `global_uid` → KeyError.
- **`--no-pair-scores` + `--drift-source none`** — DANT has no UM-style prob matrix; the sweep/tiering/AUC never need one (zero degradation; only the cosmetic match-prob bar + match-anchored drift are skipped).
- **Session tokens already 8-digit**; the pipeline's `zfill(8)`-tolerant lookups (`session_pkl`, `load_raw_mean_waveform`, `state_table_path`) resolve them to the same BG_046 pkls/RawWaveforms. Read the registry with `dtype={"session": str}` to preserve leading zeros.
- **PSTHs populate**: BG_046 pkls have real behavioral trials, so page-2 task PSTHs render (unlike trial-less new-subject pkls).
- **Staging manifest absent locally** → stage shows 'Unknown' on sheets (cosmetic only; curation unaffected).
- **Do NOT use `build_qc_sheets.py`/`validate_long_tracks.py`** — they hardcode `global_uid` (not overridable). We use the curation-sweep render path, which honors `--liberal-col`.

## 8. File / directory layout
```
scripts/tracking_dant/
  curate_dant.py        # runner: write curation registry + drive curate/validate/render + summary
  (README.md updated with the curation+render commands)
tests/tracking_dant/
  test_curate_dant.py   # unit test for write_curation_registry (dant_uid>0 filter, columns, dtypes)
data/cache/dant/BG_046/
  dant_registry_curation.csv   # dant_uid>0 only
  curation_features_dant.pkl   # feature cache (gitignored)
  states_empty/                # empty -> corroborator off
FIGURES/tracking_dant/BG_046/curation/
  curated_tracks.csv, curated_links.csv, curation_validation.json
  dant_curation_summary.{png,csv}
  sheets/{tier}_uid{u}_span{N}.pdf
```

## 9. Testing strategy
- **TDD** the pure `write_curation_registry` helper (keeps only `dant_uid>0`, preserves columns, session read as str, returns correct counts).
- **Pilot** the runner on a handful of tracks first: render with `--max-uids 5` (or specific `--uids`) and open one sheet to confirm waveforms/ISI/PSTHs populate, before the full trusted-tier render.
- The curation sweep, AUC, and renderer are already covered by `tests/analysis/test_track_curation.py` and prior use — not re-tested here.

## 10. Risks & open questions
- **No trusted tracks?** If biophysical-only tiering yields few/zero trusted (DANT tracks can be long but waveform/depth/ISI may flag review), we still render the review tier; the summary reports the distribution honestly. (DANT's longest tracks span 18–25 sessions, so some trusted is expected.)
- **Feature-cache staleness:** always `--rebuild-cache` for the DANT run (the cache key is `(uid, session)` and uid meaning differs from any prior UM cache); use the DANT-specific `--cache-path` regardless.
- **Render volume:** trusted could be many PDFs; `--max-uids` caps the review sample. Trusted rendered in full (that's the cohort to inspect).

## 11. Success criteria
- `curated_tracks.csv` + `curation_validation.json` produced for DANT under `FIGURES/tracking_dant/BG_046/curation/`, without touching the UM curation outputs.
- Trusted-tier QC sheets rendered (+ capped review sample), in the same 2-page format as the UM curation sheets, with populated waveform/ISI/PSTH panels.
- A summary (tier counts + held-out ISI AUC by tier) with the UM numbers referenced as a yardstick.
- Honest reporting of the tier distribution and any tracks/sessions skipped.
