# DANT cross-session tracking on BG_046

Runs DANT (density-based across-day neuron tracking) on BG_046's extracted Neuropixels
sessions and compares it against the existing UnitMatch tracking. Identity features are
**Waveform + ACG (autocorrelogram) only — no PETH**; the run is multi-shank (4 shanks).
See the design spec at
`docs/superpowers/specs/2026-06-23-dant-tracking-bg046-design.md`.

## Tool

- **pyDANT 1.1.2** (`pip show pyDANT`), the authors' unmodified published package — no
  source patches. It is installed only into the dedicated DANT venv (`./.venv_dant`).

## Environments

Two interpreters are used; do not mix them.

- **Analysis venv** — `E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe`
  (`<ANALYSIS_PY>`). Has `visdetect`. Used for the adapter/input build, the registry
  conversion, and evaluation.
- **DANT venv** — `./.venv_dant/Scripts/python.exe`. Has `pyDANT` (and `hjson`). Used
  **only** to run DANT itself.

## Data reality (this run)

- **41 sessions** went into DANT. Session **`13082025` was dropped**: its RawWaveform sort
  and its pkl spike sort were keyed in mismatched cluster-id spaces, so the (waveform, spike
  train) join was not trustworthy. It is excluded with `--drop-sessions 13082025`; do not
  silently re-add it without re-sorting.
- **5205 pooled units** across the 41 sessions (the full pooled identity matrix).
- **~1226 positive-going units excluded.** DANT trough-centers waveforms and assumes
  negative-going spikes, so `centering_waveforms` is ON (paper-consistent) and positive-going
  units are dropped at build time rather than fed in and mis-centered. 0 units were dropped for
  missing spikes.

## The inert `peth.npy` placeholder

pyDANT 1.1.2 **unconditionally `np.load()`s `peth.npy`** inside
`computeAllSimilarityMatrix`, even when PETH is not a clustering or motion feature. Saving
`None`/an object array there makes the run un-loadable. So `build_dant_inputs.py` writes an
inert zeros placeholder of shape `(n_unit, 1)`. PETH is **excluded from every feature set**
(clustering and motion both use Waveform + AutoCorr), so this array is loaded but never
influences any similarity. **Do not add `"PETH"` to any feature set.**

## Windows notes

- The run needs **UTF-8 stdout** (pyDANT prints a `μm` glyph; the default Windows console
  encoding raises `UnicodeEncodeError`). `run_dant_bg046.py` now **self-hardens** this by
  calling `sys.stdout.reconfigure(encoding="utf-8")` at the top, so no external
  `PYTHONIOENCODING=utf-8` is required. (Setting it still works as a belt-and-braces fallback.)
- Multi-shank diagnostic figures land under `dant_output/Shank<ID>/Figures/` (per-shank
  similarity distributions, feature scatter, motion, matched probability), with the pooled
  `unitLocations.png` under `dant_output/Figures/`.

## Pipeline (run from this worktree root)

```text
<ANALYSIS_PY> = E:/python_analysis/git_repos/vis_detect_analysis_Sep2025/.venv/Scripts/python.exe
<DANT_PY>     = .venv_dant/Scripts/python.exe
```

1. **Build inputs** (analysis venv) ->
   `data/cache/dant/BG_046/input/`
   ```bash
   <ANALYSIS_PY> scripts/tracking_dant/build_dant_inputs.py --drop-sessions 13082025
   ```
2. **Run DANT** (DANT venv; stdout self-hardened to UTF-8 inside the script) ->
   `FIGURES/tracking_dant/BG_046/dant_output/`
   ```bash
   <DANT_PY> scripts/tracking_dant/run_dant_bg046.py
   ```
3. **Convert to registry** (analysis venv) ->
   `data/cache/dant/BG_046/dant_registry.csv`
   ```bash
   <ANALYSIS_PY> scripts/tracking_dant/dant_to_registry.py
   ```
4. **Evaluate** (analysis venv) -> figures + `summary_stats.json`
   ```bash
   <ANALYSIS_PY> scripts/tracking_dant/evaluate_dant.py
   ```

## Results (headline)

| Metric | DANT | UnitMatch |
|---|---|---|
| Tracked clusters | **1022** | — |
| Tracked units (members of a ≥2-session cluster) | **3575** | 6667 |
| Mean tracked length (sessions) | **3.50** | 1.65 (~2.1x longer) |

> **Caveat:** the headline "DANT 3.50 vs UM 1.65 (~2.1x)" survival/mean is each tracker on
> its **OWN unit pool** — DANT's pool is the curated subset (positive-going units and session
> 13082025 excluded), while UM's registry includes them, so the pools differ. The defensible,
> apples-to-apples numbers are the **MATCHED metrics** computed only on the shared
> `(session, ks_unit_id)` units present in both registries: matched mean tracked length
> **DANT 3.49 vs UM 1.68** (n_shared 5194 units), co-membership **ARI 0.152**, and held-out
> **ISI AUC 0.764**. Read the matched numbers as the fair comparison; the own-pool curve in
> `survival_comparison.png` is labelled "(own pool)" with the matched means in a text box.

- **Held-out ISI-fingerprint AUC: 0.764** (8148 matched vs 8148 non-matched pairs) — DANT's
  cross-session co-memberships have ISI fingerprints far more similar than chance, i.e. the
  tracks are functionally consistent on a feature DANT did not cluster on.
- **Co-membership agreement vs UnitMatch: ARI 0.152** (n_shared 5194; pairwise precision 0.258,
  recall 0.108). The two trackers partially agree but are not interchangeable — DANT links more
  units into longer, sparser cluster chains.

Figures live in `FIGURES/tracking_dant/BG_046/`:

- `example_tracks.png` — peak-channel waveform overlaid across sessions for the 6 longest DANT
  tracks (visual sanity check: each panel's waveforms should be consistent).
- `survival_comparison.png` — tracked-length / survival curve, DANT vs UnitMatch.
- `isi_auc.png` — held-out ISI-fingerprint AUC (matched vs non-matched pairs).
- `summary_stats.json` — all numbers above.

## Notes

- Spike times are converted to ms before being handed to DANT.
- Inputs are read from the primary repo (`data/pkls/BG_046`, `data/unit_match/input/BG_046`)
  via absolute paths — **no junctions**. Nothing is written outside this worktree.
- Reproducibility: `np.random.seed(42)` is set in `run_dant_bg046.py` before importing pyDANT
  (DANT does not seed its own motion init / bootstrap).

## Out of scope (follow-ups)

- PETH as a real motion/identity feature (would require a movement-control design).
- Multi-subject (BG_031/038/039/049) DANT runs.
- Mapping DANT clusters onto the existing curation tiers (trusted/review/suspect).

## Curation + QC-sheet rendering (`curate_dant.py`)

Runs DANT's tracks through the project's existing curation + QC-sheet pipeline,
biophysical-only, into a DANT-specific output dir (the UnitMatch curation outputs
are never touched). Run from the worktree root with the analysis interpreter:

    <PRIMARY>/.venv/Scripts/python.exe scripts/tracking_dant/curate_dant.py

Steps (default: all, in order): `registry,curate,validate,render,summary`.
- `registry`  filter `dant_registry.csv` to `dant_uid > 0` -> `dant_registry_curation.csv`
- `curate`    drive `curate_tracks.py` (`--liberal-col dant_uid`, empty states dir
              -> corroborator off, `--drift-source none`) -> `curated_tracks.csv`
- `validate`  held-out ISI AUC by tier, computed IN-PROCESS (the `validate_curation.py`
              CLI hardcodes the UM dir) -> `curation_validation.json`
- `render`    `render_curation_sheets.py --no-pair-scores`: all trusted sheets +
              a capped review sample (`--review-max-uids`, default 25)
- `summary`   tier counts + AUC vs the UM yardstick -> `dant_curation_summary.{csv,png}`

Outputs land under `FIGURES/tracking_dant/BG_046/curation/` (+ `/sheets`).

Pilot a few sheets before the full render:

    ... curate_dant.py --steps render --trusted-max-uids 5

Re-render only (reuse the curate cache):

    ... curate_dant.py --steps render,summary

## Inclusive-trusted re-tiering + dual validation (`inclusive_trusted.py`)

Post-hoc re-tiers the existing `curated_tracks.csv` (NO sweep re-run, Expert-anchored
order preserved) under a looser rule — `trusted = span>=3 AND <=1 warn kept-link`
(bridges allowed) — then validates every tier on two INDEPENDENT axes:
held-out ISI AUC (identity) and functional-PSTH AUC (matched cross-session vs random
within-session, reusing `extract_unit_psths`). Also stratifies matched PSTH similarity
by the earlier session's learning epoch, with a trial-count-robust baseline-only series.

    <PRIMARY>/.venv/Scripts/python.exe scripts/tracking_dant/inclusive_trusted.py

Outputs to `FIGURES/tracking_dant/BG_046/curation/`: `inclusive_trusted_validation.{csv,png}`.

Result (BG_046): shipped-trusted 155 (ISI AUC 0.940) -> inclusive-trusted 287 (0.867);
the 132 newly-promoted tracks sit at ISI AUC 0.818. Functional agreement is modest
everywhere and declines into early learning (partly a trial-count artifact in the
hit-trial-starved Naive sessions) -- lean identity claims on biophysics, treat PSTH
shape as the signal that changes across learning.

Session-token JOIN note: always match registry sessions to `kept_sessions` via
`session_date_key` (NOT raw string ==). `curate_tracks.py` reads the registry without
`dtype=str`, so it writes `kept_sessions` with leading zeros stripped ("8092025"); a
raw-string join against the padded registry silently drops single-digit-day sessions.
