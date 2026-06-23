# DANT cross-session neuron tracking on BG_046 — Design

**Date:** 2026-06-23
**Branch / worktree:** `feature/dant-tracking` @ `E:/python_analysis/git_repos/vd_dant`
**Status:** approved design → spec review → writing-plans

---

## 1. Goal

Run **DANT** (Density-based Across-day Neuron Tracking; Huang et al., *Patterns* 2026,
doi:10.1016/j.patter.2026.101590; pkg `pyDANT`) on **BG_046's 42 already-extracted sessions**
and produce an honest read-out of **how well we can track single units across sessions**,
benchmarked against our existing UnitMatch 3.2.9 registry.

This is a first, deliberately-scoped run: one subject, waveform + spike-timing identity only,
local execution, with a clear quality verdict at the end.

## 2. What DANT is (one paragraph)

DANT pools every unit from every session into one set and tracks by **global density
clustering** (HDBSCAN) over a fused, Fisher-z, LDA-weighted similarity matrix, rather than
UnitMatch's pairwise-probability + chaining. It alternates (a) HDBSCAN clustering with (b)
inter-session probe-drift estimation + Kriging waveform re-rendering, iterating until matched
pairs plateau, then a final clustering pass. A cluster (curated to ≤1 unit/session) = one
tracked neuron. Per the paper it beats UnitMatch on yield (~8.6 vs 4.5 mean tracked sessions)
at comparable/lower false-positive rate, and is fast (>10k units < 1 h). Full method notes:
project memory `pydant_dant_tracking_reference.md`.

## 3. Scope

**In scope (this deliverable):**
- BG_046, all 42 sessions present under `data/unit_match/input/BG_046/`.
- Identity features: **Waveform + Autocorrelogram (ACG) only — NO PETH** (matches DANT's own
  identity definition and the paper's benchmarks; directly comparable to UnitMatch; avoids the
  functional-circularity caveat).
- Multi-shank run (`runDANTMultiShank`, NP2.0 4-shank).
- Validation triad + visualizations; comparison to the UnitMatch baseline.

**Out of scope (explicit follow-ups, do NOT build now — YAGNI):**
- PETH / functional features (would be motion-aid + validation only, never identity).
- Multi-subject / `--subject` generalization (BG_031/038/039/049).
- Mapping DANT clusters into our curation tiers / `track_verdict` / GLT.
- Any modification to DANT itself.

## 4. Standing instructions (apply throughout)

1. **Opus 4.8 for everything.** Every subagent / Workflow dispatch (implementers, reviewers,
   Explore, final review) uses Opus 4.8. Never downgrade to a cheaper/faster tier, even when a
   skill suggests it. State this in every boot prompt. (Ref: memory `feedback_subagent_model_opus`.)
2. **Visualize at every helpful step.** Each stage saves presentation-ready figures (not
   internal-only), with plain-language labels. Minimum set: input QC (units/session,
   depth-vs-session, shank-coloured channel map), DANT-native diagnostics, UnitMatch comparison,
   and rendered example tracks. (Refs: memory `feedback_plain_language_and_save_figures`,
   `feedback_repo_structure_scripts_figures`.)

## 5. Approach

**Chosen: pip-install `pyDANT` (unmodified) + a thin repo-side adapter + evaluation harness.**
DANT stays a black box installed in a dedicated env. We write only: an input adapter, a run
wrapper + `settings.json`, a registry converter, and an evaluation/visualization harness.
(Rejected: vendoring/forking DANT — maintenance burden; reimplementing — large, no benefit.)

## 6. Architecture & components

### 6.1 Environment & isolation
- Work happens in this worktree (`feature/dant-tracking`), isolated from parallel chats.
- Dedicated venv **`.venv_dant`** (py3.10 — satisfies pyDANT ≥3.9; no conda on this machine):
  `pip install pyDANT`. Kept separate from the analysis `.venv` so `hdbscan`/numpy pins can't
  disturb it. **Only** the `runDANTMultiShank` call runs in `.venv_dant`; the adapter and
  evaluation run in the analysis `.venv` (they need `visdetect` to read pkls).
- Gitignored inputs reach the worktree via **directory junctions**: `data/pkls/BG_046/` and
  `data/unit_match/input/BG_046/` (read-only use). **Delete junctions BEFORE any `rm -rf` or
  `git worktree remove`** (ref: memory `worktree_realdata_inputs_junctions` + the June-2026
  data-loss incident). Output dirs are real dirs in the worktree.
- The editable `visdetect` install resolves to the PRIMARY repo `src` (main). We do **not**
  modify `visdetect`, so this is correct; if that ever changes, set `PYTHONPATH=<worktree>/src`
  (ref: memory `worktree_editable_install_pythonpath`).
- Runs **locally** (~4,500 units total → well under the paper's 10k-units/<1h; no SLURM, no
  `X:` compute — all inputs are local `.npy`/`.pkl`).

### 6.2 Input adapter — `scripts/tracking_dant/build_dant_inputs.py`
Converts our extracted data into DANT's input layout under `data/cache/dant/BG_046/input/`.
- **Session ordering:** chronologically sort the 42 extracted sessions → `session_index` ∈ 1..42
  (1-based, contiguous — DANT requires this).
- **Per session, per good unit** (`cluster_group.tsv` "good" == extracted RawWaveforms):
  - Waveform: load `RawWaveforms/Unit{ks}_RawSpikes.npy` `(82, 383, 2)`; **mean over the CV axis**
    → `(82, 383)`; transpose → `(383, 82)` = (channel, sample). Stack → `waveform_all.npy`
    `(n_unit, 383, 82)`, float. Units are raw-averaged µV (unwhitened) — already what DANT needs.
  - Spike times: pull from the session pkl cluster with that ks id; **×1000 → milliseconds**;
    write `spike_times/Unit{k}.npy` (k = global pooled index).
- **Geometry:** `channel_locations.npy` ← `channel_positions.npy` `(383, 2)`. Assert all 42
  sessions share identical geometry (chronic probe; they should).
- **Shanks:** derive `channel_shanks.npy` `(383,)` from the 4 x-position groups
  (`{27,59}→0, {277,309}→1, {527,559}→2, {777,809}→3`).
- **Lookup:** write `unit_lookup.csv` (pooled_index, session_index, session_name, ks_unit_id)
  so DANT clusters map back to (session, ks_unit_id).
- **Guards (fail-loud / log-and-skip):** unit in `cluster_group` but no RawWaveform file or no
  pkl cluster → skip + log; CV axis size must be 2; per-session unit count must match
  RawWaveforms count; **flag/exclude positive-going units** (peak channel max > |min|) before
  centering.

### 6.3 DANT config + run — `scripts/tracking_dant/run_dant_bg046.py` + `settings_bg046.json`
- `runDANTMultiShank` (4-shank).
- `centering_waveforms = true` (trough-align per peak channel; removes alignment artifacts —
  see §8).
- `spikeLocation`: `monopolar_triangulation`, `n_nearest_channels = 20`.
- `waveformCorrection`: `n_nearest_channels = 38`, `linear_correction = false` (**rigid — the
  authors' default**; they enable non-rigid/depth-linear only for recordings spanning >1 mm with
  depth-dependent drift, e.g. their mPFC case, and warn it overfits on low-quality data. BG_046's
  active span is ~705 µm (1515–2220 µm) < 1 mm, so rigid is the correct default; revisit only if
  diagnostics show clear differential drift), `n_templates = 2`.
- `autoCorr`: 300 ms / 1 ms / σ5 ms. `ISI`: unused (ACG covers it).
- `motionEstimation.features = [["AutoCorr"], ["Waveform","AutoCorr"]]`, `max_iter = 15`,
  `repeat_last_feature_set = true`, `stop_early = true`.
- `clustering.features = ["Waveform","AutoCorr"]`, `max_distance = 100` µm, `n_iter = 10`.
- `autoCuration.auto_split = true`.
- **Reproducibility:** set `np.random.seed(42)` before the run (DANT does not seed its motion
  init / bootstrap).
- Output → `FIGURES/tracking_dant/BG_046/dant_output/` (DANT's `Output.npz`, `IdxCluster.npy`,
  per-feature similarity matrices, motion, and DANT's own `Figures/`).

### 6.4 Output → comparable registry — `scripts/tracking_dant/dant_to_registry.py`
Read `Output.npz` (`IdxCluster`, `Sessions`, `MatchedPairs`) + `unit_lookup.csv` → emit
`dant_registry.csv` with columns `session, ks_unit_id, dant_uid` (−1 = untracked) — the same
long shape as UnitMatch `unit_index.csv`, enabling direct comparison.

### 6.5 Validation & visualization — `scripts/tracking_dant/evaluate_dant.py`
Answers "how well can we track":
1. **DANT-native diagnostics** (surface the package's figures): tracked-length survival,
   match-probability vs Δsession, estimated motion, matched/unmatched similarity separation.
2. **Head-to-head vs UnitMatch baseline** (prefer the **local** `data/unit_match/output/BG_046_um329_CellRegistry.csv`; the canonical `unit_index.csv` lives off-repo on `X:` and should not be read during compute):
   tracked-unit count, mean/median tracked length, survival-curve overlay, and **co-membership
   agreement** (Adjusted Rand Index + pairwise precision/recall of "same neuron?") on the shared
   unit set.
3. **Held-out CV-split ISI-fingerprint AUC** (our trusted independent identity metric): matched
   cross-session pairs (same `dant_uid`, different session) vs within-session non-matched pairs.
   Reuse `tracking_qc` ISI machinery where possible.
4. **Rendered example tracks**: waveform + ACG across sessions for a sample of trusted clusters
   (sanity).
Figures → `FIGURES/tracking_dant/BG_046/`; stats CSVs alongside.

## 7. Data contracts & key gotchas

| Item | Our data | DANT needs | Adapter action |
|---|---|---|---|
| Waveform | `RawWaveforms/Unit{ks}.npy` (82, 383, 2) raw-avg µV | `waveform_all.npy` (n_unit, n_ch, n_samp) unwhitened µV | mean CV axis → transpose → stack |
| Spike times | pkl cluster, **seconds** | `spike_times/Unit{k}.npy`, **milliseconds** | **×1000** |
| Channels | `channel_positions.npy` (383,2) | `channel_locations.npy` (n_ch,2) | copy; assert identical across sessions |
| Shanks | (missing) | `channel_shanks.npy` (n_ch,) for multishank | derive from 4 x-groups |
| Session id | dir name (DDMMYYYY) | `session_index.npy` 1..N contiguous | chronological map + `unit_lookup.csv` |

## 8. Why `centering_waveforms = true`
DANT's waveform similarity is a sample-by-sample Pearson correlation; if the trough lands at a
different time-sample in two snippets, the correlation drops even for the same neuron. Centering
trough-aligns each waveform on its peak channel (docs: same-unit r 0.29 → 0.89). Assumes
negative-going spikes; positive-going units are flagged/excluded by the adapter (§6.2).

## 9. File / directory layout
```
scripts/tracking_dant/
  build_dant_inputs.py        # our data -> DANT input layout
  run_dant_bg046.py           # loads settings, runs runDANTMultiShank (in .venv_dant)
  settings_bg046.json         # DANT config (Waveform+ACG, no PETH, multishank)
  dant_to_registry.py         # Output.npz -> dant_registry.csv
  evaluate_dant.py            # validation triad + figures
  README.md                   # how to run, env, junction setup/teardown
data/cache/dant/BG_046/input/ # waveform_all, session_index, channel_*, spike_times/, unit_lookup
FIGURES/tracking_dant/BG_046/ # diagnostics, comparison, example tracks (+ dant_output/)
```

## 10. Testing strategy (TDD)
Unit-test our code (DANT itself is the package, not ours to test):
- Adapter: CV-collapse + transpose correctness (shape + values on a synthetic RawWaveform);
  seconds→ms conversion; shank derivation from x; `unit_lookup` round-trip; positive-spike guard.
- Registry converter: `IdxCluster` → long `(session, ks_unit_id, dant_uid)` mapping correctness,
  −1 handling, no duplicate (session, ks_unit_id).
- Pilot-verify end-to-end on a **2–3 session slice** before the full 42-session run.

## 11. Risks & open questions
- **Per-session geometry drift:** if `channel_positions` differs across sessions (bank changes),
  the pooled `channel_locations` is ambiguous — adapter asserts identity and fails loud if not.
- **Centering on atypical units:** positive-going / multi-trough units — guarded by exclusion.
- **hdbscan install on Windows/py3.10:** standalone `hdbscan` (pyDANT imports `import hdbscan`,
  not sklearn's) — uses prebuilt wheels; verify at env setup.
- **Determinism:** seed set, but HDBSCAN/LDA are deterministic given inputs; motion bootstrap is
  seeded.
- **CV halves:** we average them for matching; the held-out ISI AUC (full spike trains) is the
  independent check. (Using CV0-vs-CV1 as an extra waveform-consistency check is a possible
  follow-up, not in scope.)

## 12. Success criteria
- DANT runs to completion on all 42 BG_046 sessions (multi-shank) and emits `dant_registry.csv`.
- A clear, figure-backed verdict comparing DANT vs UnitMatch on: tracked-unit yield, tracked-length
  distribution, co-membership agreement, and held-out ISI-fingerprint AUC.
- All figures saved presentation-ready under `FIGURES/tracking_dant/BG_046/`.
- Honest reporting of failure modes (positive-spike exclusions, any sessions/units dropped).
