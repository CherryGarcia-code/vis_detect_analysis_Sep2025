# D4 — Artefacts: caches, figures, tables

Empirical audit of the artefact trees (`data/`, `FIGURES/`, `table_output/`).
Built across three tasks: **D4a** (session-id integrity + join loss, Task 7 —
below), **D4b** (Task 8, pending), **D4c** (Task 9, pending). Scan-only: no
artefact was modified or repaired.

## D4a — Session-id integrity extended + join-loss quantification (Task 7)

Extends the recon-era corruption check (6 caches under `data/cache/behavior/`)
to every CSV under `data/`, `FIGURES/`, and `table_output/` that carries a
session-id column, classifies each file's key domain via
`_audit_lib.classify_token`, and quantifies rows lost when the keys are joined
(through `canonical_session_id`) against `data/BG_046_staging_manifest.csv`.

- Script: `scripts/audit/d4_session_id_integrity.py`
  (`py scripts/audit/d4_session_id_integrity.py`, exit 0)
- Census CSV (gitignored; committed with `git add -f`):
  `data/cache/audit/csv_key_domains.csv`
  (`file,n_rows,domains,joinable_to_manifest,rows_lost_on_join`)
- Red-test capture: `data/cache/audit/integrity_test.txt`
- Measurement ids: `d4.ids.*` in `docs/audit/measurements.csv`

### Summary

| Measurement | Value | Baseline / expectation | Verdict |
|---|---|---|---|
| `d4.ids.files_scanned` | 327 | — | 296 under `data/`, 31 under `FIGURES/`, 0 under `table_output/` (see blind-spot note) |
| `d4.ids.files_corrupt` | 10 | 6 (recon caches) | scope growth found **4 new corrupt files** — the 3 popgeom_theta deliverables + 1 state_dynamics table |
| `d4.ids.rows_corrupt` | 15,869 | ≥ 15,802 | expectation met: 15,802 recon rows reproduced **exactly** + 67 new `00-padded` rows |
| `d4.ids.integrity_test_red` | exit 1 | expected red | `1 failed, 6 passed`; the failure is the finding — test deliberately not fixed |

No file hit the 200 MB size guard and no file produced a READ-ERROR row; the
census covers every candidate CSV it targeted.

### Corrupt-file inventory (10 files, 15,869 rows)

**The 6 recon caches — 15,802 rows, all `7digit-stripped`, exact match to the
recon baseline** (6496 + 4958 + 4322 + 12 + 9 + 5):

| File | Corrupt rows | Domain |
|---|---|---|
| `data/cache/behavior/fa_hazard_trials_BG_046.csv` | 6,496 | 7digit-stripped |
| `data/cache/behavior/fa_hazard_trials_BG_039.csv` | 4,958 | 7digit-stripped |
| `data/cache/behavior/fa_hazard_trials_BG_031.csv` | 4,322 | 7digit-stripped |
| `data/cache/behavior/early_lick_repl_BG_046.csv` | 12 | 7digit-stripped |
| `data/cache/behavior/early_lick_repl_BG_039.csv` | 9 | 7digit-stripped |
| `data/cache/behavior/early_lick_repl_BG_031.csv` | 5 | 7digit-stripped |

**The 4 files the scope extension found — 67 rows, all `00-padded`:**

| File | Corrupt rows | Domain |
|---|---|---|
| `FIGURES/popgeom_theta/theta_per_session.csv` | 16 | 00-padded |
| `FIGURES/popgeom_theta/theta_count_matched.csv` | 15 | 00-padded |
| `FIGURES/popgeom_theta/theta_support_matched.csv` | 15 | 00-padded |
| `FIGURES/state_dynamics/within_session_dynamics.csv` | 21 | 00-padded |

**popgeom_theta check (plan expectation): confirmed.** All three git-tracked
`FIGURES/popgeom_theta/*.csv` deliverables carry `00050325`-style keys, exactly
as the recon predicted. Per the deliverables' own `subject` column, every
`00-padded` token in them belongs to **BG_031** (e.g. `00050325`, `00070325`,
`00100325`): these are 6-digit DDMMYY session names ad-hoc `zfill(8)`-ed into a
form that is neither DDMMYYYY nor DDMMYY. In `within_session_dynamics.csv` the
21 `00-padded` rows split BG_031 ×18, BG_038 ×1, BG_039 ×2.

**Repairability asymmetry (measured, load-bearing for the register):**
`canonical('1072025')` → `'01072025'` — the `7digit-stripped` form (all 15,802
recon rows) is mechanically repairable at join time. But `canonical('00050325')`
→ `'00050325'` **unchanged** — the `00-padded` form is NOT repaired by the
canonicalizer (it cannot know the inner 6 digits are DDMMYY), so the 67 new
rows are corrupt keys that survive even canonical-mediated joins.

### Join loss vs the BG_046 staging manifest

`rows_lost_on_join` measures loss **after** routing both sides through
`canonical_session_id` and, where a `subject` column exists, scoping to
`subject == "BG_046"` rows only.

- **All 10 corrupt files show `rows_lost_on_join = 0`.** Two distinct reasons:
  the 15,802 `7digit-stripped` rows are repaired by `canonical()` before the
  join (so a canonical-mediated join loses nothing), and the 67 `00-padded`
  rows belong to non-BG_046 subjects and are excluded by the subject mask. The
  0 is therefore **conditional**: any consumer that joins these columns as raw
  strings (without `canonical()`) silently drops all 15,802 stripped rows —
  which is precisely what the red integrity test guards against.
- Census-wide: 281/327 files were joinable to the manifest; 134 files report
  loss > 0 totalling 119,288 rows, **but 126 of them (118,621 rows) are
  other-subject files whose subject sits in the directory path**
  (`data/cache/dant/BG_031/…`, `data/anatomy/BG_039/…`,
  `data/cache/state_tags/BG_031/…`), which the shipped filename-only
  heuristic cannot see. That bulk is a **measurement-scope caveat of the
  census column, not data corruption** — those files were never supposed to
  join a BG_046 manifest.
- **Genuine BG_046-scoped join loss: 667 rows across 8 files.** Dominated by
  the suffixed twin `BG_046_05092025_b` (562 rows:
  `data/cache/tf_glm_bg046/bg046_BG_046_05092025_b.csv` 273,
  `data/cache/tf_responsive/bg046_tf_responsive.csv` 273,
  `data/cache/tf_glm_bg046/targets_bg_046.csv` 8,
  `data/cache/tf_glm_bg046/targets_bg_striatum.csv` 8) — the deduped manifest
  legitimately excludes the `_b` re-sort twin, so this loss is by design but
  means twin-suffixed caches can never join the manifest. The remainder:
  `FIGURES/popgeom_fa_cutoff/subject_usability_raw.csv` (102 rows — a
  multi-subject roster with **no** subject column, so the census misattributes
  its non-BG_046 sessions as loss) and three 1-row edge cases
  (`data/cache/qc_alignment/segmented_verification.csv`,
  `FIGURES/qc/persession_gmm_amplitude.csv`, `FIGURES/qc/qc_recovery_sweep.csv`).

### Red integrity test (`d4.ids.integrity_test_red`)

`py -m pytest tests/test_session_id_csv_integrity.py -q` → **exit 1**
(`1 failed, 6 passed`; full output in `data/cache/audit/integrity_test.txt`).
The failing test names exactly the 6 recon caches with exactly 15,802 stripped
rows — independent confirmation of the census, from a test that predates it.
The test was deliberately left red: it is the live tripwire for this defect
class, and fixing the data is out of scope for a scan-only audit task.

### Blind-spot note (`table_output/`)

`table_output/` contributed 0 census rows because its only CSV
(`table_output/BG_046/Grand_Longitudinal_Table.csv`) carries its session key
under `Session_Date`, which is outside the brief's `ID_COLS` set. A one-off
out-of-band probe (not part of the census CSV) classified all 6,679 of its
`Session_Date` tokens as clean `8digit` — the blind spot exists but is not
hiding corruption in the current tree.

## D4b — (Task 8, pending)

## D4c — (Task 9, pending)
