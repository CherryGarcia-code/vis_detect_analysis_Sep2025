# D4 — Artefacts: caches, figures, tables

Empirical audit of the artefact trees (`data/`, `FIGURES/`, `table_output/`).
Built across three tasks: **D4a** (session-id integrity + join loss, Task 7 —
below), **D4b** (staleness ranking, filter divergence, twin collisions, Task 8
— below), **D4c** (Task 9, pending). Scan-only: no artefact was modified or
repaired.

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

## D4b — Cache staleness ranking, SESSION_FILTER divergence, twin collisions (Task 8)

Three probes over `data/cache/` and `data/pkls/`: (1) an mtime-vs-writer-commit
staleness ranking of every cache topic, (2) how many sessions a script that
reads the staging-manifest CSV directly sees beyond what
`load_staging_manifest(qc_only=True)` serves, (3) which pkl the resolver
actually serves for every twin-colliding date key.

- Script: `scripts/audit/d4_staleness.py` (`py scripts/audit/d4_staleness.py`,
  exit 0)
- Ranking CSV (gitignored; committed with `git add -f`):
  `data/cache/audit/stale_caches.csv`
  (`topic,newest_writer_commit,newest_artefact_mtime,n_files,verdict`)
- Measurement ids: `d4.stale.*`, `d4.filter.*`, `d4.twins.*` in
  `docs/audit/measurements.csv`

### Staleness ranking (`d4.stale.topics` = 7/7)

24 cache topics measured (`audit/` excluded). Verdict is tri-state on purpose:
a topic whose writer cannot be located must never read as "not stale".

| Verdict | Topics |
|---|---|
| **stale (7)** | `behavior` (writer 2026-08-03 vs artefacts 2026-07-27), `session_sorting` (see caveat 2), `talk_substrate`, `tf_glm_bg046` (2026-07-20 vs 2026-07-16), `tf_responsive` (2026-07-20 vs 2026-07-01 — the named gate case), `tracking_consensus`, `um_ref` |
| **no-writer-found (7)** | `chronic_feasibility`, `population_field`, `state_labeling`, `state_tf_learning`, `states`, `tf_labeling`, `tracking_dant` — no file under `scripts/` or `src/` names the literal `data/cache/<topic>` path (paths built from variables/config); excluded from the stale denominator |
| **current (10)** | `dant`, `decision_latents`, `evidence_learning`, `neural_latents`, `optotagging`, `preparatory_fig5`, `qc_alignment`, `state_tags`, `video_labels`, `video_sync` |

**Finding: mtime-based staleness ranking is structurally blind to caches whose
staleness is documented in-place.** The brief's code took the newest mtime over
*all* files in a topic directory. Under that rule `tf_responsive` — the one
topic whose own `README.md` opens with a "⚠️ STALE — predates the lick-channel
fix (2026-07-31)" banner — ranked **current**: the banner's correction commit
(`8054c09`, 2026-08-03) left `README.md` with an on-disk mtime of 2026-08-05,
newer than the newest writer commit (2026-07-20,
`scripts/tf_responsiveness/preparatory_fig5/prep_common.py`), while all four
data CSVs date to 2026-07-01. Writing "this cache is stale" into the cache
directory refreshes the tree's max mtime and permanently hides the staleness
from the heuristic. The shipped script therefore excludes `.md` files from the
artefact scan — a deviation from the plan's code block authorized by the plan's
own Step-2 gate ("tf_responsive MUST rank as stale (its README says so); if it
does not, the heuristic is broken — fix before committing"). Verified on the
final run: the exclusion flips **only** `tf_responsive` (current→stale); every
other topic's newest file is a genuine data artefact and no other verdict
changes.

Heuristic caveats (both directions):
1. "Writer" = any `scripts/`/`src/` file containing the literal
   `data/cache/<topic>` string — a *reader* or an audit script counts too.
   Concretely: `session_sorting`'s newest "writer" (2026-08-12) is
   `scripts/audit/d3_scripts_census.py`, an audit artefact; its true newest
   writer is 2026-08-03, same day as its artefacts, so its **stale** verdict is
   right for the wrong reason — the value-level proof below is what actually
   convicts it. (After this task's commit, `d4_staleness.py` itself becomes a
   `session_sorting` hit; future re-runs inherit that date.)
2. Filesystem mtime is not provenance: a copy/checkout refreshes it, so
   "current" here means "not provably stale by mtime", nothing stronger.
3. **Day-boundary / timezone skew** (plan-mandated comparison, defect noted):
   the two sides of the comparison use different timezone conventions —
   artefact mtimes are converted to **UTC** dates, while
   `git log --format=%cs` yields the committer's **local-timezone** date. Any
   verdict resting on a one-day margin is convention-sensitive. One committed
   verdict sits exactly on that margin: **`um_ref`** (writer 2026-07-02 vs
   artefact 2026-07-01) — its "stale" ranking is within the skew and should be
   treated as **uncertain**, not as an established staleness finding. No other
   stale verdict is a one-day call.
4. The sanctioned `.md` exclusion (see above) means a topic directory
   containing ONLY `.md` files would be skipped by the empty-`files` guard and
   vanish from the ranking silently, rather than appear as unmeasurable. No
   current topic is affected (24 rows before and after the fix), but a future
   banner-only cache directory would disappear without trace.

**Value-level staleness proof, beyond the mtime heuristic
(`d4.stale.chron_impossible` = 14 rows, LOWER BOUND):** the `chron` column of
`data/cache/session_sorting/predicted_session_groups.csv` holds tuples the
current parser cannot produce — 14 rows with month > 12, e.g. `"(325, 27, 0)"`
for session `270325` (a 6-digit DDMMYY id run through the pre-fix
year-month-day placement). This is frozen output of the pre-fix parser sitting
in a live deliverable. Lower bound: same-parse rows whose misplaced day is
≤ 12 are indistinguishable from valid tuples by pattern and are not counted.

### SESSION_FILTER divergence (`d4.filter.divergence` = 18 sessions)

The raw `data/BG_046_staging_manifest.csv` contains **18** `session_name`s that
`load_staging_manifest(qc_only=True)` filters out. Every script that reads the
CSV directly sees those 18 extra sessions. **Upper-bound proxy caveat (as the
brief directs):** 18 is the per-script *ceiling*, not the realized divergence —
some direct readers apply their own d′/trial-count filters, so the sessions
actually leaking into a given analysis may be fewer; quantifying per-script
leakage would require executing each reader. Reader-count note: the measurement
row's notes cite the recon-era figure of 28 direct-reading scripts; a same-day
re-grep of `BG_046_staging_manifest` finds **20** files under `scripts/` +
`src/` (33 repo-wide including docs/tests). The delta is tree drift between
recon and audit, not a contradiction — the recorded notes string is the recon
citation and stays as-is.

### Twin collisions vs the real pkl tree (`d4.twins.colliding_date_keys` = 11)

Date keys (first standalone 6–8-digit token of each pkl stem) with more than
one pkl, per subject: **BG_012: 9, BG_031: 1, BG_039: 1**. Which twin the
resolver (`src/visdetect/suite/loader.py:120-135`) serves is deterministic
(`d4.twins.winners`):

- **BG_012 — all 9 keys → `AMBIGUOUS(None)`.** Each date stores multiple
  protocol variants (`_prot4_lickEndsTrial`, `_airpuff`, …) with no plain
  `BG_012_<date>.pkl`; more than one suffixed candidate → the resolver returns
  `None` **on purpose** (its docstring names BG_012 as exactly this case).
  Consistent with BG_012 being parked: no session on these 9 dates is loadable
  by date key alone — callers must disambiguate via `list_session_recordings`.
- **BG_031/19052025 → `BG_031_19052025.pkl`** and **BG_039/23042025 →
  `BG_039_23042025.pkl`**: a plain file exists, so the plain form wins and the
  suffixed re-sort twin is never served — the "never concatenate twins" rule
  holds at the resolver level for these two.

## D4c — (Task 9, pending)
