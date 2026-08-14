# D8 — Known-defect register

**The deliverable that gates sub-project 3.** Per ADR-009 this is *not* a list of pass/fail
verdicts. It is the **evidence base for attribution**: when a ported component's output differs
from the old repo's, this register is what lets that delta be tagged `known-defect` instead of
`unexplained`. An entry with no stated direction attributes nothing, so every entry states one —
or says plainly that the direction is unknown and points at `quarantine.md`.

- Module classifier: `scripts/audit/d8_module_classifier.py`
  (`py scripts/audit/d8_module_classifier.py`, exit 0)
- Module map (gitignored; committed with `git add -f`):
  `data/cache/audit/module_register_map.csv` — `module, register_entries, uses_canonicaliser`
- Supporting Task-15 census output: `data/cache/audit/constants_retriage.csv`,
  `data/cache/audit/cold_list_seed.csv`
- Measurement ids: `d8.*` plus every id cited per entry, all in `docs/audit/measurements.csv`
- Companions: `quarantine.md` (undetermined, with the settling check), `drop-list.md`,
  `cold-list.md`

## How to read an entry

| Field | Meaning |
|---|---|
| **Direction of effect** | Which way a ported component's number moves relative to the old repo's, and why. This is the attribution key. |
| **Affected modules** | From `module_register_map.csv` unless stated otherwise. Modules are `visdetect.*` dotted paths. |
| **Affected artefacts** | Caches, figures, tables, pkls or docs that carry the defect's imprint. |
| **Evidence** | Measurement ids **and** `file:line`. Where a claim has no measured backing, the entry says so. |
| **Status** | `LIVE` (present in code today) · `IN-REPAIR` (fix in flight on a branch) · `CODE-FIXED / ARTEFACTS-STALE` · `SETTLED-CONVENTION` (resolved to a documented choice) · `QUARANTINED` (direction unknown) · `HISTORICAL` (cannot recur; matters only for old outputs) |
| **Re-ingest disposition** | The project owner has decided the new repo will (a) use **NWB** and (b) **re-ingest sessions from raw** rather than copy pickles or caches. `DISSOLVED` = the defect lives only in derived artefacts nobody carries forward. `SURVIVES` = it is a convention, a piece of code, or a fact about the raw data, and re-ingest does not touch it. `CONDITIONAL` = it turns on a decision the owner has not yet made. |

### ⚠️ The one thing that flips three entries: **the NI half of the re-ingest**

`build_session_from_raw` reads its **behavioural** trials from raw JSON
(`load_behavioral_trials` globs `Session/*trials.json` at `src/visdetect/core/ingest.py:71-73`,
called at `:441`) — that half genuinely is raw. Its **NI events** are not: they come from an
already-extracted `*NIdaq_events.mat` (`ingest.py:444`, glob at `:305`). It never opens
`nidq.bin`. So "re-ingest from raw", as the code stands, is *raw behaviour + the MATLAB NI
product*, and re-running it reproduces every NI-layer defect.

Entries **6 (lick channel)**, **E4 (BG_031 Laser gap)** and **8 (QC1 alignment)** hinge on this —
the three rows dispositioned `CONDITIONAL` in the index below. Re-extracting NI from
`nidq.bin`/`nidq.meta` dissolves the first two outright.

**This is no longer an open decision.** `docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` (2026-08-13/14)
is a full re-extraction of one BG_046 session directly from `nidq.bin` that both **confirms the
diagnosis** and **supplies a validated recipe** — channel map, per-channel threshold derivation,
width rules, and a 12-step pipeline checklist, with the threshold-insensitive channels reproducing
MATLAB at **0.000 ms** (`Baseline_ON` 739/739, `Change_ON` 323/323, `Valve` 251/251,
`Laser` 1003/1003). What remains open is **per-session generalisation**, not method: the spec is
one session of one subject and says so. Carried as `quarantine.md` **Q6**.

> **Evidence-source note.** Every citation below to `docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` is to a
> document produced outside this audit and **read-only** to it — nothing in it was edited. It is
> also **untracked in git**, i.e. a single uncommitted copy on the same disk as the repo, which
> makes it a work-at-risk item in its own right (see `drop-list.md`, *Not dropped, and why*, and
> `d7.untracked.at_risk`, whose 5-of-6 becomes 6-of-7 with this file).

---

## Index

| # | Defect | Status | Re-ingest | Direction, in one line |
|---|---|---|---|---|
| 1 | `load_qc_profile` returns `{}` | LIVE | SURVIVES | Named profiles silently ran function defaults; unit counts change wherever a profile was claimed |
| 2 | `TF_SAMPLE_PERIOD = 0.25` (5× too coarse) | LIVE (inert) | SURVIVES | **No live value-reader** — the hazard is a wrong canonical value beside 77 unlinked `dt` literals |
| 3 | `parse_session_date` mis-sorts stripped / 6-digit ids | LIVE | SURVIVES | Session ORDER changes → every learning-trajectory slope |
| 4 | Corrupted session-id rows in live caches | LIVE | DISSOLVED (artefacts) / SURVIVES (mechanism) | Raw-string joins drop rows; canonical joins recover them → n increases |
| 5 | Stale `tf_responsive` registries | CODE-FIXED / ARTEFACTS-STALE | DISSOLVED | Responsive-unit sets change post-lick-fix; VMS>DMS ordering unsafe |
| 6 | Lick-channel extraction defect | CODE-FIXED / ARTEFACTS-STALE | CONDITIONAL | **Bidirectional, and the root cause is now measured**: the piezo is an ANALOG sensor thresholded as TTL — 2026 `Piezo_1` explains only 14.9 % of rewards |
| 7 | TF-pulse PETH circularity + pre-fix caches | CODE-FIXED / ARTEFACTS-STALE | DISSOLVED | Sign-aligned averages lose the manufactured effect; raw PETH now validates the GLM |
| 8 | Trial/event alignment (QC1) | IN-REPAIR | CONDITIONAL | Trial↔event pairing changes on 23 of 253 pkls; `ni_events`-aligned neural results invalid there |
| 9 | Retracted transient/sustained **state** result | HISTORICAL | SURVIVES (as a claim) | Effect vanishes after FR-normalization — do not re-assert |
| 10 | Refuted "sustained StimSens = expert signature" | HISTORICAL | SURVIVES (as a claim) | Collapses to state occupancy |
| 11 | `ref`-trial change-presented ambiguity | **SETTLED-CONVENTION** (two independent sources) | SURVIVES | Change WAS presented; excluding ref is a scientific choice → including it raises trial counts on hard/fast conditions |
| 12 | `CHANGE_SIZES` membership divergence | **SETTLED-CONVENTION** | SURVIVES | No consumer mixes catch into a go loop; residual risk is naming, not numbers |
| E1 | Irreversible ingest-time QC in the pkls | LIVE | **DISSOLVED** | Unit counts under any new profile can only FALL, never rise, without re-ingest |
| E2 | BG_046 detection-composition drift | LIVE (data fact) | SURVIVES | Any Naive→Expert cell-type contrast inflated in the drift's direction |
| E3 | BG_046-calibrated track-QC thresholds on other subjects | **QUARANTINED** | SURVIVES | Direction unknown — see `quarantine.md` Q1 |
| E4 | BG_031 Laser-event extraction gap (35/43) | LIVE, **mechanism identified** | CONDITIONAL | BG_031 optotag yield UNDERSTATED; no negative claim about D2 yield supportable. The laser line peaks at **0.383 V** and never crosses a TTL threshold |
| E5 | `KNOWN_SUSPECTS = {779, 873, 872}` hardcoded | LIVE | SURVIVES | Tracked cohort slightly cleaner than uncurated, by an UNMEASURED amount |
| A1 | `canonical_session_id` manufactures `00DDMMYY` | LIVE | SURVIVES | Multi-subject joins on canonical ids mismatch; produces unrepairable keys |
| A2 | 5-digit day-stripped DDMMYY tokens (1,670 rows) | LIVE | DISSOLVED (artefacts) / SURVIVES (mechanism) | Uncounted third corruption form; same silent-wrong-date exposure |
| A3 | `parents[N]` fragile-root idiom (4 sites) | LIVE | SURVIVES | Silent wrong path after any file move — the mechanism of entry 1 |
| A4 | Silent-zeros lick read in the Khilkevich loader | LIVE | SURVIVES | Missing `daq_Lick_L.csv` → empty lick regressor, no warning |
| A5 | `SESSION_FILTER` divergence (18 sessions) | LIVE | SURVIVES | Direct manifest readers see up to 18 extra sessions → n rises, d′ floor drops |
| A6 | Twin-pkl resolution: 9 BG_012 keys AMBIGUOUS | LIVE | SURVIVES | Date-key load returns `None`; suffixed twin caches can never join the manifest |
| A7 | Dual import roots (`src.visdetect` vs `visdetect`) | LIVE | SURVIVES | Same class bound twice → `isinstance`/pickle identity silently breaks |
| A8 | Sibling repo re-declares task semantics | LIVE | SURVIVES | `EVENT_VALID_OUTCOMES` differs in keys and casing; no canonicaliser there at all |
| A9 | NWB gzip compression drops silently | **PRE-EMPTIVE** | SURVIVES (as a requirement) | An uncompressed file and a meaningless size number, with no error |
| A10 | Default `py -m pytest` yields zero results | LIVE | SURVIVES | The manual gate reports success by reporting nothing |
| A11 | Delete-guard false positive blocks all recursive deletes | LIVE | SURVIVES | Pushes an operator toward the 2026-06-07 data-loss shape |
| A12 | No built distribution has ever contained `visdetect` | LIVE | DISSOLVED | Every non-editable consumer breaks at build time |
| A13 | TF-GLM wheel regressor edge-counts one encoder line | LIVE | SURVIVES | Manufactures ~6× apparent movement in quiet windows → the "movement-controlled" GLM is not movement-controlled |
| A14 | NI event times are not stimulus times | LIVE (data fact) | SURVIVES | `Baseline_ON` precedes the physical screen change by **+67.3 ms**, comparable to striatal visual latency; every NI time also carries a ~5 ms tick |
| A15 | Optotag block→target mapping is assumed, and inverted on the one measured session | LIVE | SURVIVES | Every D1(SNr)/D2(GPe) pathway label from optotagging may be **swapped** — see `quarantine.md` Q12 |
| D1 | `QUESTION_INDEX.md:49` asserts refuted VMS engagement-gating | LIVE | SURVIVES | Misdirects design, not values |
| D2 | `2026-06-17-post-tf-null…md:4,48` names the wrong region | LIVE | SURVIVES | Its "cheap decisive control" is void as designed |
| D3 | `docs/GOTCHAS.md:10` teaches the integer session-id footgun | LIVE | SURVIVES | An agent reading the copy is instructed to create defect 4 |

**35 entries**: 12 master-design seeds (Section A), 5 ephys (Section B), 15 audit-added
(Section C), 3 documentation-layer (Section D). A13–A15 were added in the Task-15 fix pass from
`docs/raw_data/NIDAQ_AND_EVENT_SPEC.md`.

**Module coverage (acceptance criterion A2):** all **64** library modules are classified
(`d8.modules.classified`); **31** touch no symbol-matched entry (`d8.modules.clean`). No module is
unclassified. See *Module coverage* at the end for the caveat that "clean" ≠ "defect-free".

---

## Section A — the twelve master-design seeds

### 1. `load_qc_profile` returns `{}`; strict/lenient runs identical

- **Direction of effect** — every run that passed `--profile <name>` without an explicit
  `profiles_path` silently used the **function defaults** (500 spikes, 20 % ISI), not the named
  profile. A user asking for `striatal_strict` (1200 spikes, 3 % ISI) got the lenient defaults.
  Unit counts therefore change wherever a named profile was claimed; on **today's** pkls they do
  not change at all, because the ingest gate already dominates (see below), so the delta a porter
  should expect is **zero on current pkls and non-zero on any fuller population**.
- **Affected modules** — `visdetect.core.qc` (the defect), `visdetect.analysis.unit_selection`
  (the amplifier: `used_params.update(...)` on an empty dict at `unit_selection.py:249-250`
  updates nothing, so the collapse is silent).
- **Affected artefacts** — any figure or cache produced by
  `scripts/batch_processing/batch_plot_tf_pulse.py:35` or
  `scripts/analysis/tf_response/plot_tf_pulse_grid.py:56` under a named profile.
  `scripts/batch_processing/batch_plot_tf_grids.py` forwards `--profile` too, but its subprocess
  target `scripts/analysis/plot_tf_pulse_grid.py` **does not exist** (see `drop-list.md`), so that
  path produced nothing at all.
- **Evidence** — `d1.qcprofile.{default,qc_only,striatal_strict,striatal_lenient}` = `{}` × 4
  (executed); `d1.qcprofile.diff.*` = 108 units under all four YAML-intended profiles;
  mechanism at `src/visdetect/core/qc.py:218` (`parents[1]` resolves
  `src/visdetect/config/qc_profiles.yml`, which does not exist — the real file is repo-root
  `config/qc_profiles.yml`) and the silent `if not path.exists(): return {}` at
  `qc.py:220-221`.
  *Reconciled by Task 15*: the four `d1.qcprofile.diff.*` notes previously read "these intended
  counts differ"; all four measured **108**, i.e. identical. Corrected in `measurements.csv` and
  marked `[Task 15 reconciliation]`.
- **Status** — LIVE. **Re-ingest** — SURVIVES: a code defect and a claim-labelling fact about old
  outputs; re-ingest touches neither.

### 2. `TF_SAMPLE_PERIOD = 0.25` — 5× too coarse

- **Direction of effect** — **materially weaker than the seed implies, and the audit says so.**
  The constant has **zero live value-readers** (`d8.tfperiod.value_readers` = 0). Its only
  non-audit sites are the definition, the config re-export, and one **unused import**. No current
  analysis bins TF evidence at 0.25 s through the constant, so no ported component should show a
  5× delta *because of it*. What survives is a wrong canonical value sitting beside 77 **bare
  `dt` literals** (`tests` 50 / `src` 19 / `scripts` 8) that are the de-facto truth and are linked
  to nothing. Direction for the future: any new code that imports the constant in good faith gets
  a 5×-coarse grid — evidence binning 5× *coarser*, kernel/latency estimates smeared.
- **Affected modules** — `visdetect.analysis.constants` (definition),
  `visdetect.analysis.config` (re-export). Module map also flags them under `tf-period-5x`.
- **Affected artefacts** — unknown. `d1.tfperiod.figure_attribution` is **not-measured**: no
  per-figure provenance exists to say which historical outputs used which `dt` (that gap is what
  `d4.trace.untraceable_frac` = 0.42 measures). Carried in `quarantine.md` Q3.
- **Evidence** — `d8.tfperiod.value_readers` = 0 with all three sites listed;
  `d1.tfperiod.consumer_sites` = 83 (`data/cache/audit/tf_dt_sites.csv`: 6 `TF_SAMPLE_PERIOD`
  mentions of which **3 are the audit's own census script**, + 77 bare-dt);
  `d1.tfperiod.measured_s` = not-measured (stim logs are `None` on the probed pkl);
  documentary truth at `src/visdetect/analysis/psychophysical_kernel.py:18`
  ("Everything is dt = 0.05 s … Never 0.25."); value at `constants.py:113`.
- **Status** — LIVE but inert. **Re-ingest** — SURVIVES: a constant and a scattering of literals.

### 3. `parse_session_date(int(x))` mis-sorts 6-digit and day-1–9 tokens

- **Direction of effect** — session ORDER changes, so **every learning-trajectory slope** does.
  The failure is silent, not loud: `strptime('%d%m%Y')` on the int-stripped `1072025` returns
  **2025-07-10** — day 10 instead of day 1, no exception. The float-string `1072025.0` raises.
- **Affected modules** — `visdetect.analysis.config` (`parse_session_date` at `config.py:413`,
  `chronological_sort` at `:480`), and by the module map `analysis.decision_latents`,
  `analysis.neural_latents`, `analysis.state_labeling`, `suite.loader`.
- **Affected artefacts** — `data/cache/session_sorting/predicted_session_groups.csv` holds **14
  rows the current parser cannot produce** (month > 12, e.g. `"(325, 27, 0)"` for session
  `270325`) — frozen pre-fix output sitting in a live deliverable. That 14 is a **lower bound**:
  same-parse rows whose misplaced day is ≤ 12 are indistinguishable from valid tuples.
- **Evidence** — `d3.dateparser.trio`; `d4.stale.chron_impossible` = 14;
  `d3.dateparser.sites` = 19 — **the CSV under-counts**: the true parser-site population is
  **23**. The 4 out-of-regex sites are `scripts/analysis/plot_learning_curve.py:18`,
  `scripts/analysis/run_deep_unitmatch.py:75`,
  `scripts/batch_processing/build_manifest_and_behavior_summary.py:151` (all
  `pd.to_datetime(..., format='%d%m%Y')`, identical silent-wrong-date behaviour) and
  `scripts/pipelines/tracking/run_unitmatch_all.py:73` (`strptime('%d%m%y')`, the 6-digit
  variant). **Cite 23, not 19.**
- **Status** — LIVE. **Re-ingest** — SURVIVES: a parser and a convention.
- **Mitigation already in the tree** — `config.session_date_key` (`config.py:423`) is the
  multi-subject-safe ordering key. Six modules use a canonicaliser
  (`uses_canonicaliser = True` in the module map).

### 4. Corrupted session-id rows in live caches

- **Direction of effect** — a consumer that joins these key columns as **raw strings** silently
  drops all 15,802 `7digit-stripped` rows; routing both sides through `canonical_session_id`
  recovers them, so **n increases** and any per-session statistic recomputes on more data. The
  67 `00-padded` rows behave oppositely: they are **not** repaired by the canonicaliser and stay
  lost (see A1 for why they exist at all).
- **Affected modules** — none in the library; this is an artefact-layer defect. Consumers are
  scripts that read the caches.
- **Affected artefacts** — 10 files / 15,869 rows (`d4.ids.files_corrupt`,
  `d4.ids.rows_corrupt`): the 6 recon caches under `data/cache/behavior/`
  (`fa_hazard_trials_BG_046/039/031.csv`, `early_lick_repl_BG_046/039/031.csv` = 15,802 rows,
  all `7digit-stripped`) **plus** four the scope extension found —
  `FIGURES/popgeom_theta/{theta_per_session,theta_count_matched,theta_support_matched}.csv` and
  `FIGURES/state_dynamics/within_session_dynamics.csv` (67 rows, all `00-padded`, all
  non-BG_046). Add entry A2's **1,670** further rows, which fall outside the census's
  corrupt-token definition.
- **Evidence** — `d4.ids.files_corrupt` = 10; `d4.ids.rows_corrupt` = 15,869;
  `d4.ids.integrity_test_red` = exit 1 (`tests/test_session_id_csv_integrity.py`, deliberately
  left red as the live tripwire); `d8.idcorruption.fivedigit_rows` = 1,670.
  **Do not quote `csv_key_domains.csv`'s loss columns** — for 126 path-scoped other-subject files
  they report ~118,621 phantom "lost" rows because the shipped heuristic reads only the filename.
  Genuine BG_046-scoped loss is **667 rows across 8 files**, dominated by the suffixed twin
  `BG_046_05092025_b` (562 rows), which the deduped manifest legitimately excludes.
- **Status** — LIVE. **Re-ingest** — the **artefacts DISSOLVE** (every one is a derived cache or
  a figure table the new repo rebuilds), but the **mechanism SURVIVES** unless the new repo's
  typed session key (ADR-004) makes an int64 CSV round-trip impossible by construction. Port the
  red integrity test, not the data.

### 5. Stale `tf_responsive` registries behind the VMS > DMS headline

- **Direction of effect** — every session of all three subjects had its lick nuisance regressor
  changed by the 2026-07-31 lick-channel fix, so **borderline `resp_log2` calls will flip**. The
  headline ordering **VMS 5.3 % > DMS 2.8 % / 3.1 %** is unsafe until re-derived. The **number**
  of flips is unmeasured — re-running the TF GLM is compute this audit does not do.
- **Affected modules** — `visdetect.analysis.tf_glm`, `analysis.tf_glm_data`,
  `analysis.evidence_learning_io`, `analysis.state_tf_learning`, `suite.loader` (module map,
  `stale-tf-registries`).
- **Affected artefacts** — `data/cache/tf_responsive/*` (whose own `README.md` carries the
  ⚠️ STALE banner) and, downstream, **six** `docs/science` results docs that rest on the registry
  and mention it in none: `2026-07-01-B9`, `2026-07-01-B10`,
  `2026-07-02-B10-RESULTS-explained`, `2026-07-02-transient-sustained-tf-cells`,
  `2026-07-07-…-spectrum-celltype`, `2026-07-20-preparatory-activity-…`.
- **Evidence** — `d4.tfresp.flips` = **not-measured** (direction only);
  `data/cache/tf_responsive/README.md` (BG_046 old pool inflated lick counts 1.12×–4.99×;
  BG_031 7/43 sessions on a contaminated ~63 Hz line; BG_039 de-pooled 32/32);
  `d4.stale.topics` = 7/7 with `tf_responsive` ranked stale only after `.md` files were excluded
  from the mtime scan — **writing "this cache is stale" into the cache directory hides the
  staleness from any mtime heuristic**, a structural finding in its own right.
- **Status** — CODE-FIXED / ARTEFACTS-STALE. **Re-ingest** — DISSOLVED: the registry is a derived
  cache and will be rebuilt against the fixed regressor. The *claim* does not dissolve; it must
  be re-derived before being repeated.

### 6. Lick-channel extraction defect

- **ROOT CAUSE, now measured: `Piezo_1`/`Piezo_2` are ANALOG sensors that both MATLAB extractions
  thresholded as if they were TTL logic lines.** The two extractions read the *same physical
  lines* — `Lick_L` ≡ `Piezo_1` ≡ analog ch4, `Lick_R` ≡ `Piezo_2` ≡ ch5 — and **differ only in
  threshold**, a choice nobody recorded. Neither is ground truth.
- **Direction of effect — BIDIRECTIONAL, and the seed's single direction is only half of it.**
  Three distinct failures share this name:
  1. **Wrong threshold on an analog line (the dominant one).** Measured on BG_046 17092025, against
     the hardware criterion that every reward must be preceded by the lick that earned it:

     | detector | contacts | rewards preceded | median lead | ILI in the 6–8 Hz band |
     |---|---|---|---|---|
     | derived, 8σ = 0.214 V, 30 ms refractory | **3,580** | **100 %** | **4.63 ms** | 46.3 % |
     | `Lick_L` (2025) | 4,651 | 100 % | 5.00 ms | 45.9 % |
     | `Lick_R` (2025) | 3,638 | 82.2 % | 3.96 ms | 46.5 % |
     | `Piezo_1` (2026) | **494** | **14.9 %** | **35,923 ms** | 9.7 % |

     **Use these figures, not the "10–40×" hand-wave** the memory record carries. The 2026
     `Piezo_1` train is not a sparse sample of the licks — with a 36-second median gap to the
     reward its "nearest lick" earned, it is *not detecting the licks that earned the rewards at
     all*. Direction: lick counts from a 2026-convention session are understated ~7× against a
     correctly derived detector (494 vs 3,580) and are not merely thinned but **mis-timed**.
  2. **Silent zeros.** A reader hard-coding one channel name returns an **empty array** on every
     session written by the other convention. Four scripts had this bug; affected sessions read
     0 licks and cross-session lick-rate trends flatten.
  3. **Pooling (over-detection).** The previous shared helper unioned all four lines,
     **inflating lick counts 1.12×–4.99× on BG_046** (median 1.40×) — `Lick_R` is a second, denser
     detector on the *same* spout and `Piezo_2` a ~11 ms-shifted subset of `Piezo_1`, so the union
     counts one lick several times.

  So a ported lick-rate number can move **either way**, and which way depends on which code path
  the old output came from. Attribute with the path, never with the defect name alone.
- **A fourth hazard the spec adds: raw threshold crossings are not licks.** Every train —
  including the 2025 MATLAB one — is dominated by sub-10 ms intervals, which are piezo *ringing
  within one contact*. De-bounce (30 ms: below the ~125 ms lick period, above the ringing) before
  counting anything. A pipeline that skips de-bouncing over-counts in the opposite direction.
- **Affected modules** — `visdetect.analysis.lick_channels` (the canonical resolver, and the
  record of both failures), `analysis.lick`, `analysis.hmm_validation`, `analysis.tf_glm`,
  `analysis.tf_glm_data` (module map, `lick-channel`).
- **Affected artefacts** — the `tf_responsive` registries (entry 5) and every lick-rate figure
  predating 2026-07-31.
- **Evidence** — `d3.lick.overlap` = **not-measured** (the 33-session re-extraction batch list is
  still not materialized in the repo); mechanism and the pooling magnitudes at
  `src/visdetect/analysis/lick_channels.py:1-45`; `data/cache/tf_responsive/README.md`; the
  threshold table, the analog-sensor root cause and the de-bounce rule from
  `docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` §1, §3 and §7 (one session, BG_046 17092025).
  The derived train independently validates against the task's own online RT at **r = 0.9994**,
  median difference **0.36 ms**, 99.6 % of Hit trials within 50 ms — so the recipe is not
  self-referential.
- **Status** — CODE-FIXED / ARTEFACTS-STALE (all four buggy scripts now route through the
  resolver; see A4 for the one that does not). **Note the resolver fixes *selection*, not
  *threshold*:** it picks the one true line from whichever MATLAB extraction wrote the events, but
  both extractions' thresholds remain someone's unrecorded choice, so a resolver-clean session can
  still be built on the 14.9 %-recall train. **Re-ingest** — **CONDITIONAL**, and this is the
  hinge described at the top: the raw `nidq.meta` channel map is byte-identical across all 50
  BG_046 raw sessions, so re-extracting from `nidq.bin` with a per-session derived threshold
  (spec §7, §11 step 9) dissolves the defect; re-running `build_session_from_raw` against the
  existing `*NIdaq_events.mat` (`ingest.py:444`) reproduces it.

### 7. TF-pulse PETH circularity and pre-fix caches

- **Direction of effect** — the pre-fix pipeline sign-aligned each cell's pulse PETH **by the same
  data it then averaged**, and capped fast pulses at 600 per session (~1.5 % of the ~41k
  available), leaving the raw PETH noise-dominated: its post-window sign agreed with the GLM
  kernel only ~55 % of the time, i.e. near chance. Sign-aligned averages therefore carried a
  **manufactured** effect that **disappears** once alignment uses the external GLM kernel and all
  pulses are used. Fixing it turned the raw PETH into a *model-free validation* of the width axis
  (r = +0.82 against the GLM kernel, per the memory record).
- **Affected modules** — none in the library; the circularity lived in the figure scripts.
- **Affected artefacts** — every `state_conditioned` pulse figure and its `peth_traces.npz`
  caches built before commit `ee34499`; downstream, `docs/science/2026-07-02-transient-sustained-tf-cells.md`
  §3 (lines 108, 116) still states "Every cell is TF-locked (diagonal = latency tiling)" and
  "~50 % suppression-type" (corrected to 36.9 % in the memory layer) with **no in-doc marker** —
  while §7 of the *same file* carries the corpus's best in-doc retraction.
- **Evidence** — fix commit `ee34499` "fix(tf-pulse): remove circular sign-alignment + 600-pulse
  cap — raw PETH now validates the GLM";
  `scripts/tf_responsiveness/state_conditioned/heatmap_transient_sustained.py:50-56`
  (`PULSE_CAP = None`, with the ~98.5 %-discarded post-mortem in the comment);
  `heatmap_continuum.py:133`, `:208`, `:237` (pulse panel now sign-aligned **by the GLM kernel**);
  `d6.science.stale_docs` = 4 (row 3).
- **Status** — CODE-FIXED / ARTEFACTS-STALE. **Re-ingest** — DISSOLVED for the caches; the
  **method rule** survives and is the reason `feedback_circular_analysis_null_controls` is a hard
  rule: never derive sign, sort or normalisation from the data you then average.

### 8. Trial/event alignment defect (QC1)

- **Direction of effect** — trial↔event pairing changes on the affected pkls, so **every
  `ni_events`-aligned neural result on them changes**; behaviour-only work is unaffected. Three
  failure modes with different signs: split recording (a full day's behaviour attached to each
  half — BG_031 `19052025` = 231 + 339 ≈ 569), excess `Baseline_ON`, and total behavioural load
  failure. On the primary subject the two named cases are BG_046 `20082025` (+228 events) and
  `05092025_b` (−281) — **both Expert**, so a Naive-vs-Expert contrast is asymmetrically exposed.
- **A concrete, code-level mechanism for one failure mode, newly evidenced.** The root cause is
  already named in-repo: "the converter loads whatever `*trials.json` files sit in `Session/`
  without checking they belong to that recording" (`src/visdetect/core/run_alignment.py:4-6`, on
  17 sessions). The NI spec supplies the sharpest instance: BG_046 17092025 carries a
  **byte-identical `__trials (2).json` beside `__trials.json`**, and because
  `load_behavioral_trials` globs `Session/*trials.json` (`ingest.py:72`) and concatenates every
  match (`:89-98`), that session loads **1,471 trials instead of 739 — a 2× overcount**. The fix
  is one line: **de-duplicate by content hash, not by filename.** Direction: n_trials rises above
  n(`Baseline_ON`), which is the sign of the BG_046 `20082025` case (+228). *Candidate mechanism,
  not confirmed on that specific session* — the spec measured 17092025, not 20082025.
- **Affected modules** — `visdetect.core.ingest` (`load_behavioral_trials`, `ingest.py:71-98` —
  the duplicate-glob), `visdetect.core.run_alignment` (`build_trial_event_index` at
  `run_alignment.py:210`), `visdetect.core.session` (`trial_event_index` field at `session.py:58`),
  `visdetect.analysis.align`, plus every module the map flags `alignment-QC1` (14 modules,
  because `Change_ON` is in the pattern — that is the blast radius, not a bug list).
- **Affected artefacts** — 23 of 253 pkls (`ni_events`-aligned neural analyses on them are
  invalid; behaviour-only work is unaffected), and any cache or figure derived from those pkls.
- **Evidence** — `docs/science/QUESTION_INDEX.md:66` (23 of 253; the two BG_046 cases);
  `src/visdetect/core/run_alignment.py:4-6` (17 sessions, root cause named);
  `docs/superpowers/specs/2026-08-03-QC1-trial-event-alignment-repair-design.md`;
  `docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` §9 (the duplicate-JSON 2× overcount and the
  de-duplicate-by-hash rule). **In this checkout `analysis/align.py` does not read
  `trial_event_index`** — only `run_alignment.py` builds it and `session.py` stores it. The repair
  (commit `a029ba3`, "align.py honours `trial_event_index`") lives on
  `feature/early-lick-and-session-sorting`, which was still receiving commits on the day of the
  audit.
  *Cross-check worth noting*: QC1's own acceptance constant carries the spec's timing tick —
  `run_alignment.py:27` sets `ACCEPT_RESID_S = 0.05` with the comment "10x above the observed
  0.0051 s aligned value", and 5.1 ms is exactly the ~5 ms state-machine tick the spec measures
  independently (see A14).
- **Status** — IN-REPAIR (on a branch, not on `main` or `design/new-repo-foundation`).
  **Re-ingest** — CONDITIONAL, and now *partly* resolvable: the duplicate-JSON mode **dissolves**
  with a hash-dedup at ingest, and "total behavioural load failure" plausibly dissolves on a clean
  re-ingest. **Split recordings do not** — re-ingesting a split session reproduces the same
  ambiguity unless the loader is told about the halves. Port the solver; do not assume re-ingest
  fixes this.

### 9. Retracted transient/sustained **state** result

- **Direction of effect** — the positive result used a **raw-Hz** metric that scales with firing
  rate (sustained cells fire faster) — an invalid cross-neuron comparison, additionally
  pseudoreplicated. On the FR-normalized metric the class difference is **null** (pooled MWU
  p = 0.37; per subject 0.68/0.47/0.31; per region DMS 0.94 / VMS 0.31; session mixed model
  p = 0.49). A ported component reproducing the *raw-Hz* number is reproducing the artifact.
- **Affected modules** — none; a results-layer claim.
- **Affected artefacts** — `state_x_class` figures and caches.
- **Evidence** — `docs/science/2026-07-02-transient-sustained-tf-cells.md:185-196`
  ("**NULL** (corrected)"). This one is **exemplary**: it is marked in-doc four times
  (lines 8, 20–21, 185, 190), which is why `d6.science.stale_docs` does **not** count it.
- **Status** — HISTORICAL. **Re-ingest** — SURVIVES as a claim-level "do not re-assert".

### 10. Refuted "sustained StimSens = expert signature"

- **Direction of effect** — the pattern is present in Naive sessions, collapses to **state
  occupancy**, and does not replicate cross-mouse. Any ported analysis that recovers it has
  recovered occupancy, not expertise.
- **Affected modules** — none; a results-layer claim.
- **Affected artefacts** — the S1 design spec's Phase-1 framing.
- **Evidence** — `docs/superpowers/specs/2026-07-31-S1-session-grouping-learning-axis-design.md:30`
  (claim) and `:36` (refutation); `docs/science/QUESTION_INDEX.md:67` ("REFUTED en route").
  Never asserted in a `docs/science` results doc, which is why `d6.science.stale_docs` does not
  count it either.
- **Status** — HISTORICAL. **Re-ingest** — SURVIVES as a claim-level fact.

### 11. `ref`-trial change-presented ambiguity — **QUARANTINE RESOLVED**

- **Settled by Task 4.** Across 5 sessions there are 18 `ref` trials and **all 18 carry a valid
  `change_time`** (`d1.ref.with_change_time` = 18 = `d1.ref.total`). The change stimulus **was**
  presented. Median RT from change onset is **+83 ms** (`d1.ref.rt_median_ms`) — after onset, but
  far below any plausible detection latency, i.e. a reflex lick, exactly the behavioural-software
  definition. `Trial.reactiontimes` carries a `Ref` key (`d1.ref.rt_dict_keys`).
- **Consequence for the two constants** — `CHANGE_PRESENTED_OUTCOMES = {"Hit","Miss","Ref"}`
  (`src/visdetect/core/run_alignment.py:24`) is **factually right**: the event exists and can be
  aligned to. `EVENT_VALID_OUTCOMES` excluding `ref` from `Change_ON`
  (`src/visdetect/analysis/constants.py:49`, enforced at `analysis/align.py:158` and `:284`) is a
  **scientific choice — not a data fact.** The new repo must **state it as a choice**, with its
  reason (a lick uninterpretable as detection), not inherit it as a truth.
- **Direction of effect** — **trial counts on hard/fast conditions rise if ref trials are
  included**, and the added trials carry a fast, non-detection lick, so any RT distribution gains
  a short-latency tail and hit-rate on the affected conditions moves upward. Magnitude is small:
  18 trials over 5 sessions, **0 in two of them** (`d1.ref.per_session.*`).
- **SECOND SOURCE — independently confirmed at the hardware level.** The NI spec re-extracted
  `Change_ON` from `nidq.bin` on BG_046 17092025 and found **323** change events after the
  sliver/first-pulse rules — *exactly* the Hit/Miss/Ref trials, and matching the JSON identity
  `count(Hit) + count(Miss) + count(Ref) = 256 + 58 + 9 = 323`. Of the **43 surplus raw pulses**,
  **every one falls on a Hit/Miss/Ref trial and none on `fa` or `abort`**. The hardware never
  marks a change on a trial where no change was presented. This corroborates the Task-4 resolution
  from a completely independent direction — pkl trial fields vs raw voltage — and closes the
  question that `Change_ON` on `fa`/`abort` is meaningless: verified at the metal.
- **Affected modules** — `visdetect.analysis.constants`, `analysis.align`, `analysis.utils`,
  `core.run_alignment`, `analysis.tf_glm_data`, `analysis.config` (module map, `ref-ambiguity`).
- **Affected artefacts** — every `Change_ON`-aligned PETH, tensor and decoder cache built under
  the current `EVENT_VALID_OUTCOMES` (i.e. all of them); no artefact is *wrong*, but each embeds
  the unstated choice.
- **Evidence** — `d1.ref.total` = 18, `d1.ref.with_change_time` = 18, `d1.ref.rt_median_ms` = 83,
  `d1.ref.rt_dict_keys` = `FA;Miss;RT;Ref;abort;gray`, `d1.ref.per_session.*` (5 sessions);
  constants at `src/visdetect/analysis/constants.py:49` and
  `src/visdetect/core/run_alignment.py:24`, enforcement at `analysis/align.py:158`, `:284`;
  second source `docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` §5 and §9 (one session, BG_046 17092025).
- **A porting trap in the same place** — `run_alignment.py:22-24` warns verbatim: "CASE-SENSITIVE.
  Real pkl labels are capitalised: Hit/Miss/FA/abort/Ref. Do NOT refactor onto
  `EVENT_VALID_OUTCOMES` — that is lowercase and omits Ref." Two constants encode one rule in two
  casings and two memberships; the new repo needs one typed outcome enum, not a second spelling.
- **Status** — SETTLED-CONVENTION. **Re-ingest** — SURVIVES: a convention to be re-declared.

### 12. `CHANGE_SIZES` membership divergence — **QUARANTINE RESOLVED**

- **Settled by a per-consumer check** (`d8.changesizes.catch_in_go_loops` = **0**). Four
  definition sites, two memberships, and **no consumer mixes catch into a go-trial loop**:

  | Site | Membership | Consumers, and why the membership is right there |
  |---|---|---|
  | `src/visdetect/analysis/config.py:264` | go-only (5) | `sorted(ALL_GO_CHANGE_SIZES)`. Consumed by `analysis/decision_latents.py:354,:624,:693,:701` and `scripts/analysis/decision_latents/behavioral_qc_profile.py:77` — all psychometric / RT **go-trial** loops. Correct. |
  | `src/visdetect/analysis/tf_glm.py:210` | includes catch `1.0` | Sole consumer `tf_glm.py:351` builds **one FIR regressor per change size**, including `change_1.0` for catch trials. The catch "change" event physically occurs and must be modelled. Correct by design; not a go loop. |
  | `src/visdetect/analysis/tf_glm_data.py:168` | includes catch `1.0` | `_snap_change_size` at `:232` snaps a measured `Stim2TF/Stim1TF` ratio to the nearest legal value; `1.0` is the legal target for catch. Correct by design. |
  | `scripts/analysis/decision_latents/run_decision_latents_by_state.py:64` | go-only (5) | Local re-declaration; agrees with `config.CHANGE_SIZES`. Redundant, not wrong. |

- **Direction of effect** — **none measured.** The residual risk is a **naming hazard**: one
  symbol name carries two legitimate memberships in the same package, so a future author who
  imports "the" `CHANGE_SIZES` into a go-trial loop from the wrong module silently gains a catch
  condition (hit-rate on the "1.0 condition" collapses to the false-alarm rate and any
  psychometric fit gains a spurious floor point). The new repo should give the two sets **two
  names**.
- **Two further facts a porter needs** — (a) `CHANGE_SIZES` lives in **`config.py:264`, not
  `constants.py`**, contradicting CLAUDE.md's prose placement (`d1` census; the *value* CLAUDE.md
  documents is correct — Task 11 verified it by import, `d6.literals.mismatch`'s
  `CHANGE_SIZES` row is a mechanical false positive). (b) The comment at
  `src/visdetect/analysis/state_labeling.py:162` ("Keys must stay in sync with
  `constants.CHANGE_SIZES`") points at the wrong module; and
  `src/visdetect/analysis/tracking_qc.py:74-79` **deliberately** differs
  (`BIG_POOL`/`SMALL_POOL` exclude 1.5× as an ambiguous mid) — a documented divergence, not a
  defect.
- **Affected modules** — `analysis.config`, `analysis.constants`, `analysis.decision_latents`,
  `analysis.state_labeling`, `analysis.tagging`, `analysis.tf_glm`, `analysis.tf_glm_data`,
  `analysis.tracking_qc`.
- **Affected artefacts** — none identified. The per-consumer check found no contaminated loop, so
  no existing psychometric, RT or GLM artefact is implicated; this entry is a forward-looking
  naming hazard.
- **Evidence** — `d8.changesizes.catch_in_go_loops` = 0, whose evidence field lists all four sites
  with their memberships; definition sites `config.py:264`, `tf_glm.py:210`, `tf_glm_data.py:168`,
  `run_decision_latents_by_state.py:64`; consumer sites `tf_glm.py:351`, `tf_glm_data.py:232`,
  `decision_latents.py:354,:624,:693,:701`, `behavioral_qc_profile.py:77`;
  `d6.literals.mismatch` (the `CHANGE_SIZES` row is a mechanical false positive, Task 11).
- **Disclosure on the measurement** — `d8.changesizes.catch_in_go_loops` = 0 is a **hardcoded
  literal** beside a hand-written `CS_SITES` table in `scripts/audit/d8_module_classifier.py`, not
  a computed count. The four sites and their consumers were read by hand and are enumerated in
  both the script and the table above so the judgment is checkable; the same disclosure applies as
  for `d8.tfperiod.value_readers`.
- **Status** — SETTLED-CONVENTION. **Re-ingest** — SURVIVES: naming.

---

## Section B — the five ephys entries

These were added to the register by the master-design panel review. Until now their only home was
the panel-raw JSON — a dead-end pointer for a fresh implementer — so they are restated here in
full, with evidence.

### E1. Irreversible ingest-time QC (pkls store only `good_and_stable` units)

- **Direction of effect** — **unit counts under any new QC profile can only FALL, never rise,
  without re-ingest.** The excluded units are not filtered inside the pickle; they are *absent*
  from it. On session `01072025` the pkl holds **108 units of 260 Kilosort-good**.
- **Affected modules** — `visdetect.core.ingest`, `visdetect.core.qc`
  (`find_good_stable_units`, `qc.py:269`, ≥ 0.5 Hz gate), `visdetect.analysis.utils`
  (`get_good_cluster_ids`, its own hardcoded 1.0 Hz default).
- **Affected artefacts** — **every pkl**, and therefore every unit count ever published from one.
- **Evidence** — `d1.frfloor.good_and_stable` = 108, `d1.frfloor.getgood_01hz` = 108,
  `d1.frfloor.getgood_1hz` = 92, `d1.frfloor.spread` = 108/92/108 (reconciled by Task 15: **two**
  distinct populations, not three — the yml's 0.1 Hz floor cannot bind below the 0.5 Hz ingest
  gate, so paths 1 and 3 coincide; the only binding floor is the 1.0 Hz code default, which drops
  16/108 = **14.8 %**); `d9.keep_all_good` = *code-side YES; data-side not-measured*.
- **Second-order finding** — which of the two populations an analysis used depends on whether it
  called `get_good_cluster_ids` or read `good_and_stable_ids` directly. That is a silent **15 %**
  population difference between scripts that both believe they use "the" QC'd units.
- **Status** — LIVE. **Re-ingest** — **DISSOLVED, and this is the register's flagship dissolved
  entry.** `build_session_from_raw(keep_all_good=True)` already exists
  (`src/visdetect/core/ingest.py:415`; the `True` branch keeps `set(good_cluster_ids)` at
  `:492-495`) and the whole ingest chain reads **only** Kilosort/Phy `.npy`/`.tsv`
  (`ingest.py:243-267`, `:191-192`, `core/kilosort.py:42-49`) — a grep of `src/visdetect/core/`
  finds no `.ap.bin`, no `memmap`, no `np.fromfile`. What remains open is **data-store
  completeness on `X:`**, not code: see `quarantine.md` Q5.
- **One spec checklist item is ALREADY satisfied — do not re-litigate it.** The NI spec's step 11
  ("prefer `spike_times_sec_adj.npy`; the TPrime correction is not negligible — median shift
  −9.0 ms, range −14.9 to +0.03 ms") is already the code's behaviour: `ingest.py:247` tries
  `spike_times_sec_adj.npy` first and only falls back to `spike_times_sec.npy` (`:250`) or raw
  samples (`:253`), logging which it used. Recorded here so a porter does not spend the check.

### E2. BG_046 detection-composition drift (broad/SPN 89 → 15 %, amplitude halving Jun → Jul)

- **Direction of effect** — early sessions are **SPN / high-amplitude biased**, late sessions
  **FSI / low-amplitude biased**, so **any Naive→Expert cell-type contrast is inflated in the
  drift's direction**. Because the behavioural gate additionally excluded 5 of 6 SPN-rich June
  sessions, learning stage and recording epoch are **collinear by construction** — an
  epoch-confounded contrast cannot be disentangled after the fact.
- **Affected modules** — `visdetect.analysis.waveform_celltype`, `analysis.unit_selection`,
  `core.qc`, and any consumer of a cell-type label.
- **Affected artefacts** — every cross-stage cell-type comparison on BG_046.
- **Evidence** — `docs/superpowers/specs/2026-08-05-new-repo-master-design.md:692-694`
  (ADR-018's measured motivation); memory record `qc_celltype_yield_jun2026`.
  **No `d*` measurement id backs the 89 → 15 % figure** — it predates this audit and was not
  re-measured here. Cite it as inherited, not as an audit measurement.
- **Status** — LIVE (a property of the recordings). **Re-ingest** — SURVIVES: re-ingesting the
  same Kilosort output reproduces the same detection composition. The mitigation is ADR-018's
  chronic-stability control row (days-from-implant covariate / composition matching /
  tracked-subset replication / within-window comparison), not a data fix.

### E3. BG_046-calibrated track-QC thresholds applied to other subjects

- **Direction of effect** — **unknown. QUARANTINED** (see `quarantine.md` Q1). The thresholds
  were calibrated to the BG_046 cohort's distribution; applied to BG_031/038/039 they could be
  either too permissive (admitting matching errors as tracked units) or too strict (discarding
  real tracked units), and nothing in the repo measures which.
- **Affected modules** — `visdetect.analysis.tracking_qc`, `analysis.track_verdict`,
  `analysis.track_curation`, `analysis.tracking_registry`, `anatomy.peak_channel`.
- **Affected artefacts** — `data/cache/tracking_consensus/*`, `data/cache/dant/*`,
  `FIGURES/tracking_qc/*`, and any multi-subject tracked cohort.
- **Evidence** — `src/visdetect/analysis/tracking_qc.py:52`, `:62`, `:69` — three thresholds
  whose comments say verbatim "Calibrated to BG_046 cohort distribution (May 2026)" and
  "top ~25 % of BG_046 cohort".
- **Status** — QUARANTINED. **Re-ingest** — SURVIVES: code constants and a calibration choice.

### E4. BG_031 Laser-event extraction gap (35 of 43 sessions missing the event)

- **Direction of effect** — **BG_031 optotag yield is UNDERSTATED, and no negative claim about D2
  yield is supportable from it.** This is the entry that matters most for misattribution: a
  data-completeness defect that **looks like a biological result** ("we found no D2 units").
- **MECHANISM — now identified, on BG_046, and very likely the same here.** The NI spec shows the
  2025 MATLAB extraction named the 8 analog channels **positionally**, so what it called
  `Valve_R` was really **ch7, the Laser** — and it read empty for one reason: **the laser line
  peaks at 0.383 V and never crosses a conventional 2.5 V TTL threshold.** The 2026 re-extraction,
  run specifically to add the opto channel, found 1,003 pulses on the same physical line. So the
  "missing" Laser events are a **threshold artefact of the extraction, not absent data** — the
  pulses are in `nidq.bin`.
  **Hedge, stated plainly:** this was measured on **BG_046 17092025**, not on BG_031. The
  inference to BG_031's 35 sessions is by mechanism and by the shared 8-channel map, not by
  measurement. It is the single most likely explanation and it is cheap to test (quarantine Q6),
  but it is not yet demonstrated on BG_031.
- **Affected modules** — `visdetect.analysis.optotagging` (`LASER_KEY = "Laser"` at
  `optotagging.py:38`; `optotagging.py:761` raises when the key is absent, so the failure is at
  least loud at that call site).
- **Affected artefacts** — `data/cache/optotagging/*`, any BG_031 optotag yield figure.
- **Evidence** — **independently confirmed by this task**: `d8.bg031.laser_missing` = **35/43**.
  Method: byte-presence scan for the pickled dict key `b"Laser"` in each
  `data/pkls/BG_031/*.pkl` (read-only, no unpickling, no `X:` access) — 8 of 43 pkls contain it,
  35 do not, matching the figure inherited from
  `docs/superpowers/specs/2026-08-05-new-repo-master-design.md:704`, which until now carried no
  measurement id. Caveats: absence of the token is reliable, presence is an upper bound; the
  denominator includes the re-sort twin `BG_031_19052025_b.pkl`.
- **Status** — LIVE. **Re-ingest** — CONDITIONAL, on the same hinge as entry 6: the `Laser`
  channel is an NI line, and the 2026-03-06 MATLAB re-extraction was run *specifically to add it*
  (`lick_channels.py:11`). Re-extracting NI from raw SpikeGLX plausibly recovers the 35 sessions;
  re-running ingest against the existing `*NIdaq_events.mat` does not.

### E5. `KNOWN_SUSPECTS = {779, 873, 872}` hardcoded in `tracking_qc.py`

- **Direction of effect** — the **tracked cohort is slightly cleaner than an uncurated one, by an
  UNMEASURED amount.** Three unit ids are hand-flagged in library code, so any tracked-cohort
  quality statistic is optimistic by an unknown margin, and the flag does not travel with the
  data — a re-derivation without it produces a slightly *worse*-looking cohort that is actually
  the honest one.
- **Affected modules** — `visdetect.analysis.tracking_qc` only.
- **Affected artefacts** — tracked-cohort QC tables and figures carrying `suspect_known`.
- **Evidence** — `src/visdetect/analysis/tracking_qc.py:919`
  (`KNOWN_SUSPECTS: Set[int] = {779, 873, 872}`) and `:950`
  (`"suspect_known": int(uid) in KNOWN_SUSPECTS`).
- **Status** — LIVE. **Re-ingest** — SURVIVES: hand verdicts belong in ADR-018's decision log,
  not in a module-level set.

---

## Section C — entries the audit added

Each of these can move a number or a scope decision and none was in the seed list.

### A1. `canonical_session_id` manufactures the unrepairable `00DDMMYY` form

- **Direction of effect** — the canonicaliser is **DDMMYYYY-only**. It blind-`zfill(8)`s any
  numeric id (`src/visdetect/analysis/config.py:329`, `str(int(session)).zfill(8)`), so a
  **6-digit DDMMYY** session — the naming used by BG_031/038/039 — becomes `00DDMMYY`, a form that
  is neither DDMMYYYY nor DDMMYY and that the canonicaliser then leaves **unchanged** on a second
  pass. Multi-subject joins keyed on "the canonical id" therefore mismatch silently, and the key
  cannot be repaired downstream because nothing can know the inner six digits are DDMMYY.
- **This inverts Task 7's reading.** The 67 `00-padded` rows in the popgeom_theta and
  state_dynamics deliverables are not rows the canonicaliser *failed to repair* — they are rows it
  **produced**.
- **Affected modules** — `visdetect.analysis.config`; every one of the six modules the map marks
  `uses_canonicaliser = True` inherits it (`analysis.config`,
  `analysis.decision_latents_generative`, `analysis.evidence_learning_io`,
  `analysis.neural_latents`, `analysis.state_tf_learning`, `suite.loader`).
- **Affected artefacts** — the 4 files / 67 rows of `d4.ids.files_corrupt`'s `00-padded` class.
- **Evidence** — `d8.canonical.ddmmyy_behaviour`
  (`50325 -> 00050325 | 050325 -> 00050325 | 100325 -> 00100325 | 1072025 -> 01072025`);
  `config.py:295-335`, with the `str(int(session)).zfill(8)` at `:329`.
- **THE RIGHT MITIGATION ALREADY EXISTS, and it is not `session_date_key`.** Two purpose-built
  helpers sit a few lines below the canonicaliser and were built for exactly this failure:
  - **`config.restore_session_token`** (`config.py:344-374`) — **width-preserving**. It restores a
    stripped leading-zero day within whichever family the token belongs to (`1072025 → '01072025'`
    for 8-digit DDMMYYYY; `50325 → '050325'` for 6-digit DDMMYY), because 5- and 7-digit widths are
    unambiguous. Its docstring says the point verbatim: use it "when normalizing a column that may
    span subjects (**it will not mangle a 6-digit DDMMYY into `00050325`**)".
  - **`config.canonicalize_session_column`** (`config.py:377`) — the `to_csv` guard built on it,
    which casts a session column to width-preserving strings so pandas cannot round-trip it back
    to an int64. Call it immediately before every `to_csv` on a table carrying a session id.

  `session_date_key` (`config.py:423`) is **not** the mitigation for this entry: it returns a
  `Tuple[int, int, int]`, i.e. an **ordering key**, not an id normalizer. It is the right answer
  for entry 3 (multi-subject chronological sort) and the wrong answer here. An earlier draft of
  this entry pointed at it; corrected.
- **So the defect is a usage rule, not a missing capability**: the tree has both a DDMMYYYY-only
  canonicaliser and a width-preserving normalizer, with no gate deciding which a caller should
  reach for. The 6 modules marked `uses_canonicaliser = True` all reach for the DDMMYYYY-only one.
- **Status** — LIVE. **Re-ingest** — SURVIVES: the new repo's typed session key must encode the
  **subject's naming convention**, not assume DDMMYYYY — and should make the wrong call
  unrepresentable rather than merely documented.

### A2. 5-digit day-stripped DDMMYY tokens — a third corruption form, counted nowhere

- **Direction of effect** — same class as entry 4: silent join misses and silent wrong dates. It
  is invisible in every prior count because `_audit_lib.classify_token` only recognises 6/7/8-digit
  forms, so a 6-digit DDMMYY id whose leading-zero **day** was stripped by an int cast lands as
  5 digits and falls into the untriaged `other` bucket.
- **Affected modules** — none in the library produces these directly; the writers are the
  behaviour scripts under `scripts/analysis/behavior/` that emit those two caches. The **fix**
  belongs to `visdetect.analysis.config` (`restore_session_token` handles the 5→6 case explicitly;
  see A1) and to whatever calls `to_csv` without `canonicalize_session_column`.
- **Affected artefacts** — **1,670 rows**: `data/cache/behavior/fa_hazard_trials_BG_031.csv`
  (1,668, tokens `50325` = 05 Mar 25 and `70325` = 07 Mar 25) and
  `data/cache/behavior/early_lick_repl_BG_031.csv` (2). Both files are also entry-4 offenders,
  so their true corrupt-row totals are higher than the census reports.
- **Evidence** — `d8.idcorruption.fivedigit_rows` = 1,670, triaging the 1,670 `other`-domain
  tokens Task 7 carried forward. Combined with `d4.ids.rows_corrupt` = 15,869, the honest
  corrupt-row total across the scanned tree is **17,539**.
- **Status** — LIVE. **Re-ingest** — DISSOLVED for the artefacts, SURVIVES for the mechanism
  (identical to entry 4).

### A3. `Path(...).parents[N]` fragile-root idiom — the mechanism behind entry 1

- **Direction of effect** — a silently wrong path. Today three of the four sites are correct and
  one is not; **any file move makes a correct one wrong with no error**, and the failure surfaces
  as a missing config that some caller defaults around.
- **Affected modules** — `visdetect.core.qc` (`qc.py:218`, `parents[1]` — the live bug),
  `visdetect.analysis.config` (`config.py:78`), `visdetect.analysis.state_tf_learning` (`:24`),
  `visdetect.analysis.tf_labeling` (`:25`) — the last three `parents[3]`, correct today.
- **Affected artefacts** — indirectly, everything entry 1 lists: the `parents[1]` site is what
  makes `load_qc_profile` miss its YAML. No artefact carries this defect on its own.
- **Evidence** — `d2.parents.sites` = 4, `data/cache/audit/parents_sites.csv`.
- **Status** — LIVE. **Re-ingest** — SURVIVES: the new repo should resolve package data through
  `importlib.resources`, not by counting directories.

### A4. Silent-zeros lick read still live in the Khilkevich reference loader

- **Direction of effect** — `load_khilkevich_session` reads its lick train from a **hard-coded
  filename** and, if the file is absent, returns an **empty array** with no warning
  (`_daq` at `src/visdetect/analysis/tf_glm_data.py:114-120`,
  `licks = _daq("daq_Lick_L.csv")` at `:121`). That is precisely the failure mode
  `lick_channels.py` was written to make loud. A GLM fitted through this path on a session
  missing that CSV has an all-zero lick nuisance regressor, so **motor variance leaks into the
  sensory kernels** and TF-responsiveness is over-called.
- **Scope, stated precisely** — this is the loader for the **external Khilkevich `npx_converted`
  reference dataset**, not for BG_* pkls. It is not evidence that the BG_* lick fix is
  incomplete: the four scripts that had the bug now route through the resolver.
- **Affected modules** — `visdetect.analysis.tf_glm_data`.
- **Affected artefacts** — any TF-GLM fit run against the Khilkevich reference dataset through
  this loader. **None of the BG_* registries under `data/cache/tf_glm_bg046/` or
  `data/cache/tf_responsive/` come through it** — those use `session_trial_regressors` with the
  resolver — so the BG_* artefact set is not implicated by this entry.
- **Evidence** — `tf_glm_data.py:107-121`; contrast `analysis/lick_channels.py:16-19`
  ("We raise `NoLickChannelError` instead, so a mismatch fails LOUD").
- **Status** — LIVE. **Re-ingest** — SURVIVES: reference-dataset ingest is separate code.

### A5. `SESSION_FILTER` divergence — 18 sessions

- **Direction of effect** — a script that reads `data/BG_046_staging_manifest.csv` **directly**
  sees up to **18 more sessions** than `load_staging_manifest(qc_only=True)` serves. n rises,
  and the extra sessions are the ones the filter rejects — `min_dprime = 0.8`,
  `min_trials = 150` (`src/visdetect/analysis/config.py:169-176`) — so any aggregate performance
  statistic moves **down** and the merged "Learning" pool grows.
- **Affected modules** — `visdetect.analysis.behavior` (`load_staging_manifest`) and
  `visdetect.analysis.config` (`SESSION_FILTER`, `config.py:169-176`) define the filtered path;
  the defect is in the **~20 `scripts/` + `src/` files that bypass them** and read
  `data/BG_046_staging_manifest.csv` directly. No library module is itself wrong.
- **Affected artefacts** — any cache or figure whose session set came from a direct CSV read
  rather than `load_staging_manifest(qc_only=True)`. Which artefacts those are is **not
  enumerated** — the D4 traceability census (`d4.trace.untraceable_frac` = 0.42) is exactly the
  reason it cannot be.
- **Evidence** — `d4.filter.divergence` = 18. **Upper-bound proxy**: 18 is the per-script
  ceiling; some direct readers apply their own filters, and quantifying realized leakage needs
  each reader executed. The measurement row's notes cite the recon-era "28 direct-reading
  scripts"; a same-day re-grep finds **20** files under `scripts/` + `src/`.
- **Status** — LIVE. **Re-ingest** — SURVIVES: a config/convention split between "the manifest"
  and "the filtered manifest". ADR-018's *strata, not verdicts* rule replaces it.

### A6. Twin-pkl resolution — 9 BG_012 date keys resolve to `None`

- **Direction of effect** — for the 9 colliding BG_012 date keys, loading by date key returns
  **`None`, deliberately** (multiple suffixed protocol variants, no plain file), so those sessions
  are simply unloadable by key; callers must go through `list_session_recordings`. For BG_031
  `19052025` and BG_039 `23042025` a plain file exists and **wins**, so the re-sort twin is never
  served — the "never concatenate twins" rule holds at the resolver level for those two. Separately,
  twin-suffixed **caches** can never join the deduped manifest: `BG_046_05092025_b` accounts for
  562 of the 667 genuine BG_046 join-loss rows.
- **Affected modules** — `visdetect.suite.loader` (the resolver, `loader.py:120-135`); every
  caller that loads by bare date key inherits the `None`.
- **Affected artefacts** — the twin-suffixed caches that can never join the deduped manifest:
  `data/cache/tf_glm_bg046/bg046_BG_046_05092025_b.csv` (273 rows),
  `data/cache/tf_responsive/bg046_tf_responsive.csv` (273),
  `data/cache/tf_glm_bg046/targets_bg_046.csv` (8) and `targets_bg_striatum.csv` (8) — 562 of the
  667 genuine BG_046 join-loss rows.
- **Evidence** — `d4.twins.colliding_date_keys` = 11 (BG_012 9, BG_031 1, BG_039 1);
  `d4.twins.winners`; resolver at `src/visdetect/suite/loader.py:120-135`.
- **Status** — LIVE (by design). **Re-ingest** — SURVIVES: twins are re-sorts of identical
  behaviour and must be deduped, never concatenated.

### A7. Dual import roots — the same class bound twice

- **Direction of effect** — a file importing both `src.visdetect.*` and `visdetect.*` binds the
  same source to **two distinct module objects**, so classes compare unequal across the two roots
  and `isinstance` / pickle identity silently break. Symptom: a `Session` loaded one way fails an
  `isinstance` check written the other way.
- **Affected modules** — no library module is itself wrong; the defect is that `visdetect.*` and
  `src.visdetect.*` resolve to two copies of every module, so **`visdetect.core.session`
  (`Session`, `Trial`, `Cluster`) is the class set that ends up double-bound**. The offending
  files are all scripts: 7 import the `src.` root, 6 of them mixed, **all under
  `scripts/video/`**: `batch_sync_sessions.py`, `characterize_camera_signal.py`,
  `corneal_spatial_diagnostic.py`, `poc_multianchor_sync.py`, `select_roi.py`,
  `sync_validation_figure.py` (mixed) and `compare_mask_sync.py` (`src.` only).
- **Affected artefacts** — the camera / video-sync outputs those seven scripts write
  (`data/cache/video_sync/`, `data/cache/video_labels/`, `FIGURES/video*`). No numerical
  corruption is demonstrated — the hazard is an identity failure at runtime, which shows up as an
  exception or a silently skipped branch, not as a wrong number.
- **Evidence** — `d2.dualroot.src_importers` = 7, `d2.dualroot.mixed` = 6.
- **Status** — LIVE. **Re-ingest** — SURVIVES if the camera line is ported; the camera subsystem
  is cold-listed, so this is a *port-time* gate, not a today problem.

### A8. The sibling repo re-declares this repo's task semantics

- **Direction of effect** — `vis_detect_analysis_Apr2023` (photometry, `visdetect_photom`) holds
  **12 `.py` files** that independently re-declare this project's constructs. Its
  `core/constants.py:26` restates `CHANGE_SIZES = [1.25, 1.35, 1.5, 2.0, 4.0]` **verbatim
  equal**, but its `EVENT_VALID_OUTCOMES` encodes the same alignment rule under **different keys
  and casing** (`change`/`fa_lick`/`hit_lick` → `['Hit','Miss']` vs this repo's
  `Change_ON`/`FA`/`Hit` → `{'hit','miss'}`), and its `core/staging.py:12` is a **second
  `load_staging_manifest` with an incompatible contract** (takes a path, returns `None` when
  absent; no `qc_only=`, no `SESSION_FILTER`). Most sharply: it has **zero** uses of
  `canonical_session_id` or `zfill(8)` and matches sessions with a raw `==`
  (`staging.py:34`), so the leading-zero-day footgun is **entirely unfixed there**.
- **Direction for a porter** — a photometry-vs-ephys comparison built across the two repos can
  disagree on which trials are valid and on which sessions match, with no grep in either checkout
  able to find the other definition.
- **Affected modules** — on this side of the boundary, the ones whose definitions are duplicated:
  `visdetect.analysis.constants` (`CHANGE_SIZES`, `FA_RT_SPLIT`, `EVENT_VALID_OUTCOMES`),
  `visdetect.analysis.config` (`canonical_session_id`, `restore_session_token`),
  `visdetect.analysis.behavior` (`load_staging_manifest`). The 12 sibling files are outside this
  repo and outside the rebuild's scope.
- **Affected artefacts** — any cross-repo photometry-vs-ephys comparison. **None exists in this
  repo today**, so no current artefact is implicated; the exposure is prospective.
- **Evidence** — `d7.sibling.duplication` = 12 files (CHANGE_SIZES 7, staging_manifest 6,
  canonical_session_id 0, zfill(8) 0). Read-only survey; nothing in the sibling was modified.
- **Status** — LIVE. **Re-ingest** — SURVIVES: an ADR-011/ADR-015 boundary decision (publish the
  task-semantics layer, or accept the duplication and record the divergence in both repos). The
  status quo is not viable — the copies already disagree.

### A9. NWB gzip compression drops silently — a **pre-emptive** entry for the new writer

- **Direction of effect** — in `pynwb`/`hdmf`, attaching an `H5DataIO` compression wrapper
  **per row** of `units/spike_times` does not work: `hdmf` consumes the wrapper element-wise, the
  compression **silently drops**, and the write succeeds. The result is an uncompressed file and
  a size figure that means nothing. There is no error and no warning; the only symptom is a
  quietly wrong number.
- **Requirement on the new repo's NWB writer** — set the DataIO on the **concatenated column**
  (`nwb.units.spike_times.set_data_io(H5DataIO, {"compression": "gzip"})`) and **assert the codec
  post-write with `h5py`** (`assert h5["units/spike_times"].compression == "gzip"`). Generalise
  it: **any storage option set through a wrapper object must be read back off the written file and
  checked**, because the failure mode is silence.
- **Affected modules** — **none in this repo** (nothing here writes NWB). The entry addresses the
  new repo's `SessionStore` writer (ADR-015).
- **Affected artefacts** — **none yet**; prospectively, every `.nwb` the new pipeline writes, and
  any size or storage figure quoted from one.
- **Provenance, stated honestly** — inherited from the plan's own round-2 review and carried
  through the cancelled D9 spike. **Neither Task 14 nor Task 15 executed or verified it.**
- **Evidence** — `docs/audit/09-storage-spike.md` "Carry-forward 1"; `d9.compression` =
  not-measured.
- **Status** — PRE-EMPTIVE (nothing in this repo writes NWB). **Re-ingest** — SURVIVES as a
  requirement on the new writer.

### A10. The default test invocation yields zero results

- **Direction of effect** — `py -m pytest` **cannot complete collection**: two orphaned test files
  import modules deleted on 2026-02-02 (`tests/test_coding_direction.py` →
  `visdetect.analysis.coding_direction`, `tests/test_population.py` →
  `visdetect.analysis.population`; deletion commit `4f56700`), and pytest's default behaviour is
  to interrupt. Anyone "running the tests" gets **zero** test results and no failure they would
  recognise as one. With `--continue-on-collection-errors`: 1 failed, 654 passed, 4 deselected,
  2 errors in 627 s. The single failure is the **intended RED tripwire** of entry 4.
- **Affected modules** — none in the library; the defect is in `tests/` plus the `testpaths =
  tests` setting in `setup.cfg`. Its *consequence* falls on the 14 truly-untested modules
  (`d5.tests.untested_modules_ast`), which nothing would have caught anyway.
- **Affected artefacts** — none directly. Indirectly: every artefact in the tree was produced
  under a gate that reports nothing, which is the reason the other 34 entries exist.
- **Evidence** — `d5.tests.offline_runtime_s` = 630; `d5.tests.total` = 98;
  `d5.tests.untested_modules_ast` = **14** — cite the AST-corrected 14, not the shipped regex's
  32, which is a 2.3× overcount caused by an inability to match parenthesised multi-line imports.
- **Status** — LIVE. **Re-ingest** — SURVIVES until the orphans are deleted (see `drop-list.md`).

### A11. Delete-guard false positive blocks **every** recursive delete

- **Direction of effect** — `.claude/hooks/guard_recursive_delete.ps1` unconditionally adds `.`
  (the cwd = repo root) to its candidate list at `:85`, scans each candidate to
  `$MAX_DEPTH = 4` (`:37`), and denies whenever the hit list is non-empty (`:170`) **regardless of
  which candidate produced the hit**. The live depth-4 junction
  `.claude/worktrees/qc1-alignment/.superpowers -> <repo>/.superpowers` therefore trips every
  recursive delete run from this repo, whatever the target. The dangerous part is the *response*
  it invites: the guard's own suggested remedy is to delete the junction — which is a live
  worktree's access path and exactly the **2026-06-07 data-loss shape** the guard exists to
  prevent — or to reword the command to dodge the verb regex. Both are worse than the false
  positive.
- **It is worse than "recursive deletes": the verb regex matches source text.** During the Task-15
  fix pass the guard **denied a strictly read-only probe** — a `py -c` that loaded one pkl and
  printed its `ni_events` keys — because the Python source contained `del s` (freeing a local
  variable). Nothing was being deleted. So the false-positive surface is not "delete commands" but
  "any command whose text contains a delete-shaped token", which includes ordinary Python. The
  probe was re-run from a script file rather than by rewording the command, so the guard was not
  evaded.
- **Proposed fix** — scope the scan to path arguments the command actually names; fall back to
  `.` only when no path argument resolves. Independently, tighten the verb match so it cannot fire
  on in-language tokens like Python's `del`.
- **Affected modules** — none; this is repo tooling (`.claude/hooks/guard_recursive_delete.ps1`).
- **Affected artefacts** — none. The cost is operational: `data/cache/audit/nwbvenv` (321.7 MiB)
  is still on disk because the delete cannot be run, and it needs one owner command.
- **Evidence** — `d5.tooling.delete_guard_falsepositive` = `blocked-all-recursive-deletes`,
  evidence `.claude/hooks/guard_recursive_delete.ps1:85`; reproduced on a delete of
  `data/cache/audit/nwbvenv`, which was independently verified to contain **0** reparse points
  (1,204 dirs / 12,581 files walked; target not a link; resolved path == literal path; all
  ancestors plain); and reproduced a second time, on a read-only probe, as described above.
- **Status** — LIVE, standing condition. **Re-ingest** — SURVIVES: fix the guard, keep the
  junction.

### A12. No built distribution has ever contained `visdetect`

- **Direction of effect** — `py -m pip wheel .` **fails** (`error: package directory 'src\scripts'
  does not exist`): `setup.cfg` declares `packages = find:` with `package_dir = =src` but has no
  `[options.packages.find] where=src`, so `find:` scans the repo **root**, discovers the tracked
  `scripts/__init__.py` (present since `b8b0ee0`, 2026-06-18) and maps package `scripts` to a
  non-existent directory. Worse than the expected finding: before `b8b0ee0` wheels built **empty**
  — the pre-build egg-info's `top_level.txt` was empty and `SOURCES.txt` lists zero
  `src/visdetect` modules. The venv works only because the editable install degenerated to a bare
  `src`-path injection, and the **55 `repo-src` `sys.path.insert` sites are the de-facto
  distribution mechanism**. `visdetect.viz` and `visdetect.integrations` additionally lack
  `__init__.py`, so `find_packages` drops them even after a `where=src` fix, breaking ~50
  importers on any non-editable install.
- **Affected modules** — `visdetect.viz` and `visdetect.integrations` are the two packages
  `find_packages` drops for lack of `__init__.py`; every other module is affected only in that no
  distribution has ever contained it.
- **Affected artefacts** — none in `data/` or `FIGURES/`. The affected artefacts are the *build*
  ones: `src/visdetect.egg-info/` (gitignored, `top_level.txt` now reads `scripts` after the
  failed build) and the empty `data/cache/audit/wheel/`.
- **Evidence** — `d2.packaging.wheel_build` = FAIL; `d2.packaging.viz_missing` /
  `integrations_missing` = absent-a-fortiori; `d2.syspath.total` = 233.
  **Filter the audit's own rows before quoting the syspath census**: `syspath_sites.csv` contains
  **7** rows under `scripts/audit/`, of which `d2_layering.py:2`, `:30` and `:36` are the census
  matching its own docstring, comment and detector string — self-scan junk, not sites. Net of the
  audit's 8 total (7 + `tests/audit/test_audit_lib.py:5`), the tree has **225** sites.
- **Status** — LIVE. **Re-ingest** — DISSOLVED: the new repo packages itself properly.

### A13. The TF-GLM's movement regressor edge-counts a single encoder line

- **Direction of effect** — the "lick-and-movement-controlled" TF GLM builds its wheel covariate
  as **tick density of `ni_events['Rot_enc_A']`** — one line of a two-line quadrature encoder,
  unsigned, undecoded (`src/visdetect/analysis/tf_glm_data.py:501` documents it as
  "`Rot_enc_A` tick DENSITY (ticks/bin)"; `:521-522` reads and sorts the raw edge times). Counting
  raw edges conflates travel with back-and-forth jitter: in a quiet pre-change window the gross
  edge rate is **median 6/s against a dither-cancelled net displacement of median 1/s — ~6×
  manufactured movement.** Direction: the covariate **over-states movement precisely where there
  is none**, so the GLM regresses out variance that is dither, not locomotion, while leaving
  genuine signed travel under-controlled. Any claim resting on "movement was controlled for" is
  weaker than it reads, and the bias is largest in the quiet baseline windows the TF analysis
  cares most about.
- **The correct method is specified and cheap** — reconstruct A/B states and walk the Gray-code
  sequence (`00→01→11→10` one way, reverse the other), summing signed counts. On the measured
  session 977,531 transitions decode with **100 % validity** for a net displacement of −782,315
  counts. **Both lines are already in the pkls** — `Rot_enc_A` (226,405 events) and `Rot_enc_B`
  (226,758) on session `01072025` — so decoding needs no re-ingest, only the B line the current
  code ignores.
- **Affected modules** — `visdetect.analysis.tf_glm_data` (sole consumer;
  `git grep Rot_enc` finds exactly two non-test sites, both in this file).
- **Affected artefacts** — every TF-GLM fit whose design matrix included `wheel_bins`:
  `data/cache/tf_glm_bg046/*`, `data/cache/tf_responsive/*`, and the six `docs/science` docs
  resting on those registries (see entry 5, with which this compounds).
- **Evidence** — `tf_glm_data.py:501`, `:521-522`; the ~6× figure and the decoding recipe from
  `docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` §3; encoder-key presence and counts verified in-repo by
  a read-only probe of `data/pkls/BG_046/…01072025` (`ni_events` keys include `Rot_enc_A`,
  `Rot_enc_B`). Corroborating repo note: `scripts/docs/CHANGELOG_meta_file_validation.md:13`
  records "~226k rotary encoder events" polluting an earlier duration calculation.
- **A QC assertion the spec hands over free** — completed trials are near-motionless: net
  displacement in the second before the change is median **1.0** counts/s on Hit and Miss versus
  **7.0** on abort (Mann-Whitney p = 2 × 10⁻²³), with 26.8 % of Hit/Miss trials at *exactly zero*.
  If a decoding shows appreciable baseline travel on a completed trial, the decoding is wrong.
- **Status** — LIVE. **Re-ingest** — SURVIVES: the raw edges are already in the pkls; this is a
  decoding defect in analysis code, not an extraction defect. It is the one NI-layer entry that
  re-extraction does **not** fix.

### A14. NI event times are not stimulus times

- **Direction of effect — three quanta, one theme: the TTL is not the event.**
  1. **Display latency ≈ +67.3 ms (IQR 55–79, ~4 frames).** The photodiode — which watches a
     corner sync square and therefore timestamps when the stimulus *physically appeared* — lags
     the `Baseline_ON` TTL by 67.3 ms. **Aligning to the raw TTL places "stimulus onset" ~67 ms
     too early**, and the spec flags that this is *comparable to striatal visual latency itself*.
     Direction on any latency claim: measured response latencies are **overstated by ~67 ms**, and
     an onset-difference between conditions is unaffected only if the display lag is constant
     across them — untested.
  2. **A ~5 ms state-machine tick.** Every NI event time carries a quantised offset from software
     intent: measured `Change_ON` runs **5.128 ms** later than the programmed `stimT` (IQR 70 µs;
     295/323 trials at 5.1–5.2 ms, the rest at 10.2 ms = two ticks), reward follows the registered
     lick by **4.97 ms**, and every spurious TTL sliver is exactly 4.91/5.00 ms wide.
     Correlation of measured against programmed latency is **r = 1.000000** — the offset is a
     constant tick, not noise, so it biases absolute latencies and cancels in differences.
  3. **Two trials are 5 ms early in MATLAB's `Change_ON`** (trials 122 and 183), because MATLAB
     applied first-pulse-per-trial *without* first dropping sub-5 ms slivers.
- **Why one entry and not three** — they share a single mechanism class (NI timestamps ≠ physical
  or intended times), a single consumer set (everything that aligns to `Baseline_ON` or
  `Change_ON`), and a single remedy (capture photodiode onsets and record the tick), so a porter
  handles them together. Sub-fact 3 is a two-trial rounding artefact that would not merit an entry
  alone.
- **Affected modules** — `visdetect.analysis.align`, `analysis.constants` (the window constants
  are all defined relative to TTL time), `core.run_alignment`, and every module the map flags
  `alignment-QC1`.
- **Affected artefacts** — **every `Baseline_ON`- or `Change_ON`-aligned PETH, tensor, kernel and
  latency estimate in the tree.** No artefact is *wrong* in a way that changes an ordering; they
  are all shifted by the same amounts, which is why this is an attribution entry rather than a
  repair item.
- **Evidence** — `docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` §8 (display latency, tick) and §5
  (the sliver rule and the two early trials). The photodiode gives an 18.7× SD step at onset
  (0.037 → 0.698 V) confirming stimulus presence on 99.5 % of trials — but **cannot** validate
  `Change_ON`, because a TF change alters neither mean luminance nor contrast and the diode
  watches the sync square, not the grating. Independent in-repo corroboration of the tick:
  `src/visdetect/core/run_alignment.py:27` sets `ACCEPT_RESID_S = 0.05` with the comment
  "10x above the observed **0.0051 s** aligned value".
- **Scope caveat** — one session, BG_046 17092025; the spec says explicitly the display latency is
  "not yet verified on other sessions". The tick is corroborated independently by QC1's constant.
- **Status** — LIVE (a property of the acquisition, not a bug). **Re-ingest** — SURVIVES the NI
  re-extraction *unless the new pipeline captures photodiode onsets alongside the TTLs*. The
  photodiode is analog ch0 and is already in `nidq.bin`; nothing downstream has ever used it.

### A15. The optotag block→target mapping is assumed in code, and inverted on the one session where it was established

- **Direction of effect — potentially a straight swap of every D1/D2 pathway label from
  optotagging.** `split_laser_blocks` returns the two laser blocks **in time order** and its
  caller binds them positionally as `(gpe_pulses, snr_pulses)` — i.e. the code hard-assumes
  **first block = GPe (indirect, putative D2), second = SNr (direct, putative D1)**. The NI spec
  measured the opposite for BG_046 17092025: **block 0 (8908.4 → 9671.5 s) = SNr**, **block 1
  (9748.6 → 10511.8 s) = GPe**. If that ordering holds generally, every pathway label the
  optotagging module emits is **inverted**.
- **And the mapping has no recorded basis on either side.** The spec checked **all six** settings
  files and the block→target mapping is in none of them; it came from the experimenter. The
  code's assumption is likewise undocumented — `optotagging.py:505-510`'s docstring simply asserts
  "GPe (block 1) and SNr (block 2)". So this is not "the code disagrees with the truth"; it is
  **two undocumented assumptions that disagree with each other**, with no artefact in the repo
  able to arbitrate.
- **Why this is not filed as merely a naming nit** — the spec's screening found **16
  block-0-only responders against 1 block-1-only**. Under the spec's mapping that is 16 SNr
  (putative D1); under the code's it is 16 GPe (putative D2). The repo's standing claim is "3
  collision-confirmed units, **all D1**" (`optotagging_yield_jun2026`), which is consistent with
  the spec's mapping and inconsistent with the code's. That is suggestive, not decisive — the two
  results come from different sessions and different pipelines — but it is exactly the kind of
  delta ADR-009 exists to attribute.
- **Affected modules** — `visdetect.analysis.optotagging` (`split_laser_blocks` at `:500-537`;
  the positional bind at `:773`; `LASER_KEY` at `:38`).
- **Affected artefacts** — `data/cache/optotagging/optotagging_results.csv`,
  `optotagging_unit_tags.csv`, `optotagging_stats.csv`, `exemplar_fig_cache.pkl`, and every
  optotagging figure under `FIGURES/optotagging/` — wherever a `fiber` / GPe / SNr label appears.
- **Evidence** — `src/visdetect/analysis/optotagging.py:505-510` and `:773`
  (`self.gpe_pulses, self.snr_pulses = split_laser_blocks(all_pulses)`, first block → GPe);
  `docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` §10 (block 0 = SNr, block 1 = GPe; mapping absent from
  all six settings files; 16 vs 1 responder asymmetry; the SNr fibre far more effective,
  attributed to placement).
- **Two further cautions from the same section, carried so they are not lost** — never pool the
  blocks or rank by a pooled statistic (they identify different cell types, so pooling halves each
  response and buries the weaker block); and the Laser analog line is a **trigger command, not an
  optical power monitor** (identical TTL amplitude across blocks, 0.3490 vs 0.3487 V, says nothing
  about delivered light). The current code does split the blocks correctly — only the *labels* are
  in question.
- **Status** — LIVE, **direction known but polarity unresolved** → `quarantine.md` **Q12**.
  **Re-ingest** — SURVIVES: re-extraction recovers the pulses (E4) but cannot recover which fibre
  was in which structure. That fact exists only in the experimenter's records, and ADR-019 already
  requires it as acquisition metadata.

---

## Section D — documentation-layer entries

**Why these are in the register, and what they are not.** ADR-009 attribution is about numbers: a
wrong sentence produces no output, so **none of these may be used to tag a numerical delta**. They
are here anyway because the alternative placements are worse. The drop list is for content that
should be *dropped rather than analysed*, and neither `QUESTION_INDEX.md` (the live question map)
nor `2026-06-17-post-tf-null-research-direction.md` (a real record of a real pivot, still a ⭐
memory reference) should be dropped — the fix is a one-line correction, which is outside the
audit's write scope. `quarantine.md` is for claims whose validity is *undetermined*, and D2 is not
undetermined: it is known-wrong. So: register, in a section that is explicitly not part of the
attribution base. They move **design decisions**, not values — and D2 already voided one control
that way, which is the strongest argument for recording them somewhere a porter will look.

### D1. `docs/science/QUESTION_INDEX.md:49` asserts engagement-gating both child docs contradict

- **The claim** — "**VMS engagement-gated** (StimSens ≫ Disengaged)".
- **Walked back by** — memory `b10_impulsivity_kernel_jul2026`: the earlier "VMS strongly
  engagement-gated" was **per-trial pseudoreplication**; engagement modulation is **UNRESOLVED**.
  **Both results docs the index row links to say the opposite** ("Do NOT say 'tracking switches
  off when disengaged'").
- **Why it persisted** — the row is byte-identical since `39c19db` (2026-07-01); both correction
  commits (`e16fcd5`, `bfefa87`, 2026-07-02) edited only the two B10 results docs, and
  `QUESTION_INDEX.md` was edited three further times (2026-07-21, 2026-08-03 ×2) without the
  clause being touched. **The failure is the summary layer, not the science.**
- **Direction of effect** — misdirects design: a porter treating VMS engagement-gating as
  established would build on a refuted premise. **No numerical direction** — see this section's
  preamble.
- **Affected modules** — none. This is a documentation-layer entry.
- **Affected artefacts** — `docs/science/QUESTION_INDEX.md` (the B10 row at `:49`) only. The two
  child results docs it links to are **correct** and need no change.
- **Evidence** — `d6.science.stale_docs` = 4 (row 1). Flagged-not-fixed by Task 13: outside the
  audit's write scope.
- **Status** — LIVE. **Re-ingest** — SURVIVES: a prose claim, untouched by any data operation.
  Settles with a one-line edit adopting the children's wording.

### D2. `docs/science/2026-06-17-post-tf-null-research-direction.md:4,48` names the wrong region

- **The claim** — "…across BG_046 DMS, BG_031 striatum, **BG_039 cortex**, and **BG_038 GPe**";
  "Batch on **BG_039 (cortex/M2)** — tests whether the TF-null is *regional*".
- **Ground truth** — memory `multisubject_event_psth_readiness_jun2026`: BG_039 is **dorsal CP
  striatum (DMS)**, pool-compatible with BG_046; BG_038 is **cortex (MOp/SSp)**; the planning-doc
  "GPe" was a shank target, not a recording site (resolved 2026-06-30). Every `docs/science` doc
  from 2026-07-01 onward states "BG_046, BG_039 = DMS" — the corpus contradicts itself and only
  the older doc is wrong.
- **Direction of effect — load-bearing, not cosmetic.** The doc's argument is "four regions
  including cortex all at ≈ 0 % ⇒ the floor reflects the metric, not the biology" (line 6), and
  its recommended cheap control is to batch BG_039 to test regionality. BG_039 is the **same
  region** as BG_046, so **that control was void as designed**. A porter reusing the design
  reruns a control that cannot discriminate.
- **Affected modules** — none. Documentation-layer entry.
- **Affected artefacts** — `docs/science/2026-06-17-post-tf-null-research-direction.md` (lines 4
  and 48). No cache, figure or table carries the error; what it corrupted is a *design*, and the
  control it recommended was never run, so no artefact needs re-deriving.
- **Evidence** — `d6.science.stale_docs` = 4 (row 2). Flagged-not-fixed by Task 13.
- **Status** — LIVE. **Re-ingest** — SURVIVES: a prose claim. Settles with a one-line correction
  against `multisubject_event_psth_readiness_jun2026`, plus re-scoping the "cheap decisive
  control" onto a genuinely cortical subject (BG_038).

### D3. `docs/GOTCHAS.md:10` teaches the very footgun CLAUDE.md bans

- **The claim** — "Session name format | DDMMYYYY as integer (e.g., `7072025` = July 7, 2025).
  Use `parse_session_date()` and `chronological_sort()`." It recommends the **integer** form and
  never mentions `canonical_session_id()`.
- **Direction of effect** — an agent that opens `docs/GOTCHAS.md` instead of CLAUDE.md is
  **instructed to create defect 4**, whose measured cost in this repo is 15,802 corrupted cache
  rows plus entry A2's 1,670. Related twin found incidentally:
  `docs/AI_interaction/copilot-instructions.md` and `.github/copilot-instructions.md` (the copy
  Copilot actually loads) are **not byte-identical** — same divergence class.
- **Affected modules** — none directly. The *correct* API it fails to mention lives in
  `visdetect.analysis.config` (`canonical_session_id`, and for cross-subject columns
  `restore_session_token` / `canonicalize_session_column` — see A1).
- **Affected artefacts** — `docs/GOTCHAS.md:10`; and, causally, the corrupted caches of entries 4
  and A2 (17,539 rows), which are the measured cost of following this advice.
- **Evidence** — `docs/GOTCHAS.md:10`; recon line-overlap CLAUDE.md ↔ GOTCHAS.md = 78 %;
  `d6.authority.claimants` = 4 (only CLAUDE.md is loaded by the harness);
  `d6.deadpaths` = 111 refs / 53 unique paths / 24 docs, including **2 dead refs inside CLAUDE.md
  itself** (lines 224–225, pointing at `analysis_suite/utils.py`, archived 2026-07-01) and dead
  refs in two **active skills** (`research-statistician/SKILL.md:272`,
  `research-visualizer/SKILL.md:254`) that load into working sessions.
- **Status** — LIVE. **Re-ingest** — DISSOLVED by ADR-005 (the new repo deletes the copies and
  generates `CLAUDE.generated.md`), but only if the dead-path check covers the *prose* half too.

---

## Module coverage — acceptance criterion A2

All **64** library modules are classified against the register
(`d8.modules.classified`, `data/cache/audit/module_register_map.csv`); **31** match no register
symbol (`d8.modules.clean`); **6** use a session-id canonicaliser and are marked as *mitigating*,
not affected.

**Three caveats a porter must apply before trusting a `clean` verdict:**

1. **`clean` means "matches no symbol pattern", not "defect-free".** The classifier's
   `DEFECT_SYMBOLS` table predates the five ephys entries **and the three NI-spec entries added in
   the fix pass**, so it has no pattern for any of them. The clearest casualty is
   `visdetect.core.ingest`, which reads `clean` while being the module at the centre of **E1**
   (`ingest.py:415`, `:492-495`) *and* of entry 8's duplicate-JSON mechanism (`ingest.py:71-98`).
   `visdetect.analysis.optotagging` likewise reads `clean` and carries **A15**. Entries E1–E5 and
   A1–A15 carry their affected modules **by hand, from `file:line`**, in the entries above — read
   those, not the CSV, for those defects. The classifier was **not re-run** in the fix pass, so
   `d8.modules.classified` = 64 and `d8.modules.clean` = 31 are unchanged and still correct for
   what they measure.
2. **`alignment-QC1` is a blast radius, not a bug list.** The pattern includes `Change_ON`, so it
   flags every module that aligns to change onset (**14** modules). That is the correct exposure
   set for QC1 attribution; it does not mean 14 modules are broken. Per-entry module counts:
   `alignment-QC1` 14, `id-corruption` 10, `change-sizes-membership` 8, `ref-ambiguity` 6,
   `lick-channel` 5, `session-order` 5, `stale-tf-registries` 4, `qc-profile-noop` 2,
   `state-tags` 2, `tf-period-5x` 2.
3. **`change-sizes-membership` is now known to be a naming hazard, not a numerical defect**
   (entry 12), so its 8 flagged modules are a *review* set, not a *repair* set.

**Modules carrying three or more register entries** — the ones to port last, or first, depending
on appetite:

| Module | Entries |
|---|---|
| `analysis.config` | change-sizes-membership, id-corruption, ref-ambiguity, session-order, tf-period-5x (+ A1) |
| `analysis.constants` | alignment-QC1, change-sizes-membership, ref-ambiguity, tf-period-5x |
| `analysis.tf_glm_data` | alignment-QC1, change-sizes-membership, lick-channel, ref-ambiguity (+ **A4, A13**) — **6 entries, the most exposed module in the library** |
| `analysis.tf_glm` | change-sizes-membership, lick-channel, stale-tf-registries |
| `analysis.tracking_qc` | alignment-QC1, change-sizes-membership, id-corruption (+ E3, E5) |
| `analysis.decision_latents` | change-sizes-membership, id-corruption, session-order |
| `suite.loader` | id-corruption, session-order, stale-tf-registries (+ A6) |
| `core.ingest` | *reads `clean`* — but carries **E1** and entry 8's duplicate-JSON mechanism |
| `analysis.optotagging` | *reads `clean`* — but carries **A15** and is the consumer that fails on **E4** |
