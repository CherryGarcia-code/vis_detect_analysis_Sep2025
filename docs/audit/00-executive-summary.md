# Sub-project 0 — Empirical audit: executive summary

**What this is.** The capstone of the 16-task empirical audit of
`vis_detect_analysis_Sep2025`, run 2026-08-09 → 2026-08-18 on branch
`design/new-repo-foundation`. Every number below is cited by a `measurement_id`
resolving to a row of `docs/audit/measurements.csv` (118 rows: id, value, command,
script, evidence, notes). A handful of figures are **inherited** from sources outside
the audit's measurement set (the NI/event spec, memory records); each is labelled
`inherited` with its document anchor and must not be quoted as an audit measurement.

**The corpus** (the reports are the deliverable; raw evidence stays in
`data/cache/audit/`, force-added against the `data/cache/*` gitignore):
domain reports `01-constants.md` … `07-work-at-risk.md`, `09-storage-spike.md`;
`known-defect-register.md` (**37** entries, `d8.register.entries`);
`quarantine.md` (**12** entries Q1–Q12, `d8.quarantine.entries`);
`drop-list.md` (**45** objects, `d8.droplist.items`); `cold-list.md`
(seeded by **26** cold modules, `d8.coldlist.modules`); `branch-disposition.md`;
`measurements.csv`. The register maps all **64** library modules
(`d8.modules.classified`), **31** touching no entry (`d8.modules.clean`).

---

## 1. Ranked findings

Ranked by consequence for the rebuild, cross-domain. Register entry numbers
(1–12, E1–E5, A1–A16, D1–D4) refer to `known-defect-register.md`.

**F1 — Session identity is systemically corrupted: 17,539 rows across live
artefacts.** The total is the sum of two measured classes:
**15,869** stripped/00-padded rows in caches, FIGURES and table_output
(`d4.ids.rows_corrupt` — 15,802 of them the 6 offender caches the deliberately-RED
integrity test names, `d4.ids.integrity_test_red`, plus **67** unrepairable
`00DDMMYY` rows *manufactured by* `canonical_session_id` itself,
`d8.canonical.ddmmyy_behaviour`), plus **1,670** five-digit day-stripped DDMMYY rows
from the non-BG_046 subjects that no previous count captured
(`d8.idcorruption.fivedigit_rows`). **10** of **327** scanned id-bearing files are
corrupt (`d4.ids.files_corrupt`, `d4.ids.files_scanned`). The mechanism is fed by
**27** local date-parser sites (`d8.dateparser.recount`, AST census; supersedes both
the shipped 19, `d3.dateparser.sites`, and the hand-count 23), and the 7-digit form
parses to a silently WRONG date (`d3.dateparser.trio`: `1072025` → 2025-07-**10**).
Register 3, 4, A1, A2, A6.

**F2 — The qc-profile no-op is confirmed by execution.** All four named profiles
return `{}` (`d1.qcprofile.default`, `.qc_only`, `.striatal_strict`,
`.striatal_lenient`): every `--profile` invocation ever made silently ran function
defaults. Blast radius today is bounded: all four YAML-intended profiles pass the
**same 108 units** on the probe session (`d1.qcprofile.diff.*`) because the
irreversible ingest gate dominates — pkls store spikes only for `good_and_stable`
clusters (108 of 260 KS-good there; register E1), so unit counts can only FALL
without re-ingest. The live selection paths span **108/92/108** units
(`d1.frfloor.spread`); the mechanism is the `parents[1]` fragile-root idiom, **4**
sites (`d2.parents.sites`). The defect that remains is the silent failure mode.
Register 1, A3, E1.

**F3 — The NI/event layer is not what analyses assumed, and the re-extraction
method is now settled.** "Re-ingest from raw" as coded is raw behaviour + the MATLAB
NI product (`ingest.py:444`; register callout). The audited
`docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` re-extracted one session from `nidq.bin`
and matched MATLAB exactly on the threshold-insensitive channels (739/739, 323/323,
251/251 at 0.0000 ms — inherited, quarantine Q6). What it establishes:
the lick under-detection root cause (piezo thresholded as TTL, ~14.9 % reward
coverage — inherited, register 6); BG_031's Laser gap is structural, **35/43**
sessions lack the key (`d8.bg031.laser_missing`, register E4); and NI event times
are not stimulus times (`Baseline_ON` leads the screen by a median +67.3 ms, sd
14.6 ms, not frame-locked — inherited, register A14, spec §8). Per-session
generalisation is open (Q6). Register 6, 8, E4, A14.

**F4 — The single highest-consequence forward rule: the time-base trap (Q6 trap
0).** NI sample indices MUST be converted at the meta `niSampRate` = **10593.2 Hz**,
never the plausible-looking sync-fitted rate: the fitted value is NI samples per
imec-basestation-second, and adopting it misaligns events against spikes by a
46.1 ms median residual with a −8.49 ppm ramp (vs 13.9 µs median under the meta
rate — inherited, spec §4, quarantine Q6 trap 0). Every new-pipeline extraction must
assert the event-vs-spike residual ≲ 100 µs post-extraction, and any stored `*_t`
array in the companion `tmpclaude-BG_046_17092025/` tree is fitted-rate until
re-derived. This rule silently decides whether every aligned analysis in the new
repo is valid.

**F5 — Optotagging pathway polarity is unresolved — an OWNER decision, not more
audit.** The code hard-assumes first laser block = GPe; the spec's one measured
session says first block = SNr (`d8.optotag.block_map_conflict`). Every D1/D2
pathway label from optotagging may be swapped. The per-block screen found 17
block-0-only responders, 3 both, 0 block-1-only, 199 untestable (reported
separately), 437 negative (all within `d8.optotag.block_map_conflict`). Antidromic
identification is *plausible but NOT established* — the collision test is invalid
under the 10 ms sustained pulse. Settles documentarily or anatomically, not by
compute. Register A15, quarantine Q12.

**F6 — Provenance vacuum: 0.42 of censused figures have no identifiable
producer** (`d4.trace.untraceable_frac`, n = **159**, `d4.trace.sample`; all **83**
git-tracked deliverable figures included, `d4.trace.tracked_covered`). All **7**
measurable cache topics are stale or writer-untraceable (`d4.stale.topics`); a
frozen sorter output carries **14** chronologically impossible rows
(`d4.stale.chron_impossible`); direct manifest readers see **18** extra sessions
(`d4.filter.divergence`, register A5); and whether `tf_responsive` calls flip
post-lick-fix is not-measured (`d4.tfresp.flips`) — the VMS>DMS headline stays
unsafe (register 5).

**F7 — The package has NEVER been buildable.** `pip wheel` fails outright
(`d2.packaging.wheel_build`); `visdetect.viz` and `.integrations` are absent from
any distribution a fortiori (`d2.packaging.viz_missing`,
`d2.packaging.integrations_missing`) — no `visdetect` has ever been in dist
metadata; the venv works only via the editable-install path hack. Around it:
**233** `sys.path` mutation sites (`d2.syspath.total`), **18** pointing at
non-existent foreign trees (`d2.syspath.foreign_missing`); dual import roots — **7**
`src.visdetect` importers, **6** mixing both roots (`d2.dualroot.src_importers`,
`d2.dualroot.mixed`, register A7); **3** upward layer edges
(`d2.layers.upward_module_level`); and a **10.65 s** cold import of `visdetect`
(`d2.importtime.visdetect`) — **9.11 s** just for `constants`
(`d2.importtime.visdetect.analysis.constants`). Register A12, A7.

**F8 — The test gate reports success by reporting nothing.** The default
`py -m pytest` dies at collection on two orphaned test files (register A10;
drop-list §2.3). Run properly, the offline partition gives **654 passed, 1
expected RED** (the id-integrity tripwire, kept RED on purpose,
`d4.ids.integrity_test_red`), 4 deselected slow, in **630 s**
(`d5.tests.offline_runtime_s`). Of **98** test files (`d5.tests.total`), **14**
need real data (`d5.tests.need_real_data`). Truly untested library modules number
**14** (`d5.tests.untested_modules_ast`) — the shipped figure of 32
(`d5.tests.untested_modules`) is a regex artefact and must not be quoted.

**F9 — Constants: one canon, many truths.** **82** canonical constants
(`d1.constants.total`): **16** dead (`d1.constants.dead`, drop-list §2.9), **42**
not re-exported by config (`d1.constants.not_reexported`), **0** with disagreeing
retyped copies (`d1.constants.shadow_disagree`). But **43** *scientific* parameters
are re-declared with DISAGREEING values across scripts
(`d8.constants.scientific_divergent`, register A16) — the class an ADR-009 porter
will hit most often. `TF_SAMPLE_PERIOD = 0.25` has **zero** live value-readers
(`d8.tfperiod.value_readers`) — the 5× hazard is a wrong canonical value beside 77
unlinked `dt` literals (enumerated under `d1.tfperiod.consumer_sites` = 83).
Palettes: **692** hex literal occurrences, **174** distinct
(`d1.palette.hex_total`, `d1.palette.hex_distinct`), led by canonical palette
values re-hardcoded per script (`d1.palette.top_hex`).

**F10 — Two quarantines resolved into settled conventions.** *Ref trials*: the
change WAS presented — **18/18** ref trials carry a valid `change_time`
(`d1.ref.total`, `d1.ref.with_change_time`), median RT **+83 ms** after onset
(`d1.ref.rt_median_ms`); excluding `ref` from `Change_ON` alignment is a
**scientific choice, not a data fact**, and the hardware second source is the
spec-§9 set-equality-with-identical-trial-sets argument — NOT the retracted
surplus-pulse argument, NOT the `Baseline_ON`-ends-first argument (register 11).
*CHANGE_SIZES*: **0** consumers mix catch into a go loop
(`d8.changesizes.catch_in_go_loops`) — a naming hazard, not a numerical defect
(register 12).

**F11 — The AI/doc layer actively misleads.** **111** doc references to
non-existent paths (`d6.deadpaths`); **4** files claim canonical authority while
only CLAUDE.md is loaded (`d6.authority.claimants`); of **20** SYMBOL=value claims
checked, **4** mismatch the code and **1** names a symbol that does not exist
(`d6.literals.checked`, `d6.literals.mismatch`, `d6.literals.symbol_missing`);
**4** science docs assert walked-back claims with no in-doc marker
(`d6.science.stale_docs`); **3** stale model-id pins sit in active skills
(`d6.modelids`). Register D1–D4.

**F12 — Work at risk, physically.** **31** commits exist on no origin ref
(`d7.local_only.commits`) and the remote is unverifiable from this checkout
(quarantine Q9); the irreplaceable hand-label sets (224 files, 31 MB) are backed up
on the SAME physical disk (`d7.handlabels.exposure`); **5/6** untracked
working-tree files exist on no ref (`d7.untracked.at_risk`); the primary tree
carries 72.5 GB gitignored data + 37.8 GB FIGURES that no migration moves
(`d7.gitignored.volume`). The refactor guardrail was repaired mid-audit:
**1,590 → 220** real HARD violations (`d5.guardrail.before`,
`d5.guardrail.after`); the delete-guard hook falsely blocks EVERY recursive delete
(`d5.tooling.delete_guard_falsepositive`, register A11).

**The D9 storage spike was consciously not run.** All four spike measurements are
`not-measured` **by decision, not failure** (`d9.size_ratio`, `d9.readtimes`,
`d9.roundtrip`, `d9.compression`): on 2026-08-13 the project owner pre-decided NWB
+ re-ingest from raw, making the pkl→NWB comparison moot. `d9.keep_all_good` is
partial: **code-side YES** (the ingest chain never reads `.ap.bin`), **data-side
not-measured** — whether the Kilosort trees are complete on `X:` is quarantine Q5,
the audit's highest-priority open check. Carry-forward: the NWB writer must assert
its codec post-write, or compression drops silently (register A9).

---

## 2. Must fix before building

Work the new design does NOT do by itself. ADR pointers reuse the register's.

| # | Item | Anchor | Lands in |
|---|---|---|---|
| M1 | **Time-base rule + post-extraction assert** — meta rate 10593.2 Hz only; residual ≲ 100 µs or fail (inherited, spec §4; F4) | Q6 trap 0 | ADR-017 (typed time base) |
| M2 | **NI re-extraction from `nidq.bin`** per the audited spec — dissolves register 6, plausibly E4 (`d8.bg031.laser_missing`), unblocks 8/QC1; channel traps + edge-pairing asserts (Q6 items 1–5) | Q6; register 6, 8, E4 | ADR-015, sub-project 1 |
| M3 | **The Q5 `X:` sweep** — Kilosort-tree presence/completeness, raw/NI inputs, one BG_031 laser session; settles `d9.keep_all_good` data-side. The one pre-authorised `X:` read; spend it here | Q5, Q6 item 2 | sub-project 1 gate |
| M4 | **OWNER: optotag block→target polarity** — documentary or anatomical settle; record as acquisition metadata (F5) | A15, Q12, `d8.optotag.block_map_conflict` | ADR-019 |
| M5 | **OWNER: sibling-repo boundary** — publish the task-semantics layer or record the divergence; the copies already disagree (**12** files, `d7.sibling.duplication`) | A8 | ADR-011 / ADR-015 |
| M6 | **Off-disk backup of hand labels** — current backup shares the disk (`d7.handlabels.exposure`) | 07-work-at-risk | ADR-022 |
| M7 | **Push, then re-verify branch dispositions against the real remote** — **31** local-only commits (`d7.local_only.commits`); every drop-list branch row is conditional on `ls-remote` | Q9, branch-disposition | process |
| M8 | **NWB writer codec assert** — set DataIO on the concatenated column, assert with h5py post-write | A9, `d9.compression` note | ADR-015 |
| M9 | **Implement the chronic-drift control row** (composition matching / days-from-implant covariate) before any cross-stage cell-type claim; the 89→15 % figure is inherited, unmeasured here (Q7) | E2 | ADR-018 |
| M10 | **Old-repo transition hygiene** — drop the two collection-breaking orphan tests (restores the gate, register A10, drop-list §2.3); fix the delete-guard scan scoping (A11); keep the integrity test RED (`d4.ids.integrity_test_red`) until the 6 offender caches are repaired or rebuilt | A10, A11 | ADR-003 |
| M11 | **Declare `EVENT_VALID_OUTCOMES` exclusions as choices** with reasons, in one typed outcome enum (two constants currently encode one rule in two casings/memberships) | register 11 | ADR-009 |
| M12 | **Carry A14 into event semantics** — NI event times ≠ stimulus times; latency and threshold numbers re-derived per session, never transplanted (Q6 item 1) | A14 | ADR-015 / ADR-019 |

## 3. Made impossible by the new design

Classes that cannot recur once the cited ADR is implemented as specified —
"impossible" is by construction, contingent on that implementation. The register
remains the ADR-009 attribution base when a ported number differs.

| ADR | Kills this class | Register / evidence |
|---|---|---|
| ADR-004 (generated identity registry) | The entire session-id corruption class (F1): int64 round-trips, `00DDMMYY` manufacture, 27 parser sites, twin ambiguity, the **77** redundant `zfill(8)` sites (`d3.zfill.sites`) | 3, 4, A1, A2, A6; `d4.ids.rows_corrupt`, `d8.idcorruption.fivedigit_rows`, `d8.dateparser.recount` |
| ADR-005 (generated AI layer) | Doc rot: dead paths, authority claimants, literal drift, the GOTCHAS twin teaching the footgun | D3; `d6.deadpaths`, `d6.authority.claimants`, `d6.literals.mismatch` |
| ADR-001 / ADR-002 (clean room; hard layers) | Unbuildable package, dual import roots, `sys.path` sprawl, upward edges, 10-second imports (F7) | A7, A12; `d2.*` |
| ADR-003 (mechanical gates first) | The zero-result test gate class (F8) | A10 |
| ADR-008 (sibling dir, no junctions) | The junction data-loss shape that the broken delete guard over-defends against | A11; `d5.tooling.delete_guard_falsepositive` |
| ADR-009 / ADR-010 (one importable truth; paved road) | Per-name scientific-parameter divergence (F9); palette re-hardcoding | A16; `d8.constants.scientific_divergent`, `d1.palette.hex_total` |
| ADR-015 / ADR-017 (typed store; typed time base; content-addressed caches) | The stale-cache class (`d4.stale.topics`), untyped time bases (F4's structural fix), silent CSV round-trip corruption | 5; `d4.stale.chron_impossible` |
| ADR-018 (QC named, versioned, strata-not-verdicts) | Silent qc-profile no-ops (F2 — a named profile that fails to load must fail loudly), hardcoded verdict sets, the manifest fork | 1, E5, A5; `d1.qcprofile.default` |
| ADR-019 (publication layer) | The untraceable-figure class going forward (`d4.trace.untraceable_frac` = 0.42 stays unrecoverable historically — Q3) | 2, A14; `d1.tfperiod.figure_attribution` |
| ADR-022 (backup as policy) | Same-disk "backup" exposure | `d7.handlabels.exposure` |

## 4. Honest gaps

Not-measured, each with rationale and settling check: `d1.tfperiod.measured_s`,
`d1.tfperiod.figure_attribution` (Q3), `d3.lick.overlap` (Q4, needs the forbidden
`X:` read), `d4.tfresp.flips` (Q2, needs the GLM recompute),
`d5.tests.realdata_runtime_s`, `d2.sideeffects.import` (partial),
`d6.dup_pair_agreement`, the four D9 spike rows (by decision), and
`d9.keep_all_good`'s data side (Q5). Quarantine Q11 aggregates the residual gaps.
Only ONE register entry has unknown direction: E3 (`d8.register.quarantined`, → Q1).

---

## 5. Acceptance self-check (spec §5)

Performed 2026-08-18 on this summary and the committed corpus; commands run for
real, results reported as returned.

**A1 — evidence + command spot-check, 5 random rows.** Seeded draw
(`Get-Random -SetSeed 42 -Count 5` over the 118 rows): `d6.modelids`,
`d6.dup_pair_agreement`, `d1.tfperiod.consumer_sites`, `d7.branches.unmerged`,
`d5.guardrail.after`. All 5 carry a runnable command. Evidence: 2/5 in the evidence
column as paths, both verified to exist (`tf_dt_sites.csv`,
`check_refactor_guardrails.py`); 2/5 carry evidence inline in value/notes
(`d6.modelids` file:line hits; `d7.branches.unmerged`'s value IS the inventory);
1/5 (`d6.dup_pair_agreement`) is an honest `not-measured` whose note states the
reason and disposition. Blast radius lives in the register entries, per the corpus
convention. **PASS**, with the noted column-vs-notes wrinkle.

**A2 — module map covers all library modules.** `module_register_map.csv` has
**64** rows, 0 with empty classification; `src/visdetect` holds exactly 64
non-`__init__` `.py` modules. 64 = 64 (`d8.modules.classified`). **PASS.**

**A3 — every must-differ entry has a direction.** Parsed the register's 37 entry
headers: **37/37** carry a `**Direction of effect**` field (an initial stricter
grep found 34 — the other 4 use the `**Direction of effect — …**` em-dash form).
Exactly 1 entry is direction-unknown by status (E3 → Q1), matching
`d8.register.quarantined` = 1. **PASS.**

**A4 — census fraction with sample size.** `traceability_sample.csv`: 159 rows;
producers `untraceable: 67` → 67/159 = **0.421**, matching
`d4.trace.untraceable_frac` = 0.42 at n = 159 (`d4.trace.sample`), with all 83
tracked deliverable figures covered (`d4.trace.tracked_covered`). **PASS.**

**A5 — corpus is the reports, raw stays out.** `docs/audit/` holds exactly the 14
report .md files (this summary included) + `measurements.csv` (+ `.gitkeep`). The
23 raw evidence files are
tracked under `data/cache/audit/` against the `data/cache/*` gitignore rule
(verified via `git check-ignore` + `git ls-files`). No raw CSV lives in
`docs/audit/`. **PASS.**

**A6 — number traceability grep.** Sections 1–4 (the findings; this section
reports the check itself) were split into paragraph blocks (blank-line separated;
table rows individually): 49 blocks. Every block containing a numeral was flagged
unless it carries a backticked `d*.` measurement id or an explicit `inherited`
label with document anchor. Result: **9 of 49** blocks flagged, and on inspection
every one contains only structural numerals — section headings, register entry ids
(1–12/E/A/D), quarantine ids (Q1–Q12), ADR numbers, and M-row labels. **0 measured
values are uncited.** (The first run flagged 11; two were real gaps fixed before
commit — M1's spec-derived 10593.2 Hz/100 µs lacked its `inherited` label, and A5's
report-file count predated this summary's own existence.) The check command and raw
output are in the Task 16 report
(`.superpowers/sdd/2026-08-09-subproject-0-audit/task-16-report.md`). **PASS.**

---

*Sub-project 0 complete. The audit changes no analysis code; its deliverable is
this corpus. Next: sub-project 1 (data layer), gated on M1–M3.*
