# Quarantine — undetermined, with the check that settles each

Companion to `known-defect-register.md`. Under ADR-009 a defect whose **direction** cannot be
stated attributes nothing, so it must be quarantined explicitly rather than ported on assumption —
every delta it produced would be unexplained by definition. This file lists everything the audit
could not determine, and, for each, **the specific check that settles it**.

Two entries the spec quarantined are **no longer here** — they were resolved and moved into the
register:

| Was quarantined | Resolved by | Now |
|---|---|---|
| `ref`-trial change-presented ambiguity | Task 4 (`d1.ref.with_change_time` = 18 = `d1.ref.total`, median RT +83 ms), **and independently at the hardware level** — each trial's own `Baseline_ON` pulse ends *before the change was scheduled* on 100 % of FA trials (n=263, median margin 3.163 s) and 100 % of aborts (n=153, 4.749 s) | Register entry **11**, status SETTLED-CONVENTION, two independent sources |
| `CHANGE_SIZES` membership divergence | Task 15 per-consumer check (`d8.changesizes.catch_in_go_loops` = 0) | Register entry **12**, status SETTLED-CONVENTION |

**Q6 was substantially resolved after the audit's first pass** by
`docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` (2026-08-13/14) and is rewritten below rather than closed:
the method is settled and validated, per-session generalisation is not. **Q12 is new**, raised by
the same document. Both changes are Task-15-fix additions.

> **On the new evidence source.** `docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` was produced outside this
> audit and is **read-only** to it. It was **untracked** during the Task-15 fix passes — a single
> uncommitted copy that transiently made `d7.untracked.at_risk` 6 of 7 — and has since been
> **committed** (`da5fbf9`, 2026-08-15, 628 lines), returning that count to 5 of 6; the timeline is
> in the CSV note. Its scope caveat is load-bearing and is
> repeated wherever it is cited: **one session (BG_046, 17 Sep 2025) of one subject**, with every
> numeric constant to be re-derived per session.
>
> ⚠ **Cite only the audited version.** The document was adversarially reviewed by six independent
> reviewers and **explicitly retracts several of its own first-pass claims**. This audit corrected
> its register and quarantine entries against the audited text on 2026-08-14; anything marked ❌
> there is refuted and must not be re-imported. The retracted set is listed in the register's
> evidence-source note.

Ordered by how much they block. **Q1–Q5 block sub-project 3 or sub-project 1; Q6–Q12 do not** —
with the exception that Q6's *remaining* part is a scoping question for sub-project 1's `X:` pass
and Q12 blocks any D1/D2 pathway claim from optotagging.

---

## Q1 — BG_046-calibrated track-QC thresholds applied to other subjects

- **Register entry** — E3. **Direction: unknown**, which is the whole problem: applied to
  BG_031/038/039 the thresholds could be too permissive (admitting matching errors as tracked
  units, inflating tracked-cohort n and understating drift) or too strict (discarding real tracked
  units, understating n and overstating stability). Nothing in the repo measures which, so **no
  tracked-cohort claim on a non-BG_046 subject can be attributed** today.
- **Evidence of the calibration** — `src/visdetect/analysis/tracking_qc.py:52`, `:62`, `:69`,
  whose comments read "Calibrated to BG_046 cohort distribution (May 2026)" and
  "top ~25 % of BG_046 cohort".
- **The check that settles it** — re-derive the same three thresholds **per subject** from that
  subject's own distribution, using the negative-control design ADR-018 already names:
  **within-day split-half sorts** as known-positive links and, where the probe is multi-shank,
  **across-shank pairs** as known-negative links. Then report, per subject, the tracked-cohort
  size and the split-half AUC under (a) the BG_046 thresholds and (b) the subject's own. If the
  two cohorts differ by less than the tracker's own link-score noise, the entry closes as
  "no material effect"; otherwise the sign of the difference **is** the direction and the entry
  moves to the register.
- **Cost** — no `X:` access; the UnitMatch/DANT inputs are already under `data/unit_match/` and
  `data/cache/dant/`.
- **Blocks** — any multi-subject tracked-cohort claim (sub-project 3).

## Q2 — How many `tf_responsive` calls flip after the lick-channel fix

- **Register entry** — 5. Direction is known (borderline `resp_log2` calls **will** flip, and the
  **VMS 5.3 % > DMS 2.8 % / 3.1 %** ordering is unsafe); the **magnitude** is not, so no
  registry-derived number can be attributed, only distrusted.
- **Evidence** — `d4.tfresp.flips` = **not-measured**; `data/cache/tf_responsive/README.md`.
- **The check that settles it** — the cheapest sizing path is the one the cache's own README
  names: a **paired within-unit re-fit on identical seed-fixed CV folds**, restricted to the
  ~150–500 near-threshold units, old regressor vs new. Report the flip count and, separately,
  whether the region ordering survives. Six `docs/science` docs rest on this registry and mention
  it in none, so the flip count is what tells the porter which of them need re-deriving.
- **Cost** — S-layer compute (a GLM re-fit), not audit compute. Not an `X:` question.
- **Blocks** — every TF-responsiveness claim; six results docs.

## Q3 — Which published figures and caches were produced under which `dt`

- **Register entry** — 2. The `TF_SAMPLE_PERIOD = 0.25` constant turns out to have **zero live
  value-readers** (`d8.tfperiod.value_readers` = 0), so nothing *today* is 5×-coarse through it.
  What cannot be established is whether any **historical** output was, because the repo has no
  per-figure provenance.
- **Evidence** — `d1.tfperiod.figure_attribution` = **not-measured**;
  `d4.trace.untraceable_frac` = 0.42 (67 of 159 censused figures have no mechanically
  identifiable producer) — the gap itself, measured.
- **The check that settles it** — none is available in the old repo, and **this is the honest
  answer**: reconstructing per-figure `dt` provenance retrospectively is not possible without the
  sidecars that do not exist (the sidecar tier fired **0 times in 159 rows**). The entry closes
  not by measurement but by ADR-019's rule: **no figure produced by the old repo may enter the
  manuscript**. Re-derive rather than audit.
- **Blocks** — nothing, once the "old figures are archive" rule is honoured.

## Q4 — Which unguarded lick-channel scripts touched the 33 re-extracted sessions

- **Register entry** — 6. Direction is bidirectional and known per code path; what is missing is
  the **session list**, so an affected historical output cannot be identified by name.
- **Evidence** — `d3.lick.overlap` = **not-measured**: the 33-session MATLAB re-extraction batch
  list is not materialized anywhere in the repo, and deriving it requires NI-file inspection on
  `X:`, which the audit forbids.
- **The check that settles it** — materialize the batch list **as data**, once: for each BG_046
  session, record which NI channel names its `*NIdaq_events.mat` carries (`Lick_L`/`Lick_R` =
  2025 extraction, `Piezo_1`/`Piezo_2` = 2026-03-06 re-extraction). That is a per-session key
  read, not an analysis. `visdetect.analysis.lick_channels` already knows how to answer it —
  it just has never been run as a census and the answer never written down.
- **Cost** — one `X:` read, and it should be folded into the **same sweep as Q5** rather than
  spent separately.
- **Blocks** — naming which historical lick-rate figures are affected.

## Q5 — Are the Kilosort trees (and the behavioural / NI inputs) present and complete on `X:`

- **Register entry** — E1. The **code side is already settled: YES** — `keep_all_good=True`
  exists (`src/visdetect/core/ingest.py:415`, branch at `:492-495`) and the ingest chain reads
  only Kilosort/Phy `.npy`/`.tsv`, with no `.ap.bin`, `memmap` or `np.fromfile` anywhere in
  `src/visdetect/core/`. What is open is a property of the **data store**, not the code.
- **Evidence** — `d9.keep_all_good` = *code-side YES; data-side not-measured*.
- **The check that settles it** — one pre-authorised `X:` **existence-and-completeness sweep** at
  sub-project 1, per session per subject. It must cover **three input families, not one** — the
  Task-14 wording scoped it to Kilosort trees and that understates it:
  1. the Kilosort/Phy tree (`spike_times*.npy`, `spike_clusters*.npy`,
     `cluster_KSLabel.tsv` / `cluster_group.tsv`, `templates.npy`);
  2. the **behavioural trials** in `raw_dir` (`ingest.py:441`);
  3. the **`*NIdaq_events.mat`** in the processed dir (`ingest.py:444`, glob at `:305`).
  Fold Q4's channel-name census into the same pass. Do **not** spend the sweep re-deriving the
  code-side answer.
- **Blocks** — the entire raw re-ingest plan, format-independent. **This is the highest-priority
  quarantine entry.**

## Q6 — The NI re-extraction: **method settled, generalisation open**

**Substantially resolved on 2026-08-13/14, after the audit's first pass.** This entry previously
read "an unmade decision". It is no longer one.

- **What is now settled.** `docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` re-extracted one BG_046 session
  (17092025) directly from `nidq.bin` and (a) **confirms the diagnosis** — the existing pipeline's
  NI half reads the MATLAB product (`build_session_from_raw` → `*NIdaq_events.mat`,
  `ingest.py:444`; it never opens `nidq.bin`), while its *behavioural* half genuinely does read
  raw JSON (`load_behavioral_trials` globs `Session/*trials.json`, `ingest.py:71-73`) — and
  (b) **supplies a validated recipe**: settings-first discipline, channel map parsed from
  `~snsChanMap`, per-channel threshold derivation from observed levels, a merge-then-first-pulse
  edge rule, and a 13-step pipeline checklist. Under the corrected rule the re-extraction matches
  MATLAB **exactly** — `Baseline_ON` 739/739, `Change_ON` 323/323, `Valve` 251/251, all at
  **0.0000 ms**. (⚠ The earlier "Laser 1003/1003" agreement row was **retracted**: there are 1004
  raw rises — 1002 real plus two 0.094 ms artefacts, **both mid-behaviour** — and no rule yields
  1003. Likewise "reproduces `NI_Sync.txt` to 0.000 ms" was retracted as a mathematical identity.)
- **The disposition therefore changes** from *"needs an owner decision"* to **"needs the new
  pipeline to extract NI from `nidq.bin` per this spec"**. Register entries 6, E4 and 8 stay
  `CONDITIONAL` only in the sense that they are conditional on that work being done, not on a
  choice being made.
- **⚠ Read the document in its audited form.** It was adversarially reviewed by six independent
  reviewers and **retracts several of its own first-pass claims**. Anything marked ❌ there must
  not be re-imported: the ≥15 ms sliver rule, the "~5 ms tick", "~4 frames", the
  surplus-`Change_ON` argument for `EVENT_VALID_OUTCOMES`, the laser-threshold explanation of the
  2025 gap, the 8σ detector's self-validation, and the "6× / 100 %-valid" encoder figures.
- **§0 is the process lesson, and it generalises beyond NI.** Two constants were laboriously
  *fitted* that were sitting in `Session/*_session_settings.json` all along, missed because the
  file was searched with keyword regexes that matched none of the actual names. **Dump the settings
  JSONs in full and read them before deriving anything.** The file also resolves several standing
  puzzles for free: `Torientationdelaymin/…meanadd = 6/2` explains the 6–11 s change latencies,
  `Trewdavailable = 2` explains the ~2000 ms second pulses on Miss trials, `punishearly` defines
  the `fa` outcome, and `changelist1/2` = [4, 2] / [1.5, 1.35, 1.25] *are* the change sizes
  (cross-check for register entry 12).
- **What genuinely remains open — and the spec says so itself.**
  1. **Per-session generalisation.** One session, one subject, nothing replicated. The channel map
     is *reported* byte-identical across all 50 BG_046 raw sessions, but every numeric constant —
     thresholds, the ~67 ms latency, the lick threshold, the wheel scale — is explicitly to be
     **re-derived per session**. The procedures transfer; the numbers do not.
  2. **BG_031 specifically, and the check has changed shape.** E4's mechanism is no longer "the
     0.383 V laser never crossed threshold" (retracted — at 0.150 V the line has 1,007 crossings);
     it is **structural**: `Valve_R` is a per-trial field and no laser pulse falls inside any
     trial. **Check:** on one BG_031 session lacking `Laser` (e.g. `BG_031_100325`), read ch7 from
     `nidq.bin`/`nidq.meta` at a derived threshold *and* establish whether that subject's
     extraction stores laser events per-trial. One session answers both.
  3. **Channel traps that must not be re-discovered.** `Baseline_ON` has a single excursion to the
     negative rail while resting near 0 V, so a min/max midpoint threshold sits *below* the
     resting level and recovers **0 of 739** pulses — use robust levels. The Laser peaks at
     **0.383 V**, so any assumed logic level finds zero. And **guard unconnected lines**: `Airpuff`
     swings 0.028 V and a naive level estimator put a threshold inside its own noise, emitting
     **17.4 million spurious edges** — require a minimum ~0.1 V swing before treating a line as
     TTL (that keeps `Laser` and rejects `Airpuff`).
  4. **Edge-pairing is unguarded in most code and one case is live.** `dig1`/`dig2` have **1 rise
     and 0 falls** — they go high and never return. Code that pairs `rise[i]` with `fall[i]`
     assumes the line starts LOW and every rise has a later fall; violate either and every event
     pairs with the wrong partner, producing negative widths **with no error raised**. Assert both.
  5. **`dig4` is NOT a duplicate of the analog `Baseline_ON`** (earlier claim retracted): only
     **663/761** rise indices are sample-identical and the 22 slivers on each line occur at
     completely different times (0 of 22 coincide). Its real value is being **threshold-free**,
     which is what adjudicated the merge rule — and that rule matters, because the old width cut
     placed **20 trial onsets 5.02 ms late** and the trial *count* cannot detect it (both rules
     return 739).
- **Cost** — one BG_031 session's `X:` read for item 2; fold into the Q5 sweep. Items 1 and 3–5
  are new-pipeline work, not audit work.
- **Blocks** — nothing that a decision would unblock. It scopes sub-project 1's NI extraction
  stage, which `visdetect.core.spikeglx` (cold, and **truly untested**) is the natural home for.

## Q7 — The BG_046 detection-composition drift figure has no measurement of its own

- **Register entry** — E2. The direction is clear and the mechanism is not in doubt, but the
  headline numbers — **broad/SPN 89 → 15 %** and "amplitude halving Jun → Jul" — are **inherited**
  from `docs/superpowers/specs/2026-08-05-new-repo-master-design.md:692-694` and the memory record
  `qc_celltype_yield_jun2026`. **No `d*` id backs them**; this audit did not re-measure them.
- **The check that settles it** — recompute, per session, the broad/narrow waveform fraction and
  the median spike amplitude over the BG_046 series, from
  `data/cache/.../waveform_celltype_labels` or directly from `templates.npy`, and plot against
  days-from-implant. Cheap, no `X:` access, and it converts a quoted figure into a citable one —
  which matters because ADR-018 makes the chronic-stability control a **contract row** and every
  such control will be sized against this number.
- **Blocks** — nothing immediately; it weakens every citation of the figure until done.

## Q8 — `um_ref` staleness sits inside the measurement's own timezone skew

- **Evidence** — `d4.stale.topics` = 7/7. The two sides of the staleness comparison use different
  timezone conventions: artefact mtimes are converted to **UTC** dates while `git log --format=%cs`
  yields the committer's **local** date. `um_ref` (writer 2026-07-02 vs artefact 2026-07-01) is
  the one committed verdict resting on a **one-day margin**, so it should be treated as
  **uncertain**, not as an established staleness finding. Six "current" verdicts sit on zero-day
  margins and are equally convention-sensitive (`dant`, `decision_latents`, `evidence_learning`,
  `neural_latents`, `preparatory_fig5`, `state_tags`).
- **The check that settles it** — re-run the comparison with both sides in the same timezone
  (`git log --date=iso-strict-local` or mtimes in local time), or — better — abandon mtime
  entirely and compare a **content hash of the writer at the artefact's recorded commit**, which
  is what the new repo's provenance sidecar makes possible.
- **Blocks** — nothing. Do not cite `um_ref` as stale without re-running.

## Q9 — Remote reality: every `origin/*` claim rests on a cached ref

- **Evidence** — `d7.local_only.commits` = 31 as-of 2026-08-13 15:12. `ssh-add -l` reports no
  agent and `git ls-remote origin` fails with `Permission denied (publickey)`, so **every
  `origin/*` statement in `branch-disposition.md` and D7 describes a remote-tracking ref last
  fetched 2026-08-06, not the remote.** A branch recorded as "0 local-only" is safe *if* the cache
  is truthful.
- **The check that settles it** — `ssh-add ~/.ssh/id_ed25519`, then `git fetch --all --tags` and
  a real `git ls-remote origin`; re-run `d7_work_at_risk.py` and
  `d7_work_at_risk_supplement.py` back-to-back. **Do this immediately before acting on any drop
  recommendation in `drop-list.md` or `branch-disposition.md`.**
- **Blocks** — the freeze decision (sub-project 6), and every branch drop.

## Q10 — Whether the ~155 GB of gitignored artefacts is regenerable

- **Evidence** — `d7.gitignored.volume` ≈ 155 GB over 16 tree entries; ~110 GB in the primary
  checkout. It is **sized, not provenance-checked**. The four hand-label sets are the known
  irreplaceable slice (`d7.handlabels.exposure`: 269 files / 31.0 MB, **220 untracked**), and
  their only backup sits on the **same physical disk** as the repo.
- **The check that settles it** — for regenerability, the D4 traceability census is the answer
  and it is discouraging (`d4.trace.untraceable_frac` = 0.42). For the hand labels the check is
  not a measurement but an action: **an off-disk copy**, which sub-project −1 requires and which
  does not exist yet.
- **Blocks** — sub-project 6 (deletion of the old artefact tree).

## Q11 — Residual measurement gaps, with their settling checks

Recorded so nothing falls between documents. None blocks sub-project 3.

| Gap | Id | Check that settles it |
|---|---|---|
| Full enumeration of import-time side effects | `d2.sideeffects.import` = not-measured | AST-walk every module for module-level calls; the known set is `matplotlib.use("Agg")` ×4 (`tf_pulse.py:17`, `unit_selection.py:23`, `core/qc.py:27`, `suite/plotting.py:9`) plus `os.makedirs` at `suite/config.py:19` |
| Real-data test tier runtime | `d5.tests.realdata_runtime_s` = not-measured | Run the 14 `needs_real_data=True` files listed in `test_partition.csv` once the pkl tree is stable |
| Pairwise agreement of the duplicated doc pairs | `d6.dup_pair_agreement` = not-measured | Deliberately deferred: ADR-005 deletes the copies. Only the one known divergence matters (register D3) |
| `partial_spearman` estimator spread measured on **one** input | `d3.pspearman.spread` = 0.892 / 0.901 / 0.901 | Max spread 0.0090 ≤ the 0.02 upgrade threshold, so the register entry was **not** upgraded. Re-run the three families on 2–3 further real inputs with different tie structure before consolidating; B and C are an exact algebraic identity, so only A-vs-B can move |
| Whether the two 21.99 GB eye-cam copies are one file on disk | — | `fsutil hardlink list` on both paths; not probed because it is write-adjacent tooling on worktree trees |
| Historical figure `dt` provenance | `d1.tfperiod.figure_attribution` | See Q3 — unrecoverable; superseded by ADR-019 |

## Q12 — Which optotagging laser block was SNr and which was GPe

- **Register entry** — A15. Direction is known and stark — **every D1/D2 pathway label from
  optotagging may be swapped** — but the polarity is not resolvable from anything in the repo.
- **The disagreement.** `split_laser_blocks` returns the blocks **in time order** and its caller
  binds them positionally as `(gpe_pulses, snr_pulses)`, i.e. **first block = GPe**
  (`src/visdetect/analysis/optotagging.py:505-510`, `:773`). The NI spec measured **first block =
  SNr** on BG_046 17092025 (`docs/raw_data/NIDAQ_AND_EVENT_SPEC.md` §10). Neither assumption is
  recorded anywhere authoritative: the spec **checked all six settings files** and the mapping is
  in none of them.
- **Why it is quarantined rather than asserted.** The spec establishes the mapping for *one*
  session from the experimenter's own account, not from a file. It is entirely possible the block
  order varies between sessions — which is precisely why the spec's recommendation is to capture
  the mapping as acquisition metadata. Declaring the code "wrong" on one session's testimony would
  substitute one undocumented assumption for another.
- **The check that settles it, in order of strength.**
  1. **Documentary** — recover the per-session fibre→structure assignment from the experimenter's
     records or lab notebook and write it into per-session metadata (ADR-019 already requires
     `subjects` and `acquisition` tables). This is the only check that settles it *correctly*, and
     it is not a compute task.
  2. **Anatomical** — the fibre tracks are localisable; `visdetect.anatomy` already does CCF
     localization for this subject. A fibre track terminating in SNr versus GPe is decisive and
     independent of the block order.
  3. **Corroborative only** — the response asymmetry (**17** block-0-only, 3 both, **0**
     block-1-only, with 199 untestable reported separately) is consistent with the repo's standing
     "3 collision-confirmed, all D1" claim under the spec's mapping and inconsistent under the
     code's. Suggestive; different sessions and different pipelines, so it cannot arbitrate alone.
     The asymmetry itself is not a detection-power artefact — responders are not higher-firing
     (p = 0.705) and 8,397 sham tests produced zero false positives.
- **A second, independent reason this matters more than a label.** The spec's audited verdict on
  the antidromic interpretation is **"plausible but NOT established"** (§10). ⚠ An earlier
  revision of this entry said "disfavoured", on two supports the current spec withdraws *for this
  protocol*: the negative collision test (short/long-gap P(evoked) ratio **1.197, 95 % CI
  [1.048, 1.367]**) is real but **invalid under a 10 ms sustained pulse** (continued illumination
  regenerates a spike even if the first is annihilated), and post-light firing is **expected**
  from ChR2's ~10 ms closure, not evidence against conduction. What stands either way: no
  responder shows the reliable single-spike sub-millisecond antidromic signature, and pathway
  assignment is not established for any individual unit. So even with the mapping settled, these
  are **candidates, not identified cell types**. Settling Q12 fixes *which structure the fibre
  was over*; it does not by itself license a D1/D2 call.
- **Until it is settled** — no claim naming D1 or D2 from optotagging is supportable, and the
  existing `data/cache/optotagging/*` label columns should be treated as *block index*, not as
  pathway. ADR-018 already makes `celltype_label_source` a required ledger field; this entry is
  why `optotag_candidate` must carry the block→target provenance with it.
- **Blocks** — every D1/D2 pathway claim derived from optotagging.

---

## What is **not** quarantined, and why

- **`d9.size_ratio`, `d9.readtimes`, `d9.roundtrip`, `d9.compression`** — `not-measured` **by
  decision, not by failure**. The owner pre-decided NWB and a raw rebuild on 2026-08-13, which
  voids the spike's comparative purpose and makes its pkl→NWB round-trip exercise a path the
  rebuild will never run. These are not open questions for the old repo; the new repo measures
  them against its own writer. The one thing that survives is register entry **A9** (assert the
  codec post-write), and it is a requirement, not a question.
- **The two live wrong claims in `docs/science`** (`QUESTION_INDEX.md:49`,
  `2026-06-17-post-tf-null-research-direction.md:4,48`) — **not quarantined, because they are not
  undetermined.** Both are known-wrong against a cited ground truth. They are recorded as register
  entries **D1** and **D2**, in a section explicitly outside the ADR-009 attribution base. The
  argument for that placement is given there.
- **`d4.ids.files_scanned`'s `table_output/` blind spot** — the census contributed 0 rows from
  `table_output/` because its only CSV keys on `Session_Date`, outside the brief's `ID_COLS`. An
  out-of-band probe classified all 6,679 of its tokens as clean `8digit`. The blind spot is real
  but is not hiding corruption today, so it is a caveat, not a quarantine.
