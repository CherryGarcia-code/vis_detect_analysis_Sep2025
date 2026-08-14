# Quarantine — undetermined, with the check that settles each

Companion to `known-defect-register.md`. Under ADR-009 a defect whose **direction** cannot be
stated attributes nothing, so it must be quarantined explicitly rather than ported on assumption —
every delta it produced would be unexplained by definition. This file lists everything the audit
could not determine, and, for each, **the specific check that settles it**.

Two entries the spec quarantined are **no longer here** — they were resolved and moved into the
register:

| Was quarantined | Resolved by | Now |
|---|---|---|
| `ref`-trial change-presented ambiguity | Task 4 (`d1.ref.with_change_time` = 18 = `d1.ref.total`, median RT +83 ms) | Register entry **11**, status SETTLED-CONVENTION |
| `CHANGE_SIZES` membership divergence | Task 15 per-consumer check (`d8.changesizes.catch_in_go_loops` = 0) | Register entry **12**, status SETTLED-CONVENTION |

Ordered by how much they block. **Q1–Q5 block sub-project 3 or sub-project 1; Q6–Q13 do not.**

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

## Q6 — Which NI events the re-ingest actually re-extracts (the hinge decision)

- **Not a measurement gap — an unmade decision**, and it is the one that flips four register
  entries (6, E4, 8 and, indirectly, A1). As the code stands, "re-ingest from raw" means
  **re-ingest from the MATLAB extraction output**: `build_session_from_raw` reads an
  already-extracted `*NIdaq_events.mat` (`ingest.py:444`) plus behavioural trials from `raw_dir`
  (`ingest.py:441`). It never opens `nidq.bin`.
- **Why it matters** — `lick_channels.py:5-8` records that the raw `nidq.meta` channel map is
  **byte-identical across all 50 BG_046 raw sessions** and always names the lines
  `Piezo_1`/`Piezo_2`. So the naming split, and plausibly BG_031's 35-of-43 missing `Laser`
  events, are artefacts of the **extraction step**, not of the recordings. Re-extracting NI from
  SpikeGLX dissolves them; re-running ingest against the existing `.mat` files reproduces them
  exactly.
- **The check that settles it** — on **one** BG_031 session that currently lacks `Laser`
  (e.g. `BG_031_100325`), extract the NI digital lines directly from `nidq.bin`/`nidq.meta` and
  check whether a laser pulse train is present. One session answers it. If present, the new
  ingest must own NI extraction; if absent, the gap is acquisition-side and E4 becomes permanent.
- **Cost** — one session's `X:` read; fold into the Q5 sweep.
- **Blocks** — the re-ingest disposition of register entries 6, 8 and E4.

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
