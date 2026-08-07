# Expert-Panel Review of the Master Design — Synthesis and Proposed Amendments

**Date:** 2026-08-07
**Status:** Awaiting project-owner decisions (per-cluster)
**Reviews:** six independent expert lenses (data standards · statistical rigor · reproducibility
engineering · ephys QC · publication readiness · adversarial critic), each grounded in web research
and the project's own literature syntheses, plus the assistant's independent pass.
**Raw panel output:** `2026-08-07-master-design-panel-raw.json` (all ~70 proposals with full
rationale and evidence URLs). Panel agents ran on Opus 5; synthesis by Fable 5.

---

## 1. Headline verdict

All six lenses independently reached the same overall judgement: **the engineering half of the
design is at or above what top-tier systems-neuroscience labs actually run** — ADR-004 (typed
identity registry), ADR-009 (explained-difference gate), ADR-011 (attributed decision log) and S5
(execution-log-authoritative provenance) were each singled out, by multiple lenses, as *better than
field practice*. Nothing in the existing fourteen ADRs needs to be reversed.

The panel found three structural blind spots, one process failure mode, and a set of overbuilt
gates to soften:

1. **The data layer is never specified.** Pickle/CSV/`.npy` are silently inherited. The word
   "pickle" does not appear in the design; CSV's dtype loss is the *mechanism* of the 15,802-row
   defect; the `RenamingUnpickler` proves the pickle–code coupling is already load-bearing.
2. **The design records state but never pins it.** Environment recorded per artifact, never locked;
   no RNG policy; no BLAS/thread policy; ADR-009 has no numerical-noise floor, so as written it
   blocks on the last ULP.
3. **No manuscript-facing layer.** "Figure panel", "sex", "n-table", "DANDI", "source data" appear
   nowhere; realized-n per artefact is retrofit-expensive (requires re-running everything).
4. **The gates are priced individually, never summed.** Measured against the live repo: ~70–100
   approval packets (not 20–40), and an estimated 3–5× ceremony tax on an exploratory analysis with
   no exploratory tier — the exact condition under which the old repo's conventions were routed
   around. Four lenses independently converged on the same fix: **gate on promotion-to-claim, not
   on file creation.**

---

## 2. Consensus overbuild corrections (soften/cut — no user decision needed unless you object)

| # | Current design element | Correction | Converging lenses |
|---|---|---|---|
| O1 | AST near-duplicate detection as a **commit gate** | Demote to a monthly/pre-milestone triage report. Keep the two zero-false-positive checks hard: must-import registry, same-name-different-behaviour. Type-3 clone detectors run 10–80 % precision; per-session analysis loops are near-identical *by construction* | Critic, QC |
| O2 | Null control for **every analysis** | Scope to analyses producing an inferential statistic that enters the ledger; add a declared `descriptive` class (recorded, auditable, ineligible as claim evidence) | Stats, QC, Publication |
| O3 | Synthetic recovery for **every new estimator** | Narrow to estimators with free parameters or latent structure (kernel width, DDM, HMM, decoders, CDs); one demonstration per estimator *family*; exclude standard library statistics | Critic, Publication |
| O4 | Traceability census over **all 3,056 figures** | Stratified sample (~100, per topic) gives the same headline fraction with a CI at a fraction of the cost | Data |
| O5 | Ledger `Verification` as free text | Enumerated booleans generated from a `visdetect.verify` module — free text reproduces the CLAUDE.md-vs-constants drift in miniature | Stats, Publication |
| O6 | Seed-literal registry (floated in review) | **Rejected** — registering magic numbers recreates the shadow-constant pathology; register *entropy per run in the provenance log*, ban literals in code | Repro |
| O7 | DataJoint / Spyglass / DVC / DataLad / containers / workflow engines | All considered and **deliberately rejected/deferred with recorded reasons** (added to ADR Alternatives sections so the decisions aren't relitigated) | Data, Repro |
| O8 | ADR-012's dormant loader-refusal guard | Keep the `cohort` field; replace the zero-caller guard with a **session-access log** (which sessions each run touched) — accumulates from day one and is what actually makes a future confirmatory split defensible | Critic |
| O9 | Ledger backfill over all historic claims | Forward-only, plus exactly two migrations: claims in the paper outline, and every retraction/refutation (~4 rows — the most valuable in the ledger) | Critic |
| O10 | S1 "one definition" with no variant mechanism | Add **registered named variants** (e.g. `CHANGE_SIZE_POOLS['tracking_qc_v1']`) — live code already contains a correct, deliberate divergence; a gate that can't express a correct exception gets routed around | QC |

---

## 3. Proposed additions, clustered for decision

Each cluster is one accept/modify/reject decision. **Recommendation: accept all eight** — priority
order below reflects retrofit cost (what becomes impossible or very expensive if deferred).

### Cluster A — The data layer (→ new ADR-015) · retrofit-expensive, decide first
*Source: Data lens (4 critical + 4 high), Publication lens, own pass*

- **NWB/HDF5 as the canonical session store** behind a `SessionStore` boundary
  (`load_session(SessionKey) → Session`), written via NeuroConv. Buys: schema versioning, lazy
  per-unit reads (kills the `del sess; gc.collect()` convention, which is a format artefact),
  compression (~30→10 GB), `nwbinspector` validation, DANDI eligibility — and format parity with
  the Allen Visual Behavior Neuropixels dataset, *the same change-detection paradigm*, as a public
  comparator.
- **Decided by measurement, not argument**: a one-day audit spike converts 3 real sessions
  (small / large / BG_012 colliding twin), measures size, read latency for the three real access
  patterns, and round-trip equality. Result goes in `measurements.csv`; ADR-015 cites numbers.
- **Parquet for every derived table the pipeline reads back; CSV is export-only.** Kills the
  leading-zero bug at the format level (dtype preserved on disk, not just in memory). Gate on
  `pd.read_csv` under the cache root. Same for bare `.npy`: derived arrays carry named dims,
  bin centres, unit ids.
- **`schema_version` on every artefact; loader refuses unknown versions; migrations are explicit.**
  Bans the current `None`-default pattern where "field absent" and "field null" are the same value
  (`trial_event_index`: "None = alignment not yet verified" is indistinguishable from "predates
  verification").
- **Identity stamped inside the artefact** (opaque `session_uid` + subject + schema version);
  registry becomes an index over stamped ids; path-vs-stamp disagreement is a hard error. Makes the
  `_b`/`_c` twin problem a data fact rather than a naming convention.
- **One canonical per-session Units table** (spike times + anatomy + waveform + tracking + labels
  as columns with `*_version`/`*_source` attrs; `UnitID` type joins ADR-004). Ends the
  `(session_id, cluster_id)` string-join architecture the 15,802-row defect ran on.
- MATLAB NI-extraction named for what it is: either read raw SpikeGLX nidq directly in Python
  (retires the lick-defect class permanently) or declare the `.mat` an external upstream artefact
  with recorded producer + hash.

### Cluster B — Pinning and determinism (→ new ADR-016) · cheap now, unfixable retroactively
*Source: Repro lens (2 critical), Publication lens, own pass*

- **Lockfiles**: `uv.lock` (analysis stack, Windows+Linux in one file) + `pixi.lock` (heavy
  KS4/CUDA/UnitMatch layer). Python ≥3.12 (3.10 EOLs Oct 2026); confirm with a trial lock (pyddm
  pin is the likely blocker). S5's "environment" becomes *lockfile hash*, not a version list.
  Measured motivation: `setup.cfg` (unbounded) vs `environment.yml` (`numpy<2`) vs live venv
  (numpy 2.2.6) — three divergent definitions of the environment today.
- **RNG policy**: ban `np.random.seed` and bare `np.random.*` by AST gate; every stochastic
  function takes `rng: np.random.Generator`; per-run entropy recorded in the sidecar;
  `rng.spawn()` for workers. (~396 seed literals, 35 distinct values today.)
- **Thread policy**: `threadpool_limits(1)` in workers; sidecar records BLAS name/version/threads.
- **`numerical-noise` becomes the fifth ADR-009 attribution**, with a *measured* per-analysis-class
  tolerance floor (run 5× across platforms/thread settings; spread = floor). Resolves master-design
  Open Question 4; abandons bit-identity, which is unattainable (DYNAMIC_ARCH OpenBLAS) and wrong.
- **External-tools table** in S5: KS4 commit, UnitMatch/DANT versions, TPrime build, MATLAB
  release, SpikeGLX version — the tools that actually determine spike times.

### Cluster C — Content hashing and staleness (→ new ADR-017, or ADR-004/011 extensions)
*Source: Data lens (critical), Repro lens (high) — independently converged*

- **Registry rows carry sha256 of every upstream input.** Refresh diffs hashes; a `changed-inputs`
  report must be acknowledged in the decision log before downstream artefacts are valid. This is
  the mechanism that would have caught the lick-channel re-extraction (content changed at unchanged
  paths — S5 as written records paths and would have been byte-identical before/after).
- **Content-addressed cache keys**: artefact key = hash(code version ‖ constants ‖ input digests ‖
  params ‖ lockfile). Loader refuses a stale key; `allow_stale=` is explicit and ledger-recorded.
  Replaces the audit's mtime heuristic; makes the `tf_responsive`-registries class of defect
  *unrepresentable*. ADR-014's null-currency check collapses into the same key comparison.
- **Time-base provenance (TPrime)**: session artefact carries a `time_base` block (file used,
  TPrime build, residual stats); ingest **fails closed** rather than silently falling back to
  uncorrected spike times (today: a 3-branch fallback logged at INFO, invisible downstream);
  `time_base="uncorrected"` requires an explicit flag and the loader refuses it without declared
  intent — same guard pattern as ADR-012.

### Cluster D — QC becomes first-class and non-destructive (→ new ADR-018) · scientific validity
*Source: QC lens (3 critical + 5 high)*

- **Stop applying QC destructively at ingest.** Pkls currently store spikes only for
  `good_and_stable` units, so every future analysis is a subset of one unnamed 2025 decision and
  no profile can ever be *more* permissive — which quietly falsifies ADR-011 for the most
  consequential decision in the pipeline. New rule: ingest stores all KS-good units + a per-unit
  metric panel; QC is applied at analysis time as a **view**.
- **Named, hashed, versioned QC profiles** (~4: `sorting_quality`, `striatal_default`,
  `striatal_strict`, `tracking_eligible`) instead of one canonical floor — the four disagreeing
  floors are *different questions*, not four copies of one number; the defect was namelessness.
  Ledger + sidecar record profile id+hash. Changing a profile = decision-log entry.
- **Metric panel computed once at ingest** using field-standard definitions
  (SpikeInterface/Bombcell; keep the Khilkevich stability statistic as a named metric). Profiles
  threshold stored columns → re-filter in seconds, and `confirmed` claims are re-run under a second
  named profile (promotion precondition).
- **Fail closed on unknown metrics** (tri-state pass/fail/unknown; unknown fails by default) —
  `fillna(0.0)` passing the contamination gate is the `load_qc_profile() → {}` failure class again.
- **Chronic-stability control as a contract row** for every across-session claim (covariate /
  composition-matching / tracked-subset / within-window, from a named menu; ledger field where
  `none` is legal but visible). Motivation is *measured*, not cautionary: broad/SPN 89→15 % at the
  KS4 detection level, amplitude halving Jun→Jul, and the behavioural gate excludes 5 of 6 SPN-rich
  June sessions — learning stage and recording epoch are collinear **by construction**.
- **Track-QC becomes a registry table** (per-link scores, ISI-fingerprint checks, consensus flag,
  named per-subject-calibrated profile; hand verdicts move to the decision log). Per-subject
  calibration by within-day split-half sorts and (if multi-shank) across-shank negatives, with
  UnitMatch/DANT published error rates as reference points. "Tracked unit" becomes a defined,
  versioned thing.
- **Cell-type label provenance + confidence as required ledger fields**, with the hard rule: a
  claim naming D1/D2 either cites `optotag_collision_confirmed` units or says "putative" with the
  source named. (3 collision-confirmed units, all D1, zero D2; waveform AUC 0.65 is below the ~75 %
  the field accepts for the *easier* narrow/broad contrast.)
- Session-level recording-quality covariates (days-from-implant, yield, RMS, amplitude,
  narrow-fraction, drift, channel-map hash) in the registry, IBL-RIGOR-style.
- Five ephys entries added to the known-defect register with directions (incl. BG_031's Laser-event
  extraction gap: 35/43 sessions missing — a data-completeness defect that looks like a biological
  result).

### Cluster E — Statistics tightened to be self-consistent (fold into ADR-010/013/014)
*Source: Stats lens (3 critical + 7 high)*

- **Name the hierarchy.** Session = default random effect; **subject is never a random effect at
  k ≤ 5** (fixed effect or stratification) — as written, `groups=subject` at k=3 satisfies the gate
  and returns pooled inference wearing a rigorous hat. Require the mixedlm/cluster-robust pair with
  the convergence flag recorded (already lab practice in `harden-result`; must live in the ADR).
- **`per_subject` estimates + `n_subjects_replicating` + `scope_of_inference` as required ledger
  fields**, gating promotion to `confirmed`. The claim's wording must be derivable from scope
  ("striatal neurons do X" is not writable from a single-subject row). Project has measured
  Simpson's-paradox instances in both directions.
- **`inconclusive` status + TOST equivalence** against the pre-registered MIE for any claimed null.
  The project's best outputs are honest negatives; the ledger currently can't distinguish "we
  exclude d > 0.2" from "we had no power".
- **Model recovery, not just parameter recovery** (confusion matrix across candidate models) where
  a claim rests on model comparison — which is where the headlines live (drift vs threshold; HMM
  K; dimensionality). Recovery at realistic n and stimulus statistics; scatter per parameter, not
  a single r.
- **MIE derived from the estimator's measured resolution floor** (smallest recoverable effect at
  realistic n), with FR-invariance swept for the kernel-width estimator — this check, at
  estimator-write time, is precisely what would have pre-empted the retracted raw-Hz result.
- **Null controls specified**: dependency-respecting surrogate (shuffle at the level of the claim),
  quantitative flatness criterion, permutation floor (no p < 1/(n+1)).
- **Bounded specification curve on `confirmed` claims only**, made cheap by requiring declared
  analytic DoFs to be kwargs defaulting to registry constants (the sub-project 2 gate already bans
  inline literals — same check, new purpose). Session-ordering axis is the project's worked
  example.
- **`prespec_sha` with mechanical ancestry check** (pre-spec commit must precede the analysis
  commit); claims failing it are capped at `exploratory`. Applies ADR-003's own philosophy to the
  statistical layer.
- **Split-half stability field** (interleaved, never chronological — chronological is confounded
  with the learning axis).
- **Canonical effects module** (`visdetect.stats.effects`) in the must-import registry: named
  effect size per test class, computed at the inference level; ΔBIC banned as an effect size.
- **`visdetect.verify` lands in sub-project 2, not 5.** The harden-result battery is currently
  prose in the AI layer, built *last* — so the analysis port would run without the lab's strongest
  instrument. Reimplement as library functions returning structured records; the skill becomes a
  thin pointer; the ledger's Verification field is generated from the records.
- **Sanctioned constants override** (`with constants.override(min_fr=0.5):`) importable only from
  the sweep harness and tests, recorded in the sidecar — without it, S1 and sensitivity analysis
  fight, and people retype literals (the exact old pathology).

### Cluster F — The publication layer (→ new ADR-019) · cheap now, a scramble at submission
*Source: Publication lens (2 critical + 5 high)*

- **`n_table` in every provenance sidecar** (n subjects/sessions/units/trials, per condition, per
  panel; clustering variable) — gate-enforced. The single highest-value addition: journals demand
  exact n per group per panel, and the only way to add it later is to re-run everything.
- **Registry gains `subjects` and `acquisition` tables**: species, strain, genotype, **sex**, DOB,
  ethics/licence number, surgery date, implant coordinates, hemisphere; per-session probe serial,
  IMRO map, rig, sorter+params, training stage. The word "sex" currently appears nowhere in the
  design; "hemisphere = LEFT" lives in a memory note rather than the data.
- **Manuscript panels are typed artefacts** (`F3b`, `ED2a`) with a manifest binding panel → 
  component + commit + registry snapshot + environment + n_table + ledger rows; one-command
  `repro F3b`; CI smoke on two panels. Kills `fig3_v2_final_FINAL.png` the way ADR-004 killed
  session-id strings.
- **`source_data.csv` beside every figure** (the plotted values, tidy) — near-free at write time;
  satisfies source-data requests; lets ADR-009 compare *numbers* not pixels.
- **Human layer in sub-project 5**: generated README/docs from the same source as CLAUDE.md, plus
  three hand-written pages (30-min walkthrough; task/vocabulary glossary incl. the `fa` ≠ SDT-FA
  trap; data-provenance map). The approval packets are retained as a browsable corpus — they *are*
  the onboarding manual and the Methods first draft.
- **Release hygiene**: CITATION.cff + ORCID; explicit **data** licence decision (MIT covers code
  only; DANDI needs CC0/CC-BY); `paper-freeze/<name>` tags with Zenodo DOIs; generated ARRIVE-E10 +
  Reporting Summary drafts from registry+ledger (blank fields become explicit N/A disclosures; the
  blinded session sorter gets reported as the genuine blinding strength it is).
- **NWB export contract** proven now by one NeuroConv round-trip test in CI (`nwbinspector
  --config dandi`, zero CRITICAL) — deposit becomes a metadata exercise, not a conversion project.
- **Rule: no figure produced by the old repo may enter the manuscript.** Old figures are archive;
  every panel regenerates under the manifest.
- Ledger columns: `prespec_commit`, `experimental_unit` (mouse/session/unit — ARRIVE Item 1),
  `figure_panels` (a retraction immediately names the panels it invalidates).

### Cluster G — Process reality (→ new ADR-020 + edits to 007/003/§7/§9) · the critic's case
*Source: Critic (5 critical + 5 high); convergent echoes from three other lenses*

- **Two-tier contract: explore vs claim.** Tier 1 (free via scaffold): registry sessions, canonical
  constants, artefact API, layer, index. Tier 2 fires **at promotion** — the moment output is cited
  in a ledger row/figure/results doc: null, recovery, pseudoreplication, reuse statement, test,
  packet. Enforced by `ledger add` refusing rows without current Tier-2 artefacts. Rationale:
  measured Tier-2 overhead ~8–25 h vs 3–6 h to write the analysis; ~80 % of exploratory analyses
  die without a claim; and the scratch exemption as written is the loophole all 378 old scripts
  would have passed through. Gating at promotion also fires *at the moment of excitement* — ADR-014's
  own stated failure window.
- **Walking-skeleton milestone 0.5** (before the Stage-2 spec session): one real day-1–9 session +
  one BG_012 colliding twin, end-to-end: registry → typed key → constants → load → PSTH → figure
  with sidecar → ledger row → one gate. Time-boxed 3–5 days. Specs 1–6 must cite its measured
  ergonomics. (Tests the two riskiest unprecedented decisions — typed-ID ergonomics and the
  sidecar — which the audit of the *old* repo cannot test.)
- **Gate-tier table in ADR-003**: (1) pre-commit, source-only, seconds; (2) CI, source-only,
  minutes; (3) verification runs, data-dependent, local/Slurm, pre-milestone/pre-ledger — never
  per-commit, advisory-with-report. Plus the Repro lens's concrete CI topology: a committed
  **golden mini-session** (~20 units × 200 trials, includes a day-1–9 date and a suffixed twin)
  making most data-tests cloud-runnable; a commit-pinned local **receipt** for full-data runs
  (merge fails without a receipt matching HEAD); nightly Slurm tier; **no self-hosted runner on the
  data box** (public-repo fork-PR code execution).
- **Packet re-scope**: measured count is 70–100 at module granularity, not 20–40. Bundle into
  module groups / gate families (~15–25 packets); machine-generate four of six sections
  (provenance, blast radius, executed output, delta table); human writes "what it is" + decision.
  Add a **seventh, adversarial section**: "strongest objection found, and its resolution",
  produced by an independent pass before the packet reaches you — you arbitrate a disagreement
  rather than assent to a proposal.
- **One sanctioned, logged override** (`# gate-override: <rule-id> reason="..."` → append-only
  tracked log, counted per milestone) + deny `--no-verify` at the harness level. Every gate failure
  message names the rule, the ADR, the historical defect it prevents, and the override syntax.
- **Time-box + stop-loss + science-continues policy** (§9 has no schedule risk): budget per
  sub-project, a named stop-loss date, pre-agreed fallback (collapse to walking-skeleton scope +
  lazy porting), and an explicit statement of where live science continues meanwhile.
- **Port sunset criterion**: the port is done when the modules regenerating the **paper's named
  figure set** have landed; everything else → `drop-list.md` or `cold-list.md` (ported lazily on
  first use). Target v1 budget (~≤12 modules) written into the ADR.
- **Old-repo freeze gets a per-branch disposition table** (sub-project 0 deliverable): 63 unmerged
  commits live on 5 branches today; QC1 repair is simultaneously a defect-register entry and
  in-flight code — decide merge/port/abandon per branch, no new old-repo branches after freeze.
- **Fix the dependency bug**: sub-project 2's gates need the artefact/provenance API that lives in
  sub-project 4. Move the minimal sidecar writer + artefact-path API into sub-project 1.
- **FIGURES/cache disposition stated**: old tree becomes a read-only archive root; tracked
  deliverables migrate; untracked figures deleted only after submission.
- ADR-005: generated content isolated in `CLAUDE.generated.md`; CI failure names the one-command
  fix; the *prose* half gets the dead-path check (the old CLAUDE.md's 18 %-dead problem was prose,
  which a generator does not solve).

### Cluster H — Backup becomes a policy (amend ADR-008 + new success criterion)
*Source: Repro lens (critical), own pass*

- ADR-008's "off-disk safety is the remote's job" is true for code and **false for everything
  else** — the remote holds none of the 30.5 GB pkls, 15.9 GB caches, registry, or hand-labels.
  Steady state today: 1 copy, 1 medium, 0 offsite.
- Policy: the four irreplaceable hand-made files (~27 KB) enter git **now** (`git add -f`);
  registry + decision log + hash manifest sync daily to institutional storage (`rclone --checksum`,
  I/O-only carve-out from the Samba rule, off-hours); caches/figures weekly; **quarterly restore
  test** against the manifest. Manifest-of-hashes committed to git (rejecting DVC/DataLad at this
  scale, in writing).

---

## 4. Questions only the project owner can answer

1. **The paper's figure list** (even a rough one). It is the port's sunset criterion; without it
   sub-project 3 is unbounded.
2. **Repo visibility** — public or private? Decides CI options and whether the golden mini-session
   can be committed.
3. **Is the cohort single-sex?** Reporting Summary forces a positive disclosure; if mixed, sex
   becomes a registry field and covariate consideration.
4. **Does the project licence permit open data deposit**, and what is the licence number to record?
   Five minutes now; a submission blocker later.
5. **Are implant dates recorded for all six subjects?** Cheapest chronic-drift covariate; decides
   whether days-from-implant or session-index is the standing control.
6. **Deposit scope at publication**: processed NWB (tens of GB — what DANDI is built for) or also
   raw SpikeGLX (TB-scale)?
7. **Session-ordering axis**: treated as a declared multiverse DoF (slopes reported as a
   distribution) or settled once in the decision log? Different papers result.
8. **Claim that replicates in one subject and reverses in another**: recordable as `confirmed`
   with a scoped wording, or capped at `exploratory`? One rule, decided now.

---

## 5. Proposed acceptance path

If you accept the recommendations wholesale: A→H become ADR-015 … ADR-020 plus amendments to
ADR-003/005/007/008/010/012/013/014 and §3/§7/§9, and the §2 overbuild corrections are folded in.
The audit spec gains the storage spike (A) and the branch-disposition deliverable (G). The
walking-skeleton milestone (G) slots between sub-project 0 and the Stage-2 spec session.

Recommended decision order: **A and B first** (retrofit-expensive), then **G** (it reshapes how all
subsequent work is executed), then D–F (spec-level), then C, H (mechanical).
