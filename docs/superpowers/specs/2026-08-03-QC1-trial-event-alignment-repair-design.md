# QC1 — Trial↔event realignment: diagnosis and repair (DESIGN)

**Date:** 2026-08-03 · **Status:** design, approved for planning · **ID:** QC1
**Supersedes the diagnostic guesses in:** `docs/superpowers/specs/2026-08-03-QC1-trial-baselineon-realignment-handoff.md`
(that document's *scope* section still stands; its *cause* hypotheses are corrected below).

---

## 0. Standing rules

- **X: is READ-ONLY for this task** (carve-out granted 2026-08-03). Read raw/processed session
  files to diagnose; copy bytes local before parsing. Never write/move/rename/delete on X:.
  Never run pipelines or heavy compute *over* the share.
- Multiple git worktrees are live. Verify the branch before any git op.
- `py` not `python`; venv `.venv\Scripts\python.exe`.
- New work → `scripts/<topic>/`, `data/cache/<topic>/`.

---

## 1. Root cause — ONE bug with two signs

`load_behavioral_trials` (`src/visdetect/core/ingest.py:72-98`) concatenates **every** `*trials.json`
found in a recording's `Session/` directory, in filename order. `build_session_from_raw` then
index-aligns that concatenated trial table to the recording's per-trial NI event arrays. **Nothing
verifies that the JSONs present belong to that recording.**

A recording day may contain several behavioural runs, each writing its own timestamped
`<SUBJ>_<YYYYMMDD>_<HHMMSS>__trials.json`. Those files are filed into session directories by hand,
and short/aborted runs are **curated into subfolders** (`Session/delete/`, `Session/partial/`). The
glob is **non-recursive**, so subfoldered runs are invisible to the converter — while the ephys, if
it was recording throughout, still contains their events.

The trial table and the event arrays therefore drift apart by **two distinct mechanisms**:

| Sign | Mechanism | Result |
|---|---|---|
| **A** — `n_trials > n_events` | Sibling JSONs from more runs than the recording covers sit in `Session/` root and are **concatenated** | Trial table spans runs; only a contiguous block belongs to this recording |
| **B** — `n_trials < n_events` | Trial table is **correct**; the *ephys* is untrimmed, spanning earlier curated runs whose JSONs live in `delete/`/`partial/` and are skipped by the non-recursive glob | Events are all real and all belong to the day; the trials match a **later offset** into them |

The common thread — and the thing the repair fixes — is that **nothing verifies the loaded trial
table corresponds to the recording's event arrays**. But the two signs need *different* converter
fixes (§4): sign A needs run *selection*; sign B has nothing to select.

### Evidence — BG_046 `05092025_b` (sign A)

`Raw data/BG_046_05092025/Session/` is **empty**; both of that day's runs are filed under
`BG_046_05092025_b/Session/`:

| run | trials |
|---|---|
| `BG_046_20250905_104819__trials.json` | 281 |
| `BG_046_20250905_115246__trials.json` | 248 |
| **concatenated → pkl** | **529** |
| `_b` ephys `Baseline_ON` | **248** |

The `_b` recording covers run 2 only. Trials `[281:529]` are its true trial table.

### Evidence — BG_046 `20082025` (sign B)

Not spurious events. `Session/` root holds a **single** JSON with exactly **486** trials — so no
concatenation occurred here — while the ephys holds **714** Baseline_ON. The missing 228 are
accounted for **exactly** by seven curated earlier runs whose JSONs are on disk in subfolders:

| location | runs | trials |
|---|---|---|
| `Session/delete/` (aborted) | 4 | 5 + 7 + 2 + 7 = **21** |
| `Session/partial/` | 3 | 150 + 18 + 39 = **207** |
| **curated total** | 7 | **228** |
| `Session/` root (the filed run) | 1 | **486** |
| | | 228 + 486 = **714** = `len(Baseline_ON)` ✓ |

The mouse ran ~8 blocks that morning; the first seven were curated out of the analysis set, but the
probe kept recording through them, so their events remain in the NI arrays. The trials align to
`BON[228:714]` — an **offset**, not a truncation.

> **Corrected 2026-08-04 by adversarial review (Lens 6), verified directly.** The original claim —
> "run 1's JSON was never filed, the recording spans two behavioural runs" — was **wrong** on both
> counts. The JSONs *were* filed, in curated subfolders; and the pre-228 block is itself multi-run
> (the `Baseline_ON` train has **two** large gaps: 76.8 s at index 168 and 316.3 s at index 227, so
> "one gap = one run boundary" is not a safe heuristic — see §4).

### Prior art — the defect originates in the MATLAB pipeline

The legacy pipeline (`scripts/conversion/matlab_scripts/NPX-analysis-master/`) did **not** solve this
problem; it is where the problem comes from.

- `load/loadSessionBehav.m:6-46` performs the **identical** blind concatenation of every
  `*trials.json` in the `Session/` dir. The Python `load_behavioral_trials` is a faithful port.
  **This is not a Python regression** — MATLAB-derived pkls carry the same defect, so no era of the
  pipeline is clean.
- `load/loadSessionNPX_main.m:99-101` attaches `behav_data` and `NI_events` side by side with no
  alignment check of any kind.
- Downstream code assumes strict index correspondence: `analysis/showPSTHforAllUnits.m:45` sets
  `trials_numb = length(Baseline_ON_times)` and then indexes `TrialsData(tr)`, and
  `AbortTimes = Baseline_ON_times + ReactionTimesAbort` adds an **event** vector to a **trial**
  vector element-wise. On the affected sessions this silently pairs the wrong trials.
  **MATLAB-era neural results on these sessions are invalid for the same reason.**

**Ordering divergence to check.** MATLAB sorts runs by file **mtime** (`[fname.datenum]`); Python
sorts by **filename**. Filenames embed the run timestamp, so the two normally agree — but these
`Session/` directories were touched by later reorganisation passes (observed mtimes of Oct 2025 and
Mar 2026), so mtime order can diverge from true run order. Where both a MATLAB-derived and a
Python-derived pkl exist for one session, they may concatenate runs in **different orders**. The
repair must not assume the two agree.

**Precedent worth keeping.** `analysis/showPSTHforAllUnits.m:29-43` cross-checks NI `Change_ON`
against the per-trial frame times at a **0.05 s tolerance**, and on failure *refuses to substitute*,
printing the discrepancy rather than quietly correcting. That is independent corroboration of the
50 ms threshold chosen in §3, and the same refuse-don't-silently-fix stance this design takes.

*Caveat, tested:* that check compares two **NI-side** quantities, so it validates NI extraction, not
trial↔event pairing. `frame_times_tr` was evaluated as a possible second independent discriminator
and **rejected**: per-trial frame counts vary widely (6–1122) but correlate only r ≈ 0.105 with
`change_time` even on a known-good session, because trials end at lick/abort rather than at change.
The independent second check the design needed came instead from the **NaN pattern** of `Change_ON`
— see §2, Check 1.

### Correction to the handoff document

- "Mode B — cause unknown, possibly spurious NI events" is **wrong**. The events are genuine and
  belong to the day — they are earlier curated runs the ephys recorded through. No event *filtering*
  is needed; the repair is an offset.
- Modes A and B share a **consequence** (trial *i* ≠ event *i*) and a **repair** (the index map),
  but they are **different mechanisms** and need different converter fixes — see the table above
  and §4.
- `BG_046_05092025` cannot be "recovered" as a neural session: its Raw *and* Processed trees contain
  **zero files** (no `.ap.bin`, no `Nidaq`, no Kilosort output). It is an empty placeholder. Run 1's
  281 trials are behaviour-only and have no ephys anywhere.
- The audit's benign band holds **48** sessions (44 positive, **4 negative**), not 44.

---

## 2. The alignment primitive

`Baseline_ON`, `Change_ON` and `Valve_L` are **per-trial arrays** — equal length, one entry per
recorded trial. Verified on **all 17** affected non-BG_012 sessions across BG_031/038/039/041/046:
`len(Change_ON) == len(Baseline_ON) == len(Valve_L)` without exception.

Alignment is tested by **two complementary checks, both required**.

### Check 1 (primary — full trial coverage): outcome ↔ change-presence

`change_time` is the **scheduled** change time, drawn at trial start. On `fa` and `abort` trials the
mouse licks, or the trial dies, before it arrives — the change is **never presented**. (This is
precisely why `EVENT_VALID_OUTCOMES` restricts `Change_ON` to hit/miss.) `Change_ON` is `NaN` on
exactly those trials, and finite exactly when the outcome is `Hit`, `Miss` or `Ref` — verified on
`19082025`: 255 finite = 186 Hit + 65 Miss + 4 Ref, with **zero** non-NaN placeholders (the `<= 0`
count is 0 on every session checked, so `isfinite` is a safe test).

```
agreement = mean( isfinite(Change_ON[j]) == (outcome_i in {Hit, Miss, Ref}) )
```

A categorical fingerprint over **100 % of trials**, immune to the scheduled-vs-realised problem by
construction: it asks whether the change *occurred*, which is exactly what the outcome label encodes.

| session | candidate | agreement |
|---|---|---|
| `19082025` (known good) | identity | **100.00 %** (n=587) |
| `20082025` | offset 228 | **100.00 %** (n=486) |
| `20082025` | offset 0 (current, null) | 51.44 % |
| `05092025_b` | `trials[281:529]` | **100.00 %** (n=248) |
| `05092025_b` | `trials[0:248]` (null) | 50.40 % |

**Acceptance: 100 %, no tolerance** — see the sensitivity scan below.

⚠ **The outcome set is CASE-SENSITIVE and includes `Ref`.** Real pkl labels are capitalised
(`Hit`, `Miss`, `FA`, `abort`, `Ref`), but the canonical `EVENT_VALID_OUTCOMES['Change_ON']` is
lowercase `{'hit','miss'}` **without** `Ref`. The implementation must hardcode `{Hit, Miss, Ref}`
and must **not** be refactored to reuse `EVENT_VALID_OUTCOMES`, or Check 1 silently breaks. (`Ref`
trials empirically carry a finite `Change_ON` on every subject — the change *was* presented — so the
canonical constant is arguably wrong to exclude them; that is a separate issue, not fixed here.)

⚠ **The ~50 % wrong-offset floor is outcome-balance-dependent.** It is ≈50 % only when Hit/Miss/Ref
and FA/abort are roughly balanced. On high-FA/abort protocols the wrong-offset baseline rises
(toward ~88 % on the BG_012 protocols), narrowing the gap. Check 1 stays a valid *accept-at-100 %*
test, but for high-impulsivity sessions **Check 2 carries the discrimination** — report both.

### Check 2 (secondary — timing precision): scheduled-change residual

```
residual_i = (Change_ON[j] − Baseline_ON[j]) − trials[i].change_time
score      = median |residual|   over trials whose change was ACTUALLY presented
                                 (Change_ON finite — only ~45 % of trials)
```

| session | candidate | median abs residual | within 50 ms | n |
|---|---|---|---|---|
| `19082025` (known good) | identity | **0.0051 s** | 100.0 % | 255/587 |
| `20082025` | `BON[228:]` | **0.0051 s** | 100.0 % | 215/486 |
| `20082025` | `BON[0:]` (null) | 1.4052 s | 0.4 % | 263 |
| `05092025_b` | `trials[281:529]` | **0.0051 s** | 100.0 % | 126/248 |
| `05092025_b` | `trials[0:248]` (null) | 1.5302 s | 1.6 % | 126 |

This check can **only** use trials whose scheduled change was realised; the rest are `NaN` and drop
out. That is why its `n` is roughly half the trial count — and why Check 1 is the primary: Check 1
covers the trials Check 2 cannot see.

⚠ **`n = 0` must REJECT, never vacuously pass.** `np.median([])` is `nan`, and an implementation
that reads `nan < 0.05` as "not applicable → pass" would let the solver accept an all-`FA`/`abort`
candidate slice paired against an all-`NaN` `Change_ON` region — Check 1 would read 100 % on a
degenerate comparison. Require a **minimum of 20 finite-change trials** for Check 2 to be
considered evaluated; below that the candidate is **rejected**, not excused. For a design whose
thesis is "refuse rather than silently accept", this is load-bearing.

### Sensitivity — both checks are knife-edge

Offsets scanned around the correct one for `20082025`:

| shift | categorical | median abs residual |
|---|---|---|
| −3 | 53.09 % | 1.3052 s |
| −2 | 58.64 % | 1.3301 s |
| −1 | 52.06 % | 1.1948 s |
| **0** | **100.00 %** | **0.0051 s** |

A **single-trial** offset is caught by both, with no gradual degradation to misjudge. Anything below
100 % on Check 1 is a misalignment, not noise.

Together these supersede count-matching, which is only a proxy: counts cannot distinguish a benign
`+3` end-of-session artifact from a genuine 3-event offset. After this work, `neural_safe` becomes a
*measured* property rather than an inferred one — including for the 48 benign sessions, which are
currently assumed fine rather than shown to be.

**Rejected alternatives.** (i) Reconstructing inter-trial intervals from `change_time + ITI` is
underpowered: the known-good control scores only Spearman r = +0.097. (ii) `stim_vbl` (per-frame
stimulus timestamps) would be ideal but is `None` in these pkls. (iii) `frame_times_tr` frame counts
— see §1.

---

## 3. Repair representation — an index map, never truncation

Truncating `05092025_b`'s trial table to its 248 ephys-backed trials would **delete 281 valid
behavioural trials**. Those trials have no ephys, but their behaviour is real and is already used by
the early-lick, hazard and state-tagging work — which §2 of the handoff correctly says is unaffected
by this bug. Repair must not damage behaviour to fix neural alignment.

Add one field to `Session`:

```python
trial_event_index: Optional[np.ndarray] = None   # int array, length n_trials
                                                 # value = index into the per-trial ni_events arrays
                                                 # -1    = this trial has no corresponding ephys event
                                                 # None  = not yet verified (see below)
```

- Neural code indexes events through this map (see §4 for the **full** consumer list — it is larger
  than `align.py`).
- Behaviour code ignores it entirely and keeps the full trial table.

⚠ **The default must be a plain `None`.** Empirically tested against a pickle round-trip of an
old-style `Session`: `field(default_factory=lambda: np.array([]))` leaves the key out of `__dict__`,
so `session.trial_event_index` raises `AttributeError` on every existing pkl; and a field with **no**
default, placed after `Session`'s all-defaulted fields, raises `TypeError` at class-definition time
(the module will not import). Consumers should still use
`getattr(session, "trial_event_index", None)` for safety.

⚠ **The "absent field + matching counts → identity map" fallback is PROVISIONAL, not verified.**
It leans on exactly the count proxy §2 declares insufficient. It is *mitigated*, not sound: (a) Phase 2
re-verifies all 253 pkls with the measured checks, and (b) empirically every one of the 23 misaligned
sessions has `diff ≠ 0`, so no `diff == 0` misalignment exists in the current data. Until Phase 2
completes, `neural_safe` must not be reported as *verified* on the strength of count-matching alone.
Note also that the 48 benign sessions have `diff ∈ [1,9]` — **non**-matching counts — so the identity
rule does not even apply to them; they currently fall through to the min-length truncation pairing in
`align.get_event_times_by_trial` (`out[:m] = arr[:m]`).

### Decisions taken

| Decision | Choice |
|---|---|
| Where the repair lands | **Repair the pkls** (backup first) **and patch the converter**, so future conversions cannot reintroduce it |
| Unsolvable sessions | **`trial_event_index = -1` throughout**: neural code hard-skips, behaviour keeps the trial table. Explicit and auditable; nothing deleted |
| Acceptance threshold | **Both** checks must pass: Check 1 categorical agreement **= 100 %** (no tolerance), **and** Check 2 median \|residual\| **< 0.05 s** — 10× above the observed aligned value, 28× below the misaligned one, sitting in a wide empty gap. Log both scores **and the runner-up candidate** for every session so the margin is visible, never assumed |

---

## 4. Components

| Component | Responsibility |
|---|---|
| `src/visdetect/core/run_alignment.py` | Pure, unit-testable. `per_trial_event_keys(ni_events)`, `outcome_change_agreement(...)` (Check 1), `alignment_residual(...)` (Check 2), `solve_alignment(trials, ni_events) -> Alignment \| None`. Solver ranks candidates on Check 1 first (full coverage), breaking ties on Check 2; returns best **and** runner-up scores for both. |
| `scripts/QC_technical/repair_trial_event_alignment.py` | Per pkl: solve → verify → back up → write `trial_event_index`. Emits `data/cache/qc_alignment/alignment_repair_report.csv`. Backup goes to `data/pkls/<SUBJ>/qc1_backup/<file>.bak_<UTC-stamp>` — written **before** any mutation, and the repair aborts if the backup cannot be created. Re-running is idempotent: an already-repaired pkl re-solves to the same map and is not re-backed-up. |
| `src/visdetect/core/ingest.py` (patch) | **Sign A only:** select the run JSON(s) matching the recording instead of blind-concatenating. Run both checks before emitting a pkl and refuse/flag on failure. Order runs by the **timestamp embedded in the filename**, never by mtime (§1). ⚠ **Do NOT make the glob recursive** — see below. |
| `src/visdetect/analysis/tf_glm_data.py` (patch) | **Must be patched individually.** `:508-542` reads `ni_events['Baseline_ON'/'Change_ON'/'Valve_L']` directly and pairs positionally (`bon[i]`, `con[i]`, `valve[i]`), importing no `align` helper. This is the validated TF-encoding GLM already run on all three mice — it stays silently wrong on all 17 affected sessions unless fixed here. |
| `scripts/QC_technical/audit_trial_baselineon_alignment.py` (extend) | Add `outcome_agreement`, `median_resid_s`, `runner_up_*`, `aligned`. `neural_safe` becomes evidence-based (both checks), with the count check retained only as a secondary signal. |
| `src/visdetect/analysis/align.py` + tensor builders | Honour `trial_event_index`; drop `-1` trials from event-aligned analyses. Consumers routed through `get_event_times[_by_trial]` (`utils.build_population_tensor`, `su_analysis.py` ×8, `hmm_downstream.py`, `tf_pulse.py`, `unit_selection.py`) are covered automatically **once those two functions remap `i → trial_event_index[i]`**. Also short-circuit a `-1` trial *before* the `change_time` NaN-fill in `get_event_times_by_trial` and `compute_true_reaction_time`, or the hard-skip guarantee is violated. |
| **Consumer audit (work item)** | ~120 files touch `ni_events`. `tf_glm_data.py` was found by review, not by search — a systematic audit for **direct positional** `ni_events` readers is required before Phase 4 closes, and each one either routed through `align` or patched. |

The solver searches candidate `(trial_slice, event_offset)` pairs.

⚠ **Two code paths, not one.** Inside the **converter** the source JSONs are still on disk, so run
boundaries give an authoritative candidate set. The **pkl-repair** solver has no such access — run
boundaries are lost at pkl-build time — so `solve_alignment(trials, ni_events)` is necessarily a
bounded **brute-force** search. The spec's earlier "authoritative candidate set" language applies
only to the converter; do not design the repair solver as if it can see the JSONs.

⚠ **Do not use "one gap = one run boundary" as a heuristic.** `20082025` has *two* large
`Baseline_ON` gaps (76.8 s at index 168, 316.3 s at index 227) across ~8 runs; the gap structure
under-counts runs. Gaps are a useful *prior* for ordering candidates, never a segmentation.

⚠ **The converter fix must NOT recurse into `Session/` subfolders.** `delete/` and `partial/` hold
runs that were curated out of the analysis set deliberately. The current non-recursive glob is
*correct* in what it loads; recursing to "find the missing runs" would re-inject 228 aborted/partial
trials into `20082025` and make the trial table wrong in a new way. Sign B's fix lives entirely in
the event offset, not in run selection.

Uniqueness was checked, not assumed: exhaustive scans found exactly **one** candidate above 95 % in
the event-offset dimension (`20082025`, offset 228) and in the trial-slice dimension
(`05092025_b`, start 281). No degeneracy. The runner-up score is still reported per session so this
is demonstrated each time rather than trusted.

---

## 5. Verification

1. **Regression:** every currently-aligned session must solve to the identity map — Check 1 at
   100 %, Check 2 at ≤ 0.01 s.
2. **Null control:** deliberately wrong offsets must fail **both** checks (as demonstrated in §2),
   including ±1-trial shifts, which the sensitivity scan shows are caught. A solver that accepts a
   shuffled or off-by-one offset is a bug, not a finding.
3. **Behaviour unchanged:** trial outcomes, `change_size`, `change_time` diffed before/after repair
   on every touched pkl — must be identical.
4. **Uniqueness:** runner-up residual reported per session.
5. **Audit closes:** re-run the audit; every repaired session returns `aligned = True`.
6. **Cross-subject generalisation (already evidenced):** both checks were verified on 8 known-good
   (`diff == 0`) sessions, two each from BG_031, BG_038, BG_039 and BG_041 — Check 1 = 100.00 % and
   Check 2 = 0.0051 s on **every** subject, with no `<= 0` placeholders and no all-`NaN` sessions.
   The aligned residual is the same constant across subjects, which also rules out the concern that
   the Mar-2026 MATLAB NI re-extraction batch changed `Change_ON`/`Baseline_ON` semantics.

---

## 5a. Verification record — adversarial review, 2026-08-04

Ran the project's Gate-8 refutation battery: **6 independent Opus lenses, each reproducing the
numbers from data rather than reviewing the prose. Outcome: 1/6 refuted, 0 fatal.**

Gates 1, 3, 5 and 6 of `harden-result` (FR-normalisation, pseudoreplication, trial-count matching,
lick leakage) **do not apply** — this is a data-integrity spec with no firing rates, no grouping
variable and no neural magnitude comparison. Gates 2, 4, 7 and 8 do.

| Lens | Outcome |
|---|---|
| Reproduce | **0 discrepancies.** Every headline number re-derived exactly from X: and the pkls |
| Circularity (kill gate) | **Circularity hypothesis REFUTED** — Check 1 is sound. `trialoutcome`/`change_time` come only from the behavioural JSON; `Change_ON` only from `NIdaq_events.mat`; no code path in Python *or* MATLAB writes one from the other. Decisive empirical proof: on `05092025_b`, `len(Change_ON) = 248 ≠ 529` trials, and offset 0 scores 50.40 % — a behaviour-derived array would be length-529 and align at offset 0. Residuals are also non-zero (0.0051 s, 0/255 within 1e-9), so `Change_ON` is not a copy of `Baseline_ON + change_time` |
| Generalisation | Passed on all four other subjects (see §5.6). Surfaced the case-sensitivity/`Ref` caveat in §2 |
| Solver | No degeneracy; two-sided ±1 sensitivity confirmed by truncation test. Surfaced the `n = 0` vacuous-accept hole and the two-code-paths issue (§2, §4) |
| Downstream | Surfaced `tf_glm_data.py`, the dataclass-default failure modes, and the provisional status of the identity fallback (§3, §4) |
| **Alternative explanation** | **REFUTED the sign-B causal story** — the 228 events are seven curated runs in `delete/`+`partial/`, not an unfiled run. Verified independently before accepting (§1) |

**What this changes:** the *arithmetic and the repair are unchanged* — `20082025` still aligns at
offset 228, `05092025_b` still at `trials[281:529]`, both at 100 % / 0.0051 s. What changed is the
**named cause** for sign B, and therefore the converter fix: sign B needs no run selection, and
recursing the glob would actively make it worse.

**Standing caveats carried forward:** Check 1's outcome set is case-sensitive and must not be
refactored onto `EVENT_VALID_OUTCOMES`; the ~50 % wrong-offset floor is outcome-balance-dependent;
`n = 0` must reject; the identity-map fallback is provisional until Phase 2; the consumer audit is
incomplete until systematically run.

---

## 6. Phasing

| Phase | Content | Gate |
|---|---|---|
| **1** | `run_alignment.py` + tests; repair BG_046 `20082025` and `05092025_b` | Both solve at ~0.005 s; behaviour diff clean |
| **2** | Extend audit to residuals; re-verify all 253 pkls including the 48 benign | Residual distribution characterised |
| **3** | Repair the remaining 15 unsafe non-BG_012 sessions (13 ordinary + the 2 special cases in §7) | Each solves or is flagged `-1` with a recorded reason |
| **4** | Patch `ingest.py`; honour `trial_event_index` downstream; **regenerate the TF registries** | Converter refuses to emit a misaligned pkl; registries rebuilt from repaired pkls |

### Measured downstream contamination (TF-GLM registries)

Affected sessions **are** already in the shipped registries, so this is not hypothetical:

| subject | region | affected | pooled `resp_log2` (on record) | clean-only | affected-only |
|---|---|---|---|---|---|
| BG_046 | DMS | 2/46 sess, 7.1 % of units | 2.77 % | 2.84 % | 1.79 % |
| BG_031 | **VMS** | **7/42 sess, 20.0 % of units** | **5.29 %** | **6.31 %** | **1.26 %** |
| BG_039 | DMS | 1/32 sess, 3.1 % of units | 3.07 % | 3.04 % | 3.95 % (n=76 — too thin to read) |

The recorded headline "VMS 5.3 % > DMS 2.8–3.1 %" (`tf_glm_replication_jun2026`) **is** the pooled,
contaminated figure. The direction is a **deflation** — exactly the mechanism's fingerprint, since
decorrelating stimulus from spikes drives units toward non-responsive. So the bug works *against*
the VMS>DMS result rather than creating it; excluding affected sessions widens the gap from 1.9× to
2.2×. The qualitative conclusion stands; the **effect sizes on record do not**, and the registries
must be regenerated from repaired pkls in Phase 4. ⚠ "Clean-only" above is an *exclusion* analysis,
not a repaired one — post-repair those sessions contribute real units and the numbers move again.

---

## 7. Special cases (in scope, with caveats)

- **BG_031 `20052025`** — 0 trials against 556 events. The solver cannot match an empty trial table,
  so this takes the unsolvable path: `-1` throughout, with the reason recorded. Whether its
  behaviour is recoverable from a sibling directory is a separate question, not answered here.
- **BG_038 `22082025`** — carries a *second, independent* defect: ephys 289.8 s but last event at
  7436 s, i.e. truncated spike data. Alignment repair is still attempted and reported, but it does
  **not** address the truncation, and the session must not be treated as neural-usable on the
  strength of an `aligned = True` result alone.

## 8. Out of scope

- **BG_012 (6 unsafe sessions)** — parked: protocol variants, not merely an alignment problem.
- **Behavioural re-analysis** — unaffected by this bug (handoff §2). No behavioural result is
  invalidated by anything here.
- **Same-day twin concatenation** — still forbidden; twins are re-sorts of identical behaviour.
- **Re-conversion / re-sorting** — no session is re-converted or re-sorted by this work.

---

## 9. Success criteria

1. Each failure sign has a named cause evidenced against the X: source. **(Met in §1 — sign A and
   sign B both, the latter corrected on 2026-08-04 after adversarial review; see §5a.)**
2. Every repairable session returns `aligned = True` from the extended audit.
3. Behaviour provably unchanged on every touched pkl.
4. Unrepairable sessions are documented and excluded via the index map, not silently dropped.
5. The converter cannot reintroduce the defect silently.

---

## 10. Related

- Handoff: `docs/superpowers/specs/2026-08-03-QC1-trial-baselineon-realignment-handoff.md`
- Memory: `suffixed_session_files_aug2026`, `feedback_no_compute_over_samba_gateway`,
  `feedback_circular_analysis_null_controls`, `feedback_canonical_session_id`
- Commits: `943fbdf` (loader fix + audit), `c9f8735` (handoff spec)
