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
`<SUBJ>_<YYYYMMDD>_<HHMMSS>__trials.json`, and those files are filed into session directories by
hand. The filing and the ephys therefore drift apart in two directions:

| Sign | Filing error | Result |
|---|---|---|
| **A** — `n_trials > n_events` | More runs' JSONs present than the recording covers | Trial table is a concatenation spanning runs; only a prefix/suffix belongs |
| **B** — `n_trials < n_events` | A run's JSON was never filed | Events are all real; the trials match a **later offset** into the event arrays |

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

Not spurious events. The `Baseline_ON` train contains a **316.3 s gap at index 227/228** (23× the
13.5 s median interval): the recording spans two behavioural runs. `714 = 228 + 486`, and the single
filed JSON holds exactly **486** trials. Run 1's JSON was never filed. The trials align to
`BON[228:714]` — an **offset**, not a truncation.

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

- "Mode B — cause unknown, possibly spurious NI events" is **wrong**. The events are genuine; the
  behaviour file is missing. No event filtering is needed anywhere.
- Modes A and B are **the same defect**, so one fix addresses both.
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
trial_event_index: np.ndarray   # int, length n_trials
                                # value = index into the per-trial ni_events arrays
                                # -1    = this trial has no corresponding ephys event
```

- Neural code (`align.py`, tensor builders) indexes events through this map.
- Behaviour code ignores it entirely and keeps the full trial table.
- **Backwards compatible:** absent field + matching counts → identity map, so all 182 exact-match
  pkls and the 48 benign ones behave exactly as today until re-verified.

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
| `src/visdetect/core/ingest.py` (patch) | Select the run JSON(s) matching the recording instead of blind-concatenating; run the residual check before emitting a pkl and refuse/flag on failure. Order runs by the **timestamp embedded in the filename**, never by mtime — the MATLAB port used mtime and the directories have since been touched by reorganisation passes (§1). |
| `scripts/QC_technical/audit_trial_baselineon_alignment.py` (extend) | Add `outcome_agreement`, `median_resid_s`, `runner_up_*`, `aligned`. `neural_safe` becomes evidence-based (both checks), with the count check retained only as a secondary signal. |
| `src/visdetect/analysis/align.py` + tensor builders | Honour `trial_event_index`; drop `-1` trials from event-aligned analyses. |

The solver searches candidate `(trial_slice, event_offset)` pairs. Behavioural run boundaries read
from the source JSONs give an authoritative candidate set; a bounded brute-force search over offsets
is the fallback. Given the ~300× separation, a false positive is implausible — and the runner-up
score is reported so uniqueness is demonstrated rather than trusted.

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

---

## 6. Phasing

| Phase | Content | Gate |
|---|---|---|
| **1** | `run_alignment.py` + tests; repair BG_046 `20082025` and `05092025_b` | Both solve at ~0.005 s; behaviour diff clean |
| **2** | Extend audit to residuals; re-verify all 253 pkls including the 48 benign | Residual distribution characterised |
| **3** | Repair the remaining 15 unsafe non-BG_012 sessions (13 ordinary + the 2 special cases in §7) | Each solves or is flagged `-1` with a recorded reason |
| **4** | Patch `ingest.py`; honour `trial_event_index` downstream | Converter refuses to emit a misaligned pkl |

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

1. Each failure sign has a named cause evidenced against the X: source. **(Met in §1.)**
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
