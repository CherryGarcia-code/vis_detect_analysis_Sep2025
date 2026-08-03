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
The `change_time` residual of §2 remains the single strong discriminator.

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

Therefore alignment is directly testable per trial, against the trial table's own `change_time`:

```
residual_i = (Change_ON[j] − Baseline_ON[j]) − trials[i].change_time
score      = median |residual|      over trials with finite change_time
```

Measured separation:

| session | candidate | median abs residual | within 50 ms |
|---|---|---|---|
| `19082025` (known good) | offset 0 | **0.0051 s** | 100.0 % |
| `20082025` | `BON[228:]` (hypothesis) | **0.0051 s** | 100.0 % |
| `20082025` | `BON[0:]` (current, null) | 1.4052 s | 0.4 % |
| `05092025_b` | `trials[281:529]` (hypothesis) | **0.0051 s** | 100.0 % |
| `05092025_b` | `trials[0:248]` (null) | 1.5302 s | 1.6 % |

Correct alignments reproduce the known-good value exactly; wrong ones sit ~300× away. This single
quantity is both the **solver objective** and the **acceptance test**.

It supersedes count-matching, which is only a proxy: counts cannot distinguish a benign `+3`
end-of-session artifact from a genuine 3-event offset. After this work, `neural_safe` becomes a
*measured* property rather than an inferred one — including for the 48 benign sessions, which are
currently assumed fine rather than shown to be.

**Rejected alternative.** Reconstructing inter-trial intervals from `change_time + ITI` was tested
first and is underpowered: the known-good control scores only Spearman r = +0.097, so it cannot
discriminate. `stim_vbl` (per-frame stimulus timestamps) would be ideal but is `None` in these pkls.

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
| Acceptance threshold | **median \|residual\| < 0.05 s** — 10× above the observed aligned value, 28× below the misaligned one, sitting in a wide empty gap. Log the actual residual **and the runner-up candidate** for every session so the margin is visible, never assumed |

---

## 4. Components

| Component | Responsibility |
|---|---|
| `src/visdetect/core/run_alignment.py` | Pure, unit-testable. `per_trial_event_keys(ni_events)`, `alignment_residual(trials, ni_events, trial_slice, event_offset)`, `solve_alignment(trials, ni_events) -> Alignment \| None`. Returns best **and** runner-up score. |
| `scripts/QC_technical/repair_trial_event_alignment.py` | Per pkl: solve → verify → back up → write `trial_event_index`. Emits `data/cache/qc_alignment/alignment_repair_report.csv`. Backup goes to `data/pkls/<SUBJ>/qc1_backup/<file>.bak_<UTC-stamp>` — written **before** any mutation, and the repair aborts if the backup cannot be created. Re-running is idempotent: an already-repaired pkl re-solves to the same map and is not re-backed-up. |
| `src/visdetect/core/ingest.py` (patch) | Select the run JSON(s) matching the recording instead of blind-concatenating; run the residual check before emitting a pkl and refuse/flag on failure. Order runs by the **timestamp embedded in the filename**, never by mtime — the MATLAB port used mtime and the directories have since been touched by reorganisation passes (§1). |
| `scripts/QC_technical/audit_trial_baselineon_alignment.py` (extend) | Add `median_resid_s`, `runner_up_resid_s`, `aligned`. `neural_safe` becomes residual-based, with the count check retained as a secondary signal. |
| `src/visdetect/analysis/align.py` + tensor builders | Honour `trial_event_index`; drop `-1` trials from event-aligned analyses. |

The solver searches candidate `(trial_slice, event_offset)` pairs. Behavioural run boundaries read
from the source JSONs give an authoritative candidate set; a bounded brute-force search over offsets
is the fallback. Given the ~300× separation, a false positive is implausible — and the runner-up
score is reported so uniqueness is demonstrated rather than trusted.

---

## 5. Verification

1. **Regression:** every currently-aligned session must solve to the identity map at ≤ 0.01 s.
2. **Null control:** deliberately wrong offsets must fail the threshold (as demonstrated in §2). A
   solver that accepts a shuffled offset is a bug, not a finding.
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
