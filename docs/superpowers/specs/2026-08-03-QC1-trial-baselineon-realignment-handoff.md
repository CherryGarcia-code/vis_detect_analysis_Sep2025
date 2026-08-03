# QC1 — Trial↔Baseline_ON misalignment: diagnose and repair (HANDOFF)

**Date:** 2026-08-03 · **Status:** handoff, not started · **ID:** QC1 (new; data-integrity, not a science question)
**Boot this document cold in a fresh chat — it is self-contained.**

---

## 0. Standing rules, and the ONE carve-out for this task

The project hard rule is **never run pipelines/compute over the X: Samba gateway** (it locks ceph).

> **CARVE-OUT GRANTED FOR THIS TASK (user, 2026-08-03):**
> You **MAY READ** from the user's own folder on **X:** — raw/processed session files and their
> metadata — because diagnosis requires the source data.
> You **MUST NOT WRITE, MOVE, RENAME, OR DELETE ANYTHING ON X:**. Read-only. Full stop.
> Copy what you need to local disk and work there. Do not run heavy conversion/sorting compute
> *over* the share — pull the bytes you need, then compute locally (or on HPC/Slurm).

Other standing constraints that still apply:
- Multiple **git worktrees** are live (parallel chats). Verify your branch before any git op;
  never `git worktree remove`, never `rm -rf data`, never force-push or reset another branch.
- `py` not `python`; venv at `.venv\Scripts\python.exe`.
- New work → `scripts/<topic>/`, `FIGURES/<topic>/<SUBJ>/`, `data/cache/<topic>/`.

---

## 1. The problem in one paragraph

Some recording days were **restarted after a problem**, producing a second ephys file (`_b`, `_c`,
`_v2`). For at least some of those days the converter attached the **full day's behavioural trial
table to each partial ephys recording**. The result is that a pkl can contain N trials while its
`ni_events["Baseline_ON"]` holds a different number of events — so **trial *i* does not correspond
to event *i***, and every `ni_events`-aligned neural analysis on that session is silently wrong.

Worked example, BG_031 `19052025`: the plain pkl has **231** Baseline_ON events, the `_b` pkl has
**339**, and **231 + 339 = 570 ≈ the 569 trials that BOTH files claim.** The day was split across
two recordings; neither pkl is internally aligned.

---

## 2. SCOPE — read this before you panic or over-fix

| Analysis type | Affected? |
|---|---|
| Behaviour-only (trial outcomes, RTs, `change_size`, `change_time`, state tags, session sorting, early-lick/hazard work) | **NO.** These read the **trial table** only and never touch `ni_events`. All completed behavioural results stand. |
| Neural aligned to `ni_events` (Baseline_ON / Change_ON PETHs, population tensors, decoding, CDs) | **INVALID** on the flagged sessions. |

Do **not** invalidate behavioural conclusions on the strength of this bug.

---

## 3. What already exists (do not redo)

- **QC audit script (canonical):** `scripts/QC_technical/audit_trial_baselineon_alignment.py`
  Run: `py scripts/QC_technical/audit_trial_baselineon_alignment.py`
  Out: `data/cache/qc_alignment/trial_vs_baselineon_audit.csv`, with a **`neural_safe`** column
  (`|n_trials − n_Baseline_ON| ≤ 9`). **253 pkls audited → 182 exact match, 230 safe, 23 unsafe.**
- **Loader fixed** (committed `943fbdf`): `resolve_session_pkl` now falls back to a *uniquely*
  suffixed file when no plain one exists (additive — nothing that previously resolved changed);
  new `list_session_recordings(session_name, subject)` returns **every** pkl for a date;
  `list_pkl_sessions` no longer drops suffixed-only dates.
- **Manifests deduplicated** (on disk, gitignored; timestamped `.bak_*` alongside):
  BG_031 42→41 rows, BG_039 32→31.
- **Small mismatches are benign:** 44 sessions are `+1…+9` — a baseline that started but never
  logged a trial at session end. `TOL_BENIGN = 9`. Leave them alone.

---

## 4. The 23 neural-unsafe sessions, grouped by failure mode

### Mode A — behaviour spans MORE than this ephys (split recording). 12 sessions.
`n_trials ≫ n_Baseline_ON`.

| subject | file | trials | BON | diff | manifest stage |
|---|---|---|---|---|---|
| BG_038 | `08082025` | 2046 | 850 | −1196 | **Expert** |
| BG_039 | `20052025` | 1068 | 512 | −556 | **Expert** |
| BG_031 | `10042025` | 955 | 498 | −457 | Excluded |
| BG_031 | `050325` | 699 | 246 | −453 | **Naive** |
| BG_031 | `19052025` | 569 | 231 | −338 | **Expert** |
| BG_038 | `17062025_c` | 532 | 202 | −330 | **Expert** |
| BG_046 | `05092025_b` | 529 | 248 | −281 | **Expert** |
| BG_031 | `19052025_b` | 569 | 339 | −230 | (twin of above) |
| BG_031 | `15042025` | 772 | 563 | −209 | **Expert** |
| BG_031 | `03042025` | 728 | 671 | −57 | Excluded |
| BG_038 | `22082025` | 575 | 553 | −22 | **Expert** — ⚠ ephys only **289.8 s** but last BON at 7436 s: spike data looks truncated, a *different* problem |
| BG_012 | `27102023_prot4_lickEndsTrial` | 872 | 316 | −556 | (BG_012 parked, §6) |

### Mode B — MORE Baseline_ON events than trials. 10 sessions.
Cause unknown — could be spurious NI events (then it's an **event-filtering** fix, no re-conversion),
or a truncated/failed behavioural file (then it is).

| subject | file | trials | BON | diff | stage |
|---|---|---|---|---|---|
| BG_046 | `20082025` | 486 | 714 | +228 | **Expert — PRIMARY SUBJECT** |
| BG_031 | `280325` | 279 | 761 | +482 | Excluded |
| BG_041 | `02052025` | 461 | 657 | +196 | no manifest |
| BG_041 | `09052025` | 435 | 499 | +64 | no manifest |
| BG_038 | `28072025` | 538 | 558 | +20 | **Expert** |
| BG_012 | ×5 (`18102023_…imro5`, `24102023`, `25102023`, `16112023_airpuff`, `17112023_airpuff`) | — | — | +168…+832 | parked |

### Mode C — behavioural load failed entirely. 1 session.
| BG_031 | `20052025` | **0** trials | 556 BON | not in manifest |

---

## 5. Suggested phases (each ends in a decision)

**Phase 0 — Reproduce.** Re-run the audit; confirm 23 unsafe. Read `Session` (`src/visdetect/core/session.py`) and the converter (`src/visdetect/core/ingest.py`, `scripts/conversion/raw_to_pkl.py`) to understand exactly how the trial table and `ni_events` are assembled and where they can decouple.

**Phase 1 — Diagnose per mode (READ-ONLY on X:).** For 2–3 exemplars of each mode, compare the pkl against the source on `X:/public/projects/BeJG_20230130_VisDetect/wEPhys/<SUBJECT>/{Raw,Processed} data/`. Establish for each mode: is this a converter bug, a source-data property, or spurious NI events? **Gate:** do not attempt repair until each mode's cause is named.

**Phase 2 — Decide repair strategy per mode.**
- Mode A likely splits cleanly: if a file has *k* Baseline_ON events, the first *k* trials plausibly belong to it. **Verify that assumption** (e.g. do trial `change_time`s reconstruct the observed BON intervals?) before relying on it. In-place trial-table repair may suffice; re-conversion is safer.
- Mode B: if the extra events are spurious, filter them; if the behaviour is truncated, re-convert.
- Mode C: re-convert.

**Phase 3 — Repair + verify.** Re-run the audit; every repaired session must come back `neural_safe`. **Never overwrite a pkl in place without a backup.** Behaviour must be unchanged where it was already correct — diff trial outcomes before/after.

**Phase 4 — Backfill.** Re-tag any repaired session (`scripts/state_labeling/tag_sessions.py` — ⚠ it rewrites `_tag_summary.csv` from only the sessions in the run, so back that file up). Re-run the audit and commit the updated flags.

---

## 6. Explicitly OUT OF SCOPE

- **BG_012 — PARKED** (user, 2026-08-03). All 49 pkls carry protocol descriptors
  (`prot2_v2`, `prot4_lickEndsTrial`, `prot5_lickEndsTrial`, `prot4_airpuff`, `prot4_lickTimeOut`,
  `_sated`, `_imro4/_imro5`). These are **different task rules, a satiety manipulation, and
  different probe channel maps** — a comparability problem, not just an access problem. 31/40 dates
  are now reachable; 9 remain ambiguous. Needs its own protocol inventory later.
- **Do NOT concatenate same-day twins.** Verified trial-by-trial: BG_039 `23042025` vs `_v2` have
  **identical** outcomes/`change_size`/`change_time` and **byte-identical** `ni_events` — it is the
  same ephys **re-sorted** (only 6 of 73/66 clusters shared). BG_031 `19052025` vs `_b` also carry
  identical behaviour. Concatenating would **duplicate every trial**.
- Behavioural re-analysis. Not affected (§2).

---

## 7. Gotchas that will bite you

- **Join key:** for multi-subject joins use `config.session_date_key`, **not**
  `canonical_session_id` — the latter maps `'050325'→'00050325'` but `'05032025'→'05032025'`, so the
  same date misjoins across token widths (BG_031 mixes 18 six-digit + 24 eight-digit tokens; BG_039 2 + 30).
- Session ids: leading-zero DAY is dropped by any `int()` cast (`01072025` → `1072025`).
- `data/cache/state_tags/<SUBJ>/` contains a `_tag_summary.csv` **roll-up that is not a session** —
  skip `_`-prefixed files when globbing.
- Sorts disagree substantially between twins (BG_039: 6 of ~70 clusters shared), so *which file you
  load* materially changes the unit set.
- `del sess; gc.collect()` after each session in loops — pkls are large.

---

## 8. Success criteria

1. Each of the three failure modes has a **named cause**, evidenced against the X: source.
2. Every repairable session returns `neural_safe = True` from the canonical audit.
3. Behaviour is provably unchanged where it was already correct (before/after trial-table diff).
4. Sessions that **cannot** be repaired are documented and excluded via `neural_safe`, not silently dropped.
5. Priority order = active-manifest sessions first, **BG_046 `20082025` and `05092025_b` first of all** (primary subject, both Expert).

---

## 9. Related

- Memory: `suffixed_session_files_aug2026`, `session_grouping_learning_axis_jul2026`,
  `feedback_no_compute_over_samba_gateway`, `feedback_canonical_session_id`.
- Commits: `943fbdf` (loader fix + audit), `367da5d` (S1 session grouping).
- Parent thread context: `docs/superpowers/specs/2026-07-31-S1-session-grouping-learning-axis-design.md`.
