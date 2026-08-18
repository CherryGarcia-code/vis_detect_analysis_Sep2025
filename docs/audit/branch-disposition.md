# Branch disposition — DRAFT for owner review

Deliverable 7 of sub-project 0: what happens to every branch, stash-tag and
uncommitted file when the old repo is frozen (sub-project 6 cutover).

**The `DECISION` column is deliberately empty. The audit does not decide.**
`RECOMMENDED` is the evidence-backed default from the migration brief in
`docs/superpowers/specs/2026-08-05-new-repo-master-design.md`; the project owner
fills `DECISION` at review. Where the evidence contradicts a recommendation's
stated reasoning, the row is flagged inline and adjudicated in an evidence note
below the table rather than silently resolved — one row is in that state today
(`feature/fig5eh-preparatory-cellclass`: its stated precondition fails, though
the drop turns out to be safe for a different, verified reason).

**Snapshot as-of 2026-08-13 15:12.** The commit columns move: the audit's own
commits inflate `design/new-repo-foundation`, and
`feature/early-lick-and-session-sorting` gained a commit from a concurrent QC1
session *while this table was being generated* (see
`docs/audit/07-work-at-risk.md`). Re-run both D7 scripts immediately before the
freeze decision rather than trusting these counts.

Evidence: `d7.branches.unmerged`, `d7.local_only.commits`,
`d7.untracked.at_risk`, `d7.stash.0`, `d7.stash.1` in
`docs/audit/measurements.csv`. Generated read-only — **no branch, worktree,
stash-tag or untracked file was created, staged, modified or deleted by this
audit.**

## Reading the commit columns

- **ahead (raw/cherry)** — `git rev-list --count main..B` / count of `+` lines
  from `git cherry main B`. `git cherry` skips merge commits, so `4/3` normally
  means "one of the four is a merge", **not** "one is already applied".
- **cherry `-`** is the only signal that means *already applied upstream under a
  rewritten sha* → safe to drop. Exactly one branch shows it
  (`feature/tf-transient-sustained-spectrum`).
- **local-only** — commits on this branch that exist on no `origin/*` ref
  (`git rev-list --count origin/B..B`). These are the ones a disk failure
  destroys. ⚠️ `origin/*` are **cached** remote-tracking refs (last fetch
  2026-08-06); `git ls-remote` fails here because the SSH agent is not running,
  so remote state is asserted from the cache, not verified.

## Branches

| Branch | Worktree | Last commit | Ahead (raw/cherry) | Local-only | What it holds | RECOMMENDED | DECISION |
|---|---|---|---|---|---|---|---|
| `design/new-repo-foundation` | *(primary checkout)* | 2026-08-13 | 64/63 (1 merge) | **16** | The new-repo design corpus (8 ADRs, master design, panel review) + 16 audit commits (Tasks 1–12) — the 16 local-only commits **are** the audit commits | **Keep — active work.** Push the local-only commits before any freeze step | |
| `feature/camera-tagger-2b` | `camera-tagger-2b` | 2026-08-06 | 0/0 | 0 | Nothing unique — merged to main as `caa377d` (amortized ROI + per-frame pupil label capture, Plan 2b) | **Port-on-first-use (whole subsystem)** — the camera/tagger subsystem is cold-listed under ADR-020, not carried at cutover. *(This recommendation is about the subsystem's ADR-020 cold-list status, not the ref — the ref itself is droppable because its content is on `main`; mirrored from `drop-list.md` §1, 2026-08-18, Task 16 final wave)* | |
| `feature/early-lick-and-session-sorting` | `qc1-alignment` | **2026-08-13 (moved during this audit)** | 33/32 (1 merge) | **3** | Live QC1 trial/event-alignment repair: 32 files, +8,821/−23 (solver, repair script, integrity gate, verification harnesses) | **Owner decision — live work in flight.** Not a freeze candidate while QC1 is open; push the 3 local-only commits regardless | |
| `feature/fig5eh-preparatory-cellclass` | *(none)* | 2026-07-24 | 4/4 | 0 | 4 doc/spec commits (prep-activity vs regulation-axis spec, Phase-2 trajectories, Appendix A dataset inventory) | **Drop after verifying still-ancestor** — ⚠️ *the stated precondition FAILS (not an ancestor of main, 4 `+` patches), but the drop is safe by a different route: all 4 commits are ancestors of `design/new-repo-foundation` and its origin ref. See evidence note* | |
| `feature/population-field-plan2` | `population-field-plan2` | 2026-08-04 | 4/3 (1 merge) | 0 | 3 unique doc commits: Plan 2 analysis-layers design spec, Plan 2a implementation plan, NI lick-channel semantics null control | **Merge docs to main before freeze** | |
| `feature/tf-transient-sustained-spectrum` | *(none)* | 2026-07-10 | 1/0 | 0 | 1 commit (`2f82abe`, anatomy TF/kernel-width cell maps) — `git cherry` reports `-`: already applied upstream under a rewritten sha | **Drop (cherry-verified applied)** | |
| `fix/lick-channel-resolver` | `lick-channel-fix` | 2026-08-03 | 0/0 | 0 | Nothing unique — fully contained in main | **Drop** — 0 unique commits | |
| `hardening/fa-psth-and-manifest-sort` | *(none)* | 2026-07-21 | 3/3 | 0 | `2f6fcdc` (centralised `fa_lick` PSTH condition + manifest sort fix for DDMMYY tokens) plus 2 doc commits shared with `fig5eh` | **Carry the fix into the new-repo foundation** — the sort fix is the known-defect register's session-id/date-ordering entry. ⚠️ *Load-bearing: `2f6fcdc` is contained in no other branch (see evidence note) — dropping this ref strands the fix* | |
| `worktree-camera-tagging` | `camera-tagger-2a` | 2026-07-29 | 0/0 | 0 | Nothing unique — fully contained in main | **Drop** — 0 unique commits | |
| `worktree-population-field` | `population-field` | 2026-07-09 | 0/0 | 0 | Nothing unique — fully contained in main | **Drop** — 0 unique commits | |
| `worktree-theta-prototype` | `theta-prototype` | 2026-07-21 | 0/0 | 0 | Nothing unique — fully contained in main | **Drop** — 0 unique commits | |

`main` is not a disposition row — it is the freeze baseline. It nevertheless
carries **22 commits that exist on no `origin/*` ref** (tip `caa377d`,
2026-08-07; cached `origin/main` tip 2026-08-04), so "freeze main" is not a
no-op: main must be pushed first.

### Evidence note — `feature/fig5eh-preparatory-cellclass`: precondition fails, drop is still safe

Two separate questions, and they come apart on this branch.

**1. The brief's stated precondition does not discharge.** "Drop after verifying
still-ancestor" is a claim about `main`, and against `main` it is false:

```
$ git merge-base --is-ancestor feature/fig5eh-preparatory-cellclass main   -> NOT-ANCESTOR (exit 1)
$ git cherry main feature/fig5eh-preparatory-cellclass                     -> 4 patches, all "+"
```

So the branch is *not* contained in main, and nobody should drop it on the
strength of the recommendation as worded.

**2. The drop is nevertheless safe, by a different route.** All four commits are
ancestors of the active `design/new-repo-foundation` branch (and of
`feature/early-lick-and-session-sorting`), and of the cached `origin/*` refs for
both:

```
$ git branch -a --contains 5d1e732        (identical for 6cad30e; 381b4d3/6da0f31 additionally list
                                           hardening/fa-psth-and-manifest-sort + its origin ref —
                                           extra containment only strengthens safe-to-drop)
* design/new-repo-foundation
+ feature/early-lick-and-session-sorting
  feature/fig5eh-preparatory-cellclass
  remotes/origin/design/new-repo-foundation
  remotes/origin/feature/early-lick-and-session-sorting
  remotes/origin/feature/fig5eh-preparatory-cellclass

$ git merge-base --is-ancestor <each of the 4> design/new-repo-foundation        -> ANCESTOR (exit 0)
$ git merge-base --is-ancestor <each of the 4> origin/design/new-repo-foundation -> ANCESTOR (exit 0)
$ git log --ancestry-path --oneline 5d1e732..design/new-repo-foundation          -> 61 commits (linear path exists)
```

*(Annotation corrected 2026-08-18, Task 16 final wave: it previously claimed
identical `--contains` output for all four commits, which is false for
`381b4d3`/`6da0f31` — re-verified read-only; those two are exactly the doc
commits the contrast note below records as shared with
`hardening/fa-psth-and-manifest-sort`.)*

**Deleting the `fig5eh` ref therefore loses nothing**, and the content reaches
main automatically when `design/new-repo-foundation` merges. No merge-to-main
of this branch is needed, and dropping it is not a discard of anything. This is
consistent with the table row's `Local-only = 0`. The four commits are all
documentation (`381b4d3`, `6da0f31` regulation-axis spec; `5d1e732` Phase-2
trajectories + circularity-scope fix; `6cad30e` Appendix A dataset inventory) —
the ref is a redundant label on content held by two live branches.

**Contrast — `hardening/fa-psth-and-manifest-sort` is the opposite case, and its
recommendation is load-bearing.** Its code fix exists nowhere else:

```
$ git branch -a --contains 2f6fcdc
  hardening/fa-psth-and-manifest-sort
  remotes/origin/hardening/fa-psth-and-manifest-sort
$ git merge-base --is-ancestor 2f6fcdc main                      -> NOT-ANCESTOR
$ git merge-base --is-ancestor 2f6fcdc design/new-repo-foundation -> NOT-ANCESTOR
```

Dropping that ref *would* strand the centralised `fa_lick` PSTH condition and
the DDMMYY manifest-sort fix. (Its two doc commits `381b4d3`/`6da0f31` are the
shared ones above and are safe regardless — only `2f6fcdc` is unique to it.)
**`fig5eh` = safe to drop despite the failed precondition; `hardening` = must be
carried.** Both show `Local-only = 0`, so that column alone does not
distinguish them — "on a remote" is not the same question as "held by a branch
that will survive the freeze".

## Stash-tags

Both tags exist and are readable. **Neither is on any remote ref** — `git push`
does not push tags by default, so their 6 commits are part of the 31 local-only
commits recorded in `d7.local_only.commits`.

| Tag | Date | Content (`git show --stat`) | Still-live targets | RECOMMENDED | DECISION |
|---|---|---|---|---|---|
| `pre-tidy-20260628/stash-0` | 2026-05-14 | 9 files, +304/−78 (changed lines per file): `corneal_spatial_diagnostic.py` 111, `hmm.py` 105, `fit_behavioral_hmm.py` 64, `video_sync.py` 54, `analysis/config.py` 19, `loader.py` 11, `codebase-auditor/SKILL.md` 7, `core/ingest.py` 6, `b_hmm_state_dynamics.py` 5 | 7 of 9 — `analysis_suite/01_behavior/b_hmm_state_dynamics.py` and `analysis_suite/loader.py` no longer exist on main (archived 2026-07-01) | **Owner decision.** 3 months stale, from a migration experiment, partly aimed at an archived tree — but 7 targets are still live and the diff was never reviewed. Push the tag, then adjudicate; do not carry blind | |
| `pre-tidy-20260628/stash-1` | 2026-05-13 | 1 file, +31/−8: `scripts/video/corneal_spatial_diagnostic.py` | 1 of 1 | **Owner decision.** Same vintage; strictly a subset in scope of stash-0's change to the same file. Likely superseded — diff against stash-0 before discarding | |

The 9 `pre-tidy-20260628/feature/*` branch-archive tags carry **0** local-only
commits (every commit they name is reachable from an `origin/*` ref), so the
*work* they preserve is safe even if the *tag labels* are local-only —
unverifiable here because `ls-remote` fails.

## Uncommitted working-tree files (primary checkout)

Never staged, modified or deleted by this audit. "On a ref?" is a basename-wide
search over all `refs/heads` + `refs/tags` with a byte-level content compare.

| Path | Size / age | On a ref? | RECOMMENDED | DECISION |
|---|---|---|---|---|
| `scripts/QC_technical/characterize_unsolvable_alignment.py` | 676 lines, 2026-08-05 | **Yes — byte-identical** to the copy on `feature/early-lick-and-session-sorting` (`188608c`) | **Owner decision (live work).** Content is *not* at risk; this is a stray duplicate of committed QC1 work sitting in the wrong checkout | |
| `scripts/QC_technical/validate_event_spike_clock_drift.py` | 144 lines, 2026-07-24 | **No ref** | **Owner decision (live work)** — commit to the QC1 branch or discard deliberately | |
| `scripts/chronic_feasibility/chronic_feasibility_figure.py` | 475 lines, 2026-07-23 | **No ref** | **Owner decision** — single-copy figure script, no branch home | |
| `scripts/optotagging/render_opto_exemplar_figure.py` | 282 lines, 2026-07-23 | **No ref** | **Owner decision** — single-copy figure script, no branch home | |
| `scripts/tracking_dant/exemplar_tracking_figure.py` | 588 lines, 2026-07-21 | **No ref** | **Owner decision** — single-copy figure script, no branch home | |
| `scratchpad_state_bout_inventory.csv` | 187 rows, 2026-07-31 | **No ref** | **Owner decision** — scratch output at repo root; regenerable only if the producing script still exists | |

`.claude/settings.json` shows as modified and is **excluded by design**: it is
the owner's environment configuration (subagent-model repin), not project work.
The audit's own new scripts under `scripts/audit/` are committed by this task
and are not disposition rows.

## Not carried by any disposition: the gitignored artefact mass

Every row above concerns *git* content. The ~155 GB of gitignored
`data/`+`FIGURES/` artefacts (~110 GB of it in the primary checkout,
`d7.gitignored.volume`) is carried by **no**
branch migration and has to be dispositioned separately — see
`docs/audit/07-work-at-risk.md`, which also records the four hand-labelled sets
no code can regenerate.
