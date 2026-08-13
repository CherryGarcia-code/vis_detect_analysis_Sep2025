# D7 — work at risk

What exists in only one place. Sub-project −1 exists because the scoping recon
measured this exposure and found it unacceptable to freeze or replace a repo
whose only copy is local. D7 re-measures it after the 2026-08-06 push, sizes the
gitignored artefact mass no branch migration carries, and drafts the per-branch
disposition table (deliverable 7).

**Strictly read-only.** The scripts walk and `lstat`; they issue `git`
*plumbing* queries only. No `git worktree` command was run, and nothing under
`.claude/worktrees/`, `data/` or `FIGURES/` was created, staged, modified or
deleted.

- Script: `scripts/audit/d7_work_at_risk.py`
  (`py scripts/audit/d7_work_at_risk.py`, exit 0)
- Supplement: `scripts/audit/d7_work_at_risk_supplement.py`
  (`py scripts/audit/d7_work_at_risk_supplement.py`, exit 0) — the three rows
  the disposition table's verdicts rest on
- Disposition draft: `docs/audit/branch-disposition.md`
  (RECOMMENDED filled, **DECISION empty — the owner fills it at review**)
- Measurement ids: `d7.*` in `docs/audit/measurements.csv`

## Summary

| Measurement | Value | Notes |
|---|---|---|
| `d7.gitignored.volume` | **~155 GB** over 16 entries (8 checkouts × `data/`+`FIGURES/`) | ~110 GB of it is the primary tree (`data/` 54,352 files / 72.5 GB + `FIGURES/` 3,056 files / 37.8 GB); ~45 GB is one duplicated staged video; **2 junctions pruned** |
| `d7.branches.unmerged` | 11 non-main branches: **6 ahead of main, 5 fully contained in it** | only 1 of the 6 is cherry-verified already-applied |
| `d7.local_only.commits` | **31** (20 branch + 6 stash-tag + 5 harness checkpoint refs) | down from the recon's 139, but regrowing since the 2026-08-06 push |
| `d7.untracked.at_risk` | **5 of 6** untracked entries exist on no ref | the 6th is byte-identical to a committed QC1 copy |
| `d7.handlabels.exposure` | 269 files / 31.0 MB of hand labels, **220 of them untracked** | backed up 2026-08-06 — but to the **same physical disk** |
| `d7.stash.0` / `d7.stash.1` | both tags present and readable | 2026-05-13/14; neither is on any remote ref |

## The junction hazard is live, and the prune works

The repo's worst historical incident (2026-06-07) was a `git worktree remove`
that followed an NTFS junction out of a worktree and deleted the primary pkls
and FIGURES. That shape is **still present on disk today**:

```
.claude/worktrees/qc1-alignment/data/pkls          -> junction
.claude/worktrees/qc1-alignment/data/cache/qc_alignment -> junction
```

`data/pkls` in the primary tree is **31.97 GB / 285 files**. A naive
`Path.rglob` inventory would have walked into it through the junction and
reported it twice; a naive cleanup would delete it. NTFS junctions are *not*
symlinks to Python 3.10 — `is_symlink()` returns `False` for them — so the
inventory prunes on `FILE_ATTRIBUTE_REPARSE_POINT` at **every** directory level
and treats an unreadable entry as a boundary rather than descending. Two
junctions were pruned in this run; no other tree contained one.

## Gitignored volume: ~155 GB, carried by no branch

Per-tree, junction-pruned (`d7.gitignored.volume`). The primary checkout holds
essentially all of it:

| Tree | `data/` | `FIGURES/` |
|---|---|---|
| primary | 54,352 files / **72.5 GB** | 3,056 files / **37.8 GB** |
| `camera-tagger-2a` | 67 files / 22.3 GB | 139 files / ~0 GB |
| `camera-tagger-2b` | 77 files / 22.3 GB | 139 files / ~0 GB |
| `lick-channel-fix`, `population-field-plan2`, `qc1-alignment`, `theta-prototype` | 49–62 files / ~0 GB | 139 files / ~0 GB |
| `population-field` | 49 files / ~0 GB | 4 files / ~0 GB |

Two readings matter:

- **The worktree `FIGURES/` dirs are not artefact mass.** 139 files is exactly
  the number of `FIGURES/` paths tracked in git (`git ls-files FIGURES | wc -l`
  = 139, force-added under a `FIGURES/` ignore rule). Those trees hold committed
  figures and nothing else — they are cheap to discard.
- **The two camera worktrees are the exception, and it is a duplicate.** 22.3 GB
  each is a single staged file, `data/_staging/video/BG_031/09042025/
  BG_031_090425_Eye_cam.mp4` (21.99 GB), present in *both* worktrees: ~45 GB of
  the same staged raw video living in gitignored worktree trees.

Primary composition: `data/pkls` 31.97 GB, `data/cache` 16.71 GB (16,802
files), `data/anatomy` 14.88 GB, `data/unit_match` 8.96 GB (37,041 files);
`FIGURES/tracking_dant` alone is 37.51 GB of the 37.8 GB figure total. None of
this moves with a branch, and the pkl tree is the object ADR/sub-project 6
schedules for deletion only after a digest-verified round-trip.

## Local-only commits: 31, and what they actually are

`git rev-list --count --all --not --remotes` = **31**, decomposed:

| Ref class | Count | What |
|---|---|---|
| `refs/heads` | **20** | real project work: `main` +22 vs cached `origin/main`, `design/new-repo-foundation` +16, `feature/early-lick-and-session-sorting` +3 (20 unique after overlap) |
| `refs/tags` | **6** | the two stash-tags (3 commits each: stash + index + untracked) — **never pushed**; `git push` does not push tags by default |
| `refs/sessions/**` | **5** | Claude Code harness turn-checkpoints, not project work |

Compared with the recon's 139, the 2026-08-06 push closed most of the exposure —
but it is regrowing: **all 16 local-only commits on
`design/new-repo-foundation` are this audit's own** (Tasks 1–12), and the
mechanism that closed the exposure is currently
unavailable: `ssh-add -l` reports no agent, and `git ls-remote origin` fails
with `Permission denied (publickey)`.

**This makes every `origin/*` claim in this document a statement about a cached
remote-tracking ref last updated 2026-08-06, not about the remote.** A branch
recorded here as "0 local-only" is safe *if* the cache is truthful. Re-verify
with a real `ls-remote` before acting on any drop recommendation.

### These two ids are a snapshot of a moving target (as-of 2026-08-13 15:12)

Two distinct sources of drift were observed *during this task*, and both are
recorded rather than smoothed over:

1. **Self-reference.** `d7.local_only.commits` counts the audit's own commits.
   The first run read 30; committing D7 made it 31. It reads N+1 the moment this
   document is committed, and there is no fixed point — the number is only ever
   correct as-of a timestamp.
2. **A concurrent session moved a branch mid-audit.** The first run of
   `d7_work_at_risk.py` (14:34) recorded
   `feature/early-lick-and-session-sorting` at 32/31; by 14:52 a QC1 session in
   the `qc1-alignment` worktree had committed `a029ba3` ("align.py honours
   `trial_event_index`"), making it 33/32. That skewed the two ids against each
   other, so **both scripts were re-run back-to-back and the mutually consistent
   snapshot is what is recorded.** The operative point for the disposition table
   is not the arithmetic: **the QC1 branch is live work, still receiving commits
   on the day of the audit,** and is not a freeze candidate.

## Untracked work: 5 of 6 entries exist on no ref

Basename-wide search across all `refs/heads` + `refs/tags` with a byte-level
compare (`d7.untracked.at_risk`):

| Path | Verdict |
|---|---|
| `scripts/QC_technical/characterize_unsolvable_alignment.py` | on `feature/early-lick-and-session-sorting` (`188608c`), **byte-identical** — a stray duplicate, not at risk |
| `scripts/QC_technical/validate_event_spike_clock_drift.py` | **no ref** — 144 lines |
| `scripts/chronic_feasibility/chronic_feasibility_figure.py` | **no ref** — 475 lines |
| `scripts/optotagging/render_opto_exemplar_figure.py` | **no ref** — 282 lines |
| `scripts/tracking_dant/exemplar_tracking_figure.py` | **no ref** — 588 lines |
| `scratchpad_state_bout_inventory.csv` | **no ref** — 187 rows |

That is 1,489 lines of single-copy analysis/figure code, all dated 21–24 July
2026 (the CSV, 2026-07-31), preserved by nothing.

Method caveats: the compare is per file basename, so a renamed file with edited content
reports `no ref` (correct for "this exact content is unpreserved", conservative
for "this work is unpreserved"); the byte compare normalises `\r\n` → `\n` and
must read bytes, not decoded text — decoding `git show` output with the console
codec (cp1252) while the file is UTF-8 produced a spurious `DIVERGENT` verdict
on the first run of this probe, corrected before recording.

`.claude/settings.json` (modified) is the owner's environment configuration, not
project work, and is out of scope by design. This audit never staged, edited or
reverted any of the seven paths.

## The irreplaceable slice: hand labels

Most of the 155 GB is regenerable given the pkls. These four sets are not — no
code produces them, only a human did (`d7.handlabels.exposure`):

| Set | Files | Size | Tracked in git |
|---|---|---|---|
| `data/cache/tf_labeling` (TF unit labels, 4,725 units) | 1 | 1.5 MB | **0** |
| `data/cache/state_tags` (behavioural state tags) | 202 | 29.4 MB | **0** |
| `data/cache/session_sorting` (blinded session sorter) | 13 | 0.1 MB | 1 |
| `data/cache/video_sync` (video-sync / pupil labels) | 53 | ~0 MB | 48 |

220 of 269 files exist only as gitignored bytes on disk. A backup does exist —
`e:/python_analysis/_handlabel_backup_20260806`, 224 files / 31.0 MB, covering
all four sets plus the camera pilot backups — **but it sits on the same physical
disk (`E:`) as the repo.** That is a second copy, not the off-disk copy
sub-project −1 requires: one drive failure still takes both.

## Stash-tags: both present, both unpushed, both stale

Neither tag is an error case — both resolve and both diff cleanly.

- **`pre-tidy-20260628/stash-0`** (2026-05-14, "On main:
  migration:main-agents/neuroscience-data-analysis-setup"): 9 files,
  +304/−78, touching `hmm.py`, `video_sync.py`,
  `corneal_spatial_diagnostic.py`, `fit_behavioral_hmm.py`,
  `analysis/config.py`, `core/ingest.py` and the `codebase-auditor` skill.
  **7 of its 9 targets still exist on main**; the other two
  (`analysis_suite/01_behavior/b_hmm_state_dynamics.py`,
  `analysis_suite/loader.py`) were archived 2026-07-01.
- **`pre-tidy-20260628/stash-1`** (2026-05-13, video-sync baseline-onset
  detection): 1 file, +31/−8, `scripts/video/corneal_spatial_diagnostic.py` —
  the same file stash-0 also changes, so it is plausibly a subset superseded by
  stash-0.

Three months of subsequent work has landed on 7 of those 9 files, so neither
stash can be applied blind; the question at review is whether the *intent* of
either change survived. The audit's finding is narrower and firm: **the two tags
carry 6 commits that exist on no remote ref**, so securing them is a push, not a
merge decision.

## What D7 does not measure

- **Whether a dropped branch's ideas were reimplemented elsewhere.** `git cherry`
  detects patch-identity, not intellectual overlap. The only branch *cherry*
  clears outright is `feature/tf-transient-sustained-spectrum` — but cherry
  against `main` is the wrong question for a branch whose commits are already
  held by another live branch. `git branch -a --contains` is that test, and it
  clears `feature/fig5eh-preparatory-cellclass` too (all 4 commits are ancestors
  of `design/new-repo-foundation`) while showing
  `hardening/fa-psth-and-manifest-sort`'s fix is held nowhere else. See the
  evidence note in `docs/audit/branch-disposition.md`.
- **Remote reality** — see the `ls-remote` caveat above.
- **Whether the gitignored artefacts are actually regenerable.** The 155 GB is
  sized, not provenance-checked; D4's artefact provenance survey is the place
  that question is answered, and it found 15.8k corrupted cache rows.
- **Disk-level identity of the two 21.99 GB video copies.** Same name and size
  in both camera worktrees; whether NTFS deduplicates or hardlinks them was not
  probed (that would require write-adjacent tooling on the worktree trees).
