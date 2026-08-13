# D9 — storage-format spike (NOT RUN)

The D9 spike was scoped to convert three sessions to NWB and measure what ADR-015
would have cited: on-disk size against the current pickles, three read patterns,
and pkl→NWB round-trip fidelity. **It was not run.** This document records why,
what is therefore unknown, and the two items that survive the cancellation and
must be carried into the new repo.

- Script: **none** — `scripts/audit/d9_nwb_spike.py` was never written
- Scratch venv: **not created by this task** (see *Repo state* at the end)
- `.nwb` files produced: **none**
- Measurement ids: `d9.*` in `docs/audit/measurements.csv` — **all five
  `not-measured`**, recorded via `record()` one-liners (`script = manual`)

## What the spike would have measured

Three targets — smallest BG_049 session, BG_046 `01092025`, and a BG_012 twin —
each converted with direct `pynwb` in a throwaway venv, then:

| Id | Measurement |
|---|---|
| `d9.size_ratio` | gzip-NWB / pkl on-disk size ratio (< 1.0 anticipated) |
| `d9.readtimes` | `load_pkl` / `write_nwb` / `read_trials` / `read_one_unit` / `read_window_all_units` |
| `d9.roundtrip` | equality of `spike_times` (unit 0) and the trials columns after the round trip |
| `d9.compression` | the HDF5 codec actually present on `units/spike_times` after writing |
| `d9.keep_all_good` | whether every Kilosort-good unit can be re-ingested without re-reading raw `.ap.bin` |

## Why it was not run

**Owner decision, 2026-08-13.** The project owner has pre-decided to adopt **NWB**
in the new repo, and to rebuild session files **from raw** rather than convert the
existing pickles. That voids the spike on both counts:

1. Its purpose was **comparative evidence for a decision that is now made.** The
   numbers would have informed a format choice; the format is chosen.
2. Its method exercises a **pkl→NWB conversion path the new pipeline will never
   use.** A raw→NWB rebuild does not read a pickle, so round-trip fidelity of the
   pickle path validates nothing on the critical path.

Running it anyway would have spent the time-box producing numbers about a
migration that will not happen. The five ids are recorded as `not-measured` with
this reason rather than silently skipped, per the plan's global constraint.

## What is therefore NOT known

Plainly: **there are no D9 numbers, and this audit is not the place they will come
from.** Specifically unknown —

- **Size.** Whether gzip-NWB is smaller than the current pickles, and by how much.
  No `.nwb` file was written, so no ratio exists.
- **Read performance.** The claim that NWB's lazy per-column / per-unit reads beat
  the pickle's all-or-nothing load is, in this repo, **asserted and not measured.**
  It is a reasonable expectation from the format's design, not a result. Nothing
  downstream should cite a speedup figure, because none was produced.
- **Round-trip fidelity.** Untested — and moot for the rebuild, which ingests raw.

The new repo will discover all of these during its own build, against its own
writer, which is the only place the numbers would have been valid anyway. They
should be measured there before any of them is quoted.

## Carry-forward 1 — the gzip-compression gotcha

> **Provenance: inherited from the plan's own round-2 review. This task did NOT
> execute or verify it.** It is carried forward here so that cancelling the spike
> does not lose it.

In `pynwb`/`hdmf`, attaching an `H5DataIO` compression wrapper **per row** of
`units/spike_times` does not work. `hdmf` consumes the wrapper element-wise, the
compression **silently drops**, and the write succeeds — yielding an uncompressed
file and a size figure that means nothing. There is no error and no warning; the
only symptom is a number that is quietly wrong.

The correct approach is to set the DataIO on the **concatenated column**:

```python
nwb.units.spike_times.set_data_io(H5DataIO, {"compression": "gzip"})
```

…and then **verify after writing**, with `h5py`, that the codec is really there:

```python
with h5py.File(out, "r") as h5:
    assert h5["units/spike_times"].compression == "gzip"
```

**Requirement on the new repo's NWB writer: never trust that compression was
applied — assert it post-write.** A dropped codec must surface as a failed
assertion, not as a plausible-looking size number. This generalises past gzip:
any storage option set through a wrapper object should be read back off the
written file and checked, because the failure mode is silence.

## Carry-forward 2 — `d9.keep_all_good` is a live blocker

**This one is not about storage format at all, and it does not go away with the
spike.** It sits on the critical path for the owner's raw re-ingest plan whichever
format is chosen.

**The question:** can every Kilosort-good unit be re-ingested **without** re-reading
the raw `.ap.bin` files?

**Why it is unanswered:** settling it requires reading a session's Kilosort output
tree (`spike_clusters.npy`) on the **`X:` mount**, which this audit is forbidden to
touch. It is a cheap read — one file, one session — but it is outside the audit's
permission boundary, so it stays open.

**Why it matters.** The current pickles are the product of an **irreversible
ingest-time QC gate**: they store spikes only for `good_and_stable` units. On
session `01072025` that is **108 units of 260 Kilosort-good**
(`docs/audit/01-constants.md:152`; ids `d1.frfloor.good_and_stable`,
`d1.frfloor.getgood_01hz`, `d1.frfloor.getgood_1hz`, `d1.frfloor.spread`). The
consequence is directional and hard: **unit counts under any new QC profile can
only FALL, never rise, without re-ingest.** The other 152 units are not filtered in
the pickle — they are absent from it. If the rebuild cannot recover them from the
Kilosort tree alone, the new repo either inherits the old gate or pays for a full
raw re-read.

- **Evidence:** `src/visdetect/core/ingest.py`
- **Settles with:** one pre-authorised lightweight `X:` read of a session's
  `spike_clusters.npy`, at sub-project 1
- **Blocking:** the raw re-ingest plan, format-independent

## Repo state — the scratch venv was not deleted

An earlier interrupted attempt left a scratch venv at
`data/cache/audit/nwbvenv/` (**1,204 dirs / 12,581 files / 321.7 MB**). This task
was to delete it. **It is still present.**

The tree was verified safe to delete — walked with
`os.lstat(...).st_file_attributes & FILE_ATTRIBUTE_REPARSE_POINT` (the test
`scripts/audit/d7_work_at_risk.py` uses; `is_symlink()` misses junctions on
Python 3.10): **0 reparse points inside**, target itself not a reparse point, its
resolved path identical to its literal path, and every ancestor up to the repo
root plain.

The delete was nonetheless blocked by the repo's own `PreToolUse` guard,
`.claude/hooks/guard_recursive_delete.ps1` — **a false positive with respect to
this target.** The guard always adds `.` (the cwd, here the repo root) to its
candidate list, scans it to depth 4, and denies on any junction found anywhere in
that scan. It reported:

```
.claude/worktrees/qc1-alignment/.superpowers  [Junction] -> <repo>/.superpowers
```

which is **outside** `data/cache/audit/nwbvenv`. The guard's suggested remedy —
delete the link first — was **deliberately not applied**: `qc1-alignment` is a live
registered worktree (`feature/early-lick-and-session-sorting`), and that junction
is its access path to `.superpowers`, the directory holding this audit's own plan
and briefs. Removing it to satisfy a false positive would be a real change to
another worktree to work around a scoping bug. The guard was not evaded either.

Two consequences worth the owner's attention:

1. **This is a standing condition, not a one-off.** Because `.` is always a
   candidate, *every* recursive delete run from this repo is currently denied while
   that junction exists — regardless of what is actually being deleted.
2. **The fix is in the guard, not the junction.** Scoping the scan to the paths the
   command actually names (dropping the unconditional `.`, or only using `.` when
   the command has no resolvable path argument) would restore the guard's precision
   without weakening it. The junction itself is legitimate and should stay.

The leftover venv is gitignored (`.gitignore:48`, `data/cache/*`), so it costs
321.7 MB of disk and nothing in the repo. It can be removed by the owner with:

```powershell
Remove-Item -Recurse -Force 'E:\python_analysis\git_repos\vis_detect_analysis_Sep2025\data\cache\audit\nwbvenv'
```
