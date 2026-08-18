# D9 — storage-format spike (NOT RUN)

The D9 spike was scoped to convert three sessions to NWB and measure what ADR-015
would have cited: on-disk size against the current pickles, three read patterns,
and pkl→NWB round-trip fidelity. **It was not run.** This document records why,
what is therefore unknown, and the two items that survive the cancellation and
must be carried into the new repo.

- Script: **none** — `scripts/audit/d9_nwb_spike.py` was never written
- Scratch venv: **not created by this task** (see *Repo state* at the end)
- `.nwb` files produced: **none**
- Measurement ids: `d9.*` in `docs/audit/measurements.csv` — the four spike
  measurements are **`not-measured`**; `d9.keep_all_good` is
  **`code-side YES; data-side not-measured`** (see carry-forward 2). All recorded
  via `record()` one-liners (`script = manual`)
- Also recorded from this task: `d5.tooling.delete_guard_falsepositive` — a
  tooling defect found while cleaning up, **for Task 15's register**

## What the spike would have measured

Three targets — smallest BG_049 session, BG_046 `01092025`, and a BG_012 twin —
each converted with direct `pynwb` in a throwaway venv, then:

| Id | Measurement |
|---|---|
| `d9.size_ratio` | gzip-NWB / pkl on-disk size ratio (< 1.0 anticipated) |
| `d9.readtimes` | `load_pkl` / `write_nwb` / `read_trials` / `read_one_unit` / `read_window_all_units` |
| `d9.roundtrip` | equality of `spike_times` (unit 0) and the trials columns after the round trip |
| `d9.compression` | the HDF5 codec actually present on `units/spike_times` after writing |
| `d9.keep_all_good` | whether every Kilosort-good unit can be re-ingested without re-reading raw `.ap.bin` — **partly answered in-repo after all; see carry-forward 2** |

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
migration that will not happen. All five ids are recorded with this reason rather
than silently skipped, per the plan's global constraint: the four spike
measurements as `not-measured`, and `d9.keep_all_good` at its true partial state
(code-side answered in-repo, data-side still open — carry-forward 2).

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

> **Note for Task 16.** The executive-summary brief asks for "the D9 numbers".
> There are none. Task 16 must instead cite the four not-measured `d9.*` spike ids
> plus `d9.keep_all_good`'s partial state, and the owner's 2026-08-13 format
> decision recorded here. The absence is a decision, not an omission, and should
> be presented as one. *(Corrected from "the five `not-measured` `d9.*` ids"
> 2026-08-18, Task 16 final wave, to match this document's own header —
> `d9.keep_all_good` is partial, not `not-measured`; the executive summary had
> already cited it correctly.)*

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

## Carry-forward 2 — `d9.keep_all_good`: code-side YES, data-side open

**This one is not about storage format at all, and it does not go away with the
spike.** It sits on the critical path for the owner's raw re-ingest plan whichever
format is chosen. But the open part is **narrower than the spike's framing
suggested**, and the difference matters for how sub-project 1 spends its one
pre-authorised `X:` read.

**The question:** can every Kilosort-good unit be re-ingested **without** re-reading
the raw `.ap.bin` files?

### The code-side answer is already YES — verified in-repo

This was settled by reading the repo, inside the audit's permitted scope. No `X:`
access was needed or used.

- **The flag already exists.** `build_session_from_raw` takes
  `keep_all_good: bool = False` (`src/visdetect/core/ingest.py:415`). Its `True`
  branch keeps `set(good_cluster_ids)` — every KS-good cluster — instead of the
  default `set(good_and_stable_ids)` (`ingest.py:492-495`). The capability is not
  hypothetical; it is a parameter with a live branch.
- **The ingest path never opens raw ephys.** Spike times come from
  `spike_times_sec_adj.npy` / `spike_times_sec.npy` / `spike_times.npy`, cluster
  assignments from `spike_clusters.npy` / `spike_clusters_ks.npy`
  (`ingest.py:243-267`), quality labels from `cluster_KSLabel.tsv` /
  `cluster_group.tsv` (`ingest.py:191-192`), and waveforms from `templates.npy`
  (`core/kilosort.py:42-49`). A grep across `src/visdetect/core/` finds **no
  `.ap.bin`, no `memmap`, and no `np.fromfile`** anywhere in the chain; the only
  binary read in the package core is the pickle loader at `core/session.py:196`.

So: re-ingesting all KS-good units requires the **Kilosort/Phy output tree only**,
and the code to do it is already written.

### What is genuinely still open

Only this: **are the Kilosort trees actually present and complete on `X:` for every
session?** The code can consume them; whether they all exist, for every session of
every subject, with the `.npy`/`.tsv` files the path requires, is a property of the
data store — not of the code — and that is the part the audit cannot see.

**That is what the one pre-authorised `X:` read should be spent on:** an existence
and completeness sweep of the per-session Kilosort directories, **not** a
re-derivation of the code-side answer, which this document has already established.

### Why it matters

The current pickles are the product of an **irreversible ingest-time QC gate**:
they store spikes only for `good_and_stable` units. On session `01072025` that is
**108 units of 260 Kilosort-good** (`docs/audit/01-constants.md:152`; ids
`d1.frfloor.good_and_stable`, `d1.frfloor.getgood_01hz`, `d1.frfloor.getgood_1hz`,
`d1.frfloor.spread`). The consequence is directional and hard: **unit counts under
any new QC profile can only FALL, never rise, without re-ingest.** The other 152
units are not filtered in the pickle — they are absent from it. Recovering them
means re-ingesting with `keep_all_good=True`, which the code supports, against
Kilosort trees whose completeness is the remaining unknown.

- **Evidence:** `src/visdetect/core/ingest.py:415` (flag), `:492-495` (branch),
  `:243-267` + `:191-192` and `core/kilosort.py:42-49` (inputs are `.npy`/`.tsv`
  only)
- **Settles with:** one pre-authorised `X:` sweep for Kilosort-tree
  presence/completeness per session, at sub-project 1
- **Blocking:** the raw re-ingest plan, format-independent

## Repo state — the scratch venv was not deleted

An earlier interrupted attempt left a scratch venv at
`data/cache/audit/nwbvenv/` (**1,204 dirs / 12,581 files / 321.7 MiB**, i.e.
337.4 MB decimal / 337,352,165 bytes). This task was to delete it. **It is still
present.**

The tree was verified safe to delete — walked with
`os.lstat(...).st_file_attributes & FILE_ATTRIBUTE_REPARSE_POINT` (the test
`scripts/audit/d7_work_at_risk.py` uses; `is_symlink()` misses junctions on
Python 3.10): **0 reparse points inside**, target itself not a reparse point, its
resolved path identical to its literal path, and every ancestor up to the repo
root plain.

The delete was nonetheless blocked by the repo's own `PreToolUse` guard,
`.claude/hooks/guard_recursive_delete.ps1` — **a false positive with respect to
this target.** `Get-Candidates` unconditionally adds `.` (the cwd, here the repo
root) to its candidate list (`:85`), each candidate is scanned to
`$MAX_DEPTH = 4` (`:37`), and the hook denies whenever the hit list is non-empty
(`:170`) regardless of which candidate produced the hit. It reported:

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
   command actually names — dropping the unconditional `.` at `:85`, or falling
   back to `.` only when no path argument resolves — would restore the guard's
   precision without weakening it: a command that names its target would then be
   judged on that target alone. The junction itself is legitimate and should stay.

> **This is a tooling defect, not a D9 finding, and it must not fall through the
> gap between this document and sub-project 1.** It is recorded as
> `d5.tooling.delete_guard_falsepositive` in `measurements.csv` (domain D5,
> value `blocked-all-recursive-deletes`, evidence
> `.claude/hooks/guard_recursive_delete.ps1:85`) so it has an id to cite.
> **Task 15 must carry it into the known-defect register** — direction of effect:
> every recursive delete in the repo is denied while the depth-4 junction exists,
> which pushes an operator toward either the guard's own suggested remedy
> (deleting a live worktree's junction — the 2026-06-07 hazard shape) or toward
> rewording the command to dodge the verb regex. Both are worse than the false
> positive itself, which is what makes this worth fixing before sub-project 1
> rather than after.

The leftover venv is gitignored (`.gitignore:48`, `data/cache/*`), so it costs
321.7 MiB of disk and nothing in the repo. It can be removed by the owner with:

```powershell
Remove-Item -Recurse -Force 'E:\python_analysis\git_repos\vis_detect_analysis_Sep2025\data\cache\audit\nwbvenv'
```
