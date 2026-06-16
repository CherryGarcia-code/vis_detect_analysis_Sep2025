# Running UnitMatch for BG_031 / BG_038 / BG_039

Replicates the BG_046 UnitMatch pipeline for the three additional subjects.
Two stages: (1) **raw-waveform extraction** locally (reads the raw `.ap.bin`
over the `X:` mount), then (2) **UnitMatch** on the ceph SLURM cluster (one
batch over all of a subject's sessions, exactly like BG_046).

`X:\public\projects\BeJG_20230130_VisDetect\` **is** the cluster filesystem
`/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/` — so the data the
extraction reads here is the same data the cluster job reads.

## Session inventory (driven from the successfully-built PKLs)

Every PKL maps to valid Kilosort output + a decompressed `.ap.bin` on X:
(verified pre-flight, all runnable):

| Subject | PKL sessions | KS layout |
|---------|--------------|-----------|
| BG_031  | 43           | 8 spikes-in-probe-dir, 35 in `kilosort4/` subfolder |
| BG_038  | 43           | all in `kilosort4/` subfolder |
| BG_039  | 32           | all in `kilosort4/` subfolder |

The newer subjects store KS4 output in a `…_imec0/kilosort4/` subfolder (BG_046
put it directly in `…_imec0/`). The extraction now resolves both layouts
(mirrors `visdetect.core.ingest`), reading the raw `.ap.bin`/`.ap.meta` from the
probe (`imec0`) folder and the spike `.npy`s from `kilosort4/` when present.

Per-subject input lists were generated from the PKLs:
`data/<SUBJECT>_um_extract_manifest.csv` (columns `session_name,path`).

---

## Stage 1 — Raw-waveform extraction (local, reads X:)

Produces, per session, `data/unit_match/input/<SUBJECT>/<session>/RawWaveforms/UnitN_RawSpikes.npy`
(shape `(time, channels, 2)`, CV-split **raw averaged** spikes — not KS
templates) plus the channel/label metadata UnitMatch needs.

```bash
# from the repo root, on the Windows box that has the X: mount
for SUB in BG_031 BG_038 BG_039; do
  py scripts/analysis/prep_unitmatch_full_trial_waveforms.py \
     --subject  "$SUB" \
     --manifest "data/${SUB}_um_extract_manifest.csv" \
     --output   "data/unit_match/input/$SUB" \
     --n_workers 4
done
```

This is I/O-bound (each `.ap.bin` is tens-to-hundreds of GB, read over SMB), so
it runs for hours — launch it detached / overnight. A single session, or a
re-run of a few, can be limited with `--sessions <name> [<name> …]`.

### Validate it's raw, not templates

```bash
for SUB in BG_031 BG_038 BG_039; do
  py scripts/pipelines/tracking/validate_waveforms.py --input "data/unit_match/input/$SUB"
done
```
Expect the **RAW-AVERAGED** verdict (real baseline noise floor + noise-like CV
half differences). A *TEMPLATE-like* verdict means re-extract.

---

## Stage 2 — Stage to the cluster (ceph)

The extraction output lives on the local `e:` drive; copy it (and the updated
runner) onto ceph under each subject's `unit_match/` dir. `X:` is the ceph
mount, so this is a plain copy from the Windows box:

```bash
CEPH="X:/public/projects/BeJG_20230130_VisDetect"
for SUB in BG_031 BG_038 BG_039; do
  mkdir -p "$CEPH/wEPhys/$SUB/unit_match/logs"
  cp -r "data/unit_match/input/$SUB"           "$CEPH/wEPhys/$SUB/unit_match/input"
  cp    scripts/pipelines/tracking/run_unitmatch_all.py \
        "$CEPH/wEPhys/$SUB/unit_match/run_unitmatch_all.py"
done
```

(Copying many small `.npy` files over SMB is slow but harmless; `robocopy` is a
faster alternative on Windows.) UnitMatchPy itself is already installed in the
ceph env `conda_envs/unitmatch`, so it does not need staging.

---

## Stage 3 — Run UnitMatch on the cluster (SLURM)

The launcher `scripts/pipelines/tracking/slurm/run_unitmatch_subject.sbatch` is
a parametrized clone of BG_046's `run_unitmatch_cluster.sh`. Stage it to ceph
(once) and submit one job per subject from that subject's `unit_match/` dir so
the logs land beside the output:

```bash
# on the cluster login node
CEPH=/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect
cp run_unitmatch_subject.sbatch "$CEPH/wEPhys/"      # or any shared path

for SUB in BG_031 BG_038 BG_039; do
  cd "$CEPH/wEPhys/$SUB/unit_match"
  sbatch "$CEPH/wEPhys/run_unitmatch_subject.sbatch" "$SUB"
done
```

Each job runs **all of that subject's sessions in ONE batch**
(`--batch-size N --overlap 0`), the same "all at once" run you did for BG_046.

### RAM sizing
The job defaults to the `a100` partition with `--mem=400G` (what BG_046's
~6679-unit single-batch run needed). The new subjects have fewer sessions; once
a job starts, its log prints
`n_units=…  ((t,N,N) array ~XX GB)` — if that is comfortably under ~200 GB you
can resubmit on the lighter `gpu` partition (see the commented `#SBATCH` block
in the sbatch). If a subject ever OOMs, fall back to overlapping batches by
editing the `--batch-size/--overlap` in the sbatch (e.g. `--batch-size 28
--overlap 14`), which UnitMatch then reconciles via union-find.

### Output (per subject, `…/unit_match/output/all_sessions/`)
- `cell_registry.csv` — global UID × session wide table (KS cluster ids)
- `unit_index.csv` — `(session, ks_unit_id) → global_uid`
- `run_summary.json` — params + track-span stats
- `output_prob_matrix.npy` (under `batch0/`)

---

## What changed in the repo (branch `feature/unitmatch-multisubject`)

- `scripts/analysis/prep_unitmatch_full_trial_waveforms.py` — added `--subject`;
  kilosort4-aware KS-output resolution; exact-folder match first (glob fallback
  keeps BG_046's bare-date manifest working); `.ap.meta` read from the probe
  folder.
- `scripts/pipelines/tracking/run_unitmatch_all.py` — `parse_session_date` now
  tolerant of full-stem dir names (`BG_031_01042025`, 6-digit `BG_031_050325`,
  `…_v2`/`…_b` suffixes); BG_046 bare dates still parse.
- `scripts/pipelines/tracking/slurm/run_unitmatch_subject.sbatch` — new
  subject-parametrized SLURM launcher.
- `data/<SUBJECT>_um_extract_manifest.csv` — generated input lists (gitignored).
