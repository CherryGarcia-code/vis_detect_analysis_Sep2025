# Cluster TF-GLM replication (Khilkevich–Lohse, brain-wide)

SLURM-array pipeline that fits the corrected **`log2(TF)/0.25`-octave,
movement-controlled** per-neuron Poisson encoding GLM across many
`npx_converted` sessions/regions and reports, per region:

- the **movement-controlled** TF-responsive fraction (C2, p<0.01) — the faithful
  Khilkevich–Lohse number (compare vs their **5–45 %** and cortex>striatum);
- the **survival** test: of units flagged TF-responsive *without* movement
  control, what fraction **survive** movement control (genuine TF) vs **collapse**
  (movement confound).

Each unit is fit with **four** column-masked variants of one movement-inclusive
design (so the two C2 tests share the exact CV split):
`full_move`, `reduced_move (−TF)`, `full_nomove (−movement)`,
`reduced_nomove (−TF −movement)`.

## Files

| File | Role |
|------|------|
| `build_targets.py` | Enumerate `(session, region, unit-chunk)` tasks from `clusters.csv` (cheap; no spike load). Prints the `--array=1-N` range. |
| `tf_glm_cluster_task.py` | Array-task worker: fit one task's units, append rows to `results/task_<id>.csv` (resume-safe). |
| `tf_glm_array.sbatch` | SLURM array (partition `cpu`, conda `tfglm`, `PYTHONPATH=<stage>/code/src`). |
| `aggregate.py` | Concatenate `results/task_*.csv` → `master.csv` + `region_fractions.csv` + summary figure. |

## Paths

- **Stage (ceph):** `/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster`
  (Windows: `X:/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster`)
- **Data root (ceph):** `/ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted`
- **Conda env:** `/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/conda_envs/tfglm` (sklearn 1.7.2)

## One-time staging (run on Windows; X: == ceph)

```bash
STAGE="X:/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster"
mkdir -p "$STAGE/code/src" "$STAGE/code/scripts/tf_responsiveness/cluster" "$STAGE/logs" "$STAGE/results"
cp -r src/visdetect "$STAGE/code/src/"
cp scripts/tf_responsiveness/cluster/*.py "$STAGE/code/scripts/tf_responsiveness/cluster/"
cp scripts/tf_responsiveness/cluster/tf_glm_array.sbatch "$STAGE/code/scripts/tf_responsiveness/cluster/"
cp data/cache/tf_glm/cluster/targets_decisive.csv "$STAGE/targets.csv"
```

## Submit (on the cluster, from your HOME dir)

1. Set `--array=1-N` in `tf_glm_array.sbatch` to the count `build_targets.py`
   printed (decisive VISp+CP = **`--array=1-10`**).
2. `cd ~ && sbatch <STAGE>/code/scripts/tf_responsiveness/cluster/tf_glm_array.sbatch`
3. Monitor: `squeue -u $USER`; logs in `<STAGE>/logs/tfglm-*.out|err`.

Re-running the same array is safe — completed units are skipped (resume).

## Aggregate (on Windows after the array finishes)

```bash
py scripts/tf_responsiveness/cluster/aggregate.py --results "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster/results"
```

→ `master.csv`, `region_fractions.csv`, `tfglm_replication_summary.png`.

## Brain-wide sweep (after the decisive VISp+CP run validates)

```bash
py scripts/tf_responsiveness/cluster/build_targets.py \
  --scan-root "X:/.../npx_converted" --regions all --chunk 15 \
  --out data/cache/tf_glm/cluster/targets_brainwide.csv
# then re-stage targets_brainwide.csv as $STAGE/targets.csv, set --array=1-N, resubmit
```

## Runtime / resources

≈ 4 ridge-Poisson fits/unit; worst high-FR unit ≈ 650 s, typical ≈ 100–300 s.
Chunk 15 → ≈ 0.5–2.5 h/task. `--cpus-per-task=2`, `--mem=16G`, `-t 8h` are
generous; tune `%concurrency` in `--array` to your QOS. Pass `--no-both-models`
to the worker to fit only the movement-controlled model (half the fits, drops
the survival comparison).
