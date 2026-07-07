#!/usr/bin/env bash
#
# run_extract_array.sbatch <SUBJECT>     (submit as a SLURM ARRAY)
# ==============================================================
# Stage 3 (UnitMatch waveform extraction) ONLY, parallelized ONE SESSION PER
# ARRAY TASK. Run this AFTER run_prep_subject.sbatch has done TPrime + raw_to_pkl
# (stages 1-2) for the subject -- it reuses the session list that prep's pre-flight
# wrote to "<MANIFEST>.sessions".
#
# WHY: extraction is I/O-latency-bound -- ~1000 scattered random reads into the
# ~100 GB .ap.bin memmap per unit, ~100 ms each on CephFS => ~117 s/unit =>
# ~3.5 h/session. Sequentially over 16 sessions that's ~50 h (blows the wall
# limit; BG_040's first attempt 3183710 timed out at 12 h having done 3 sessions).
# One array task per session runs them all CONCURRENTLY on separate nodes, so wall
# time collapses to ~one session (~4 h). The extractor is resumable: any session
# with _extraction_complete.txt is skipped instantly, so re-submitting is safe.
#
# ── RUN ───────────────────────────────────────────────────────────────────────
#   # N = number of sessions in <MANIFEST>.sessions (BG_040 = 16). Use 0..N-1.
#   sbatch --array=0-15%8 ~/vis_detect/scripts/pipelines/tracking/slurm/run_extract_array.sbatch BG_040
#   # %8 = at most 8 sessions extracting at once (raise/lower to taste). Already-done
#   #      sessions finish in seconds, so the real work is the not-yet-done ones.
#   # then, when ALL tasks finish:
#   cd $CEPH/wEPhys/BG_040/unit_match && sbatch $CEPH/wEPhys/run_unitmatch_subject.sbatch BG_040
#
#SBATCH -J um_extract
#SBATCH -o um_extract-%A_%a.out
#SBATCH -e um_extract-%A_%a.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-12:00   # raised from 08:00: large sessions (e.g. BG_012's 191-unit days) extract in ~8h+ and were hitting the wall
#SBATCH -p cpu
#SBATCH --mem=96G

set -u
SUBJECT="${1:?usage: sbatch --array=0-<N-1>%8 run_extract_array.sbatch <SUBJECT>}"

# ===== same setup as run_prep_subject.sbatch =====
REPO="${REPO:-$HOME/vis_detect}"
CEPH="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect"
CONDA_ENV="${CONDA_ENV:-$CEPH/conda_envs/unitmatch}"
PKL_DIR="${PKL_DIR:-$CEPH/derivatives/pkls/$SUBJECT}"
NWORKERS="${NWORKERS:-16}"

PROC="$CEPH/wEPhys/$SUBJECT/Processed data"
UM_DIR="$CEPH/wEPhys/$SUBJECT/unit_match"
UM_INPUT="$UM_DIR/input"
MANIFEST="$UM_DIR/${SUBJECT}_um_extract_manifest.csv"
SESS_FILE="${MANIFEST}.sessions"

# env python via PATH prepend -- NOT `conda activate` (see run_prep_subject.sbatch).
if [ -x "$CONDA_ENV/bin/python" ]; then
  export PATH="$CONDA_ENV/bin:$PATH"; hash -r 2>/dev/null || true
else
  echo "WARN: $CONDA_ENV/bin/python not found; falling back to conda activate"
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "$REPO"

[ -s "$SESS_FILE" ] || { echo "!!! sessions file missing: $SESS_FILE -- run run_prep_subject.sbatch first"; exit 4; }
read -r -a SESS_ARR <<< "$(cat "$SESS_FILE")"
NSESS="${#SESS_ARR[@]}"
IDX="${SLURM_ARRAY_TASK_ID:?must be an array job: sbatch --array=0-$((NSESS-1))%8 run_extract_array.sbatch $SUBJECT}"
if [ "$IDX" -ge "$NSESS" ]; then
  echo "array index $IDX >= $NSESS sessions in $SESS_FILE -- nothing to do (lower --array upper bound)"; exit 0
fi
S="${SESS_ARR[$IDX]}"

echo "python -> $(command -v python)  ($(python --version 2>&1))"
echo "==== EXTRACT $SUBJECT  task $IDX/$((NSESS-1)) = $S  on $(hostname) @ $(date) ===="

# One session. Resumable: skips instantly if _extraction_complete.txt exists.
# The extractor exits nonzero if the session yields zero waveforms (fail-loud).
python scripts/analysis/prep_unitmatch_full_trial_waveforms.py --subject "$SUBJECT" \
  --processed-root "$PROC" --manifest "$MANIFEST" --output "$UM_INPUT" \
  --n_workers "$NWORKERS" --sessions "$S" \
  || { echo "!!! EXTRACT FAIL for $S"; exit 1; }

echo "==== DONE $S @ $(date) ===="
