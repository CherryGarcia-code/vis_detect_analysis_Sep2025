#!/usr/bin/env bash
#
# BG-mouse TF-GLM sweep -- SLURM array job (pkl path, our own mice).
#
# One array task = one (subject, session, unit-chunk) row of targets_bg.csv.
# Each unit is fit with the corrected pulse-criterion GLM (log2-TF + linear
# control); see cluster_bg/tf_glm_bg_task.py. Reads Session pkls staged to ceph
# (NEVER the Samba gateway); writes per-task CSV + pulse-PETH npz.
#
# SUBMIT (from your HOME dir, after staging pkls+code+targets):
#   cd ~
#   # set --array=1-N to the count build_bg_targets.py printed, then:
#   sbatch /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster/bg_mice/code/scripts/tf_responsiveness/cluster_bg/tf_glm_bg_array.sbatch
#
#SBATCH -J tfglm_bg
#SBATCH -o /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster/bg_mice/logs/tfglmbg-%A_%a.out
#SBATCH -e /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster/bg_mice/logs/tfglmbg-%A_%a.err
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --ntasks=1
# MINIMAL footprint to BACKFILL into idle CPUs while pending on 'Priority'
# (idle nodes reserved for higher-priority jobs; we only get in via short small
# gaps). 1 core doubles the idle slots we fit (the Poisson solver is ~single-
# threaded, so 1 vs 2 cores barely changes fit time); 8G covers peak (~6G on the
# 853-trial sessions); 1.5h covers the largest chunk (~35 units, ~80 min at 1
# core) with margin. Tasks are resumable, so a rare slow-node kill just re-runs
# and skips units already in its CSV.
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 0-01:30
#SBATCH --array=1-20%80

set -uo pipefail

STAGE=/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster
BG=$STAGE/bg_mice
DATA_ROOT=$BG/bg_pkls
# Per-subject sub-jobs so each runs/aggregates/resumes independently (and the
# priority mouse can go first). Override at submit:
#   sbatch --export=ALL,SUBJ=046 --array=1-368%80 ...   # BG_046 (46 sess) FIRST
#   sbatch --export=ALL,SUBJ=031 --array=1-344%80 ...   # BG_031 (43 sess, resumes 64 done)
#   sbatch --export=ALL,SUBJ=039 --array=1-256%80 ...   # BG_039 (32 sess)
SUBJ=${SUBJ:-046}
TARGETS=$BG/targets_bg_${SUBJ}.csv
RESULTS=$BG/results_bg_${SUBJ}
# The shared tfglm conda env lives under wEPhys/ (one level ABOVE tf_glm_cluster/),
# same path the MoHa sbatch used -- NOT under tf_glm_cluster/.
ENV=/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/conda_envs/tfglm

# ---- conda ----
module load miniconda/23.10.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

# The env lives on ceph; under storage degradation `conda activate` can silently
# fail, leaving a python without numpy. Verify, retry once, then abort with a
# CLEAR message (not a cryptic numpy traceback).
if ! python -c "import numpy" 2>/dev/null; then
    echo "WARN task ${SLURM_ARRAY_TASK_ID}: env not ready after conda activate on $(hostname); retrying in 30s (likely transient ceph/env-load)"
    sleep 30
    conda activate "$ENV"
fi
if ! python -c "import numpy" 2>/dev/null; then
    echo "FATAL task ${SLURM_ARRAY_TASK_ID}: conda env failed to activate (numpy missing) on $(hostname) -- STORAGE/env-load failure (ceph), NOT the analysis code. Re-submit when storage is healthy."
    exit 1
fi

mkdir -p "$BG/logs" "$RESULTS"
echo "SUBJ=$SUBJ | targets=$TARGETS | results=$RESULTS"

# Pin BLAS/OpenMP to the task's CPU allocation.
export PYTHONPATH="$BG/code/src"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

echo "task ${SLURM_ARRAY_TASK_ID} on $(hostname) | $(date)"

# Fail-fast import check (lean stubbed env: root + analysis + CORE inits empty).
# The BG worker additionally imports visdetect.core.session for load_session.
python -c "import visdetect.core.session, visdetect.analysis.tf_glm, visdetect.analysis.tf_glm_data; print('import OK')" || {
    echo "IMPORT CHECK FAILED on $(hostname) -- aborting task ${SLURM_ARRAY_TASK_ID}"
    exit 1
}

python "$BG/code/scripts/tf_responsiveness/cluster_bg/tf_glm_bg_task.py" \
    --targets "$TARGETS" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --data-root "$DATA_ROOT" \
    --out-dir "$RESULTS"
rc=$?
echo "task ${SLURM_ARRAY_TASK_ID} finished rc=$rc | $(date)"
exit $rc
