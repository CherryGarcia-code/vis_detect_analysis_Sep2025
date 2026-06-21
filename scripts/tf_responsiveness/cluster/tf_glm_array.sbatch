#!/usr/bin/env bash
#
# Brain-wide Khilkevich-Lohse TF-GLM replication -- SLURM array job.
#
# One array task = one (session, region, unit-chunk) row of targets.csv. Each
# unit is fit with the corrected (log2(TF)/0.25-octave) movement-controlled
# Poisson encoding GLM; BOTH the movement-controlled and the no-movement
# TF-ablation C2 are recorded (see tf_glm_cluster_task.py).
#
# SUBMIT (from your HOME dir, after staging + writing targets.csv):
#   cd ~
#   sbatch /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster/code/scripts/tf_responsiveness/cluster/tf_glm_array.sbatch
#
# EDIT --array=1-N below to match the task count build_targets.py printed.
# The %20 throttles to 20 concurrent tasks; raise/lower per your QOS.
#
#SBATCH -J tfglm
#SBATCH -o /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster/logs/tfglm-%A_%a.out
#SBATCH -e /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster/logs/tfglm-%A_%a.err
#SBATCH -N 1
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 0-08:00
#SBATCH --array=1-10%20

set -uo pipefail

STAGE=/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster
DATA_ROOT=/ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx_converted
ENV=/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/conda_envs/tfglm

# ---- conda ----
module load miniconda/23.10.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

mkdir -p "$STAGE/logs" "$STAGE/results"

# Pin BLAS/OpenMP threads to the task's CPU allocation (the Poisson solver's
# matrix ops would otherwise oversubscribe the node).
export PYTHONPATH="$STAGE/code/src"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

echo "task ${SLURM_ARRAY_TASK_ID} on $(hostname) | $(date)"

# Fail-fast import sanity check: if the lean env can't import the leaf modules,
# say so loudly and exit non-zero (do NOT silently 'finish').
python -c "import visdetect.analysis.tf_glm, visdetect.analysis.tf_glm_data; print('import OK')" || {
    echo "IMPORT CHECK FAILED on $(hostname) — aborting task ${SLURM_ARRAY_TASK_ID}"
    exit 1
}

python "$STAGE/code/scripts/tf_responsiveness/cluster/tf_glm_cluster_task.py" \
    --targets "$STAGE/targets.csv" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --data-root "$DATA_ROOT" \
    --out-dir "$STAGE/results"
rc=$?
echo "task ${SLURM_ARRAY_TASK_ID} finished rc=$rc | $(date)"
exit $rc
