#!/usr/bin/env bash
#
# run_ks4_retry_v4_task136.bash — Single remaining task: 136 (W33, shk3)
#
# This task was system-RAM OOM-killed at 94.7% through template matching
# (batch 30024/31721) on an A100-40GB with --mem=128G (infra retry 2466578).
# It DOES NOT need raised thresholds — it passed clustering fine.
# Just needs more RAM (192G) and the L40S for headroom.
#
# Submit:
#   sbatch run_ks4_retry_v4_task136.bash
#
#SBATCH -J ks4_v4_136
#SBATCH -o Documents/ks4/logs_ks4_concat/ks4_retry_v4-%A_%a.out
#SBATCH -e Documents/ks4/logs_ks4_concat/ks4_retry_v4-%A_%a.err
#SBATCH -N 1
#SBATCH -t 0-17:59
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --array=136
#SBATCH --nodelist=gpu-sr675-31

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
MANIFEST="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/ks4_runs/ks4_run_manifest.json"
SCRIPT_DIR="$HOME/Documents/ks4"
CONDA_ENV="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/conda_envs/kilosort4"

# ═══════════════════════════════════════════════════════════════════════

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Activate environment WITHOUT module system ──────────────────────
export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# Create logs directory
mkdir -p ~/Documents/ks4/logs_ks4_concat

# Print job info
echo "═══════════════════════════════════════════════════════════"
echo "SLURM Job ${SLURM_JOB_ID}, Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Retry V4: Default thresholds, 192G RAM, L40S"
echo "Manifest: ${MANIFEST}"
echo "═══════════════════════════════════════════════════════════"

# Run KS4 with default thresholds (this task only needs more RAM, not higher Th)
python "${SCRIPT_DIR}/run_kilosort4.py" ${SLURM_ARRAY_TASK_ID} \
    --manifest "${MANIFEST}"
