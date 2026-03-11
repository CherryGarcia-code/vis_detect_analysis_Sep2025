#!/usr/bin/env bash
#
# run_ks4_retry_v5_task136.bash — Task 136 (W33, shk3)
#
# Previous attempt (v4) hit CUDA OOM during clustering: "Tried to allocate
# 17.05 GiB" on L40S (44.39 GB). Needs moderate threshold increase.
# Nearby shank 3 windows (W31/128, W32/132) succeeded at Th 10/9.
#
# Submit:
#   sbatch run_ks4_retry_v5_task136.bash
#
#SBATCH -J ks4_v5_136
#SBATCH -o Documents/ks4/logs_ks4_concat/ks4_retry_v5-%A_%a.out
#SBATCH -e Documents/ks4/logs_ks4_concat/ks4_retry_v5-%A_%a.err
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
echo "Retry V5: MODERATE thresholds (Th_universal=10, Th_learned=9)"
echo "Manifest: ${MANIFEST}"
echo "═══════════════════════════════════════════════════════════"

# Run KS4 with moderate thresholds
python "${SCRIPT_DIR}/run_kilosort4.py" ${SLURM_ARRAY_TASK_ID} \
    --manifest "${MANIFEST}" \
    --Th-universal 10 \
    --Th-learned 9
