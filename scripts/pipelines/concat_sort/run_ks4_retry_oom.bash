#!/usr/bin/env bash
#
# run_ks4_retry_oom.bash — Re-run 21 tasks that CUDA OOM'd during clustering.
#
# These tasks OOM during clustering_qr.py → assign_iclust() or kmeans_plusplus()
# because too many spikes are loaded onto the GPU at once.
#
# Fix: Raise detection thresholds Th_universal (9→10) and Th_learned (8→9)
# to reduce the total number of detected spikes by ~20-30%.
#
# OOM tasks (1-based): 15,19,20,24,38,39,40,42,43,93,94,95,101,108,109,110,111,112,114,115,116
#
# Submit:
#   sbatch run_ks4_retry_oom.bash
#
#SBATCH -J ks4_oom
#SBATCH -o Documents/ks4/logs_ks4_concat/ks4_retry_oom-%A_%a.out
#SBATCH -e Documents/ks4/logs_ks4_concat/ks4_retry_oom-%A_%a.err
#SBATCH -N 1
#SBATCH -t 0-17:59
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
# 21 specific task IDs that OOM'd during clustering
#SBATCH --array=15,19,20,24,38,39,40,42,43,93,94,95,101,108,109,110,111,112,114,115,116%6
# Only target GPUs with enough VRAM (>=40 GB)
#SBATCH --nodelist=gpu-sr675-31,gpu-sr670-20,gpu-sr670-21,gpu-sr670-22,gpu-sr670-23

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
MANIFEST="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/ks4_runs/ks4_run_manifest.json"
SCRIPT_DIR="$HOME/Documents/ks4"
CONDA_ENV="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/conda_envs/kilosort4"

# ═══════════════════════════════════════════════════════════════════════

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Activate environment WITHOUT module system ──────────────────────
# The cluster module system (modulecmd.tcl) is currently broken on GPU
# nodes. Bypass it entirely: the conda env has Python + PyTorch +
# bundled CUDA runtime (pip nvidia-* packages), so no system CUDA needed.
export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"

# Ensure linker can find any libs in the env
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# Create logs directory
mkdir -p ~/Documents/ks4/logs_ks4_concat

# Print job info
echo "═══════════════════════════════════════════════════════════"
echo "SLURM Job ${SLURM_JOB_ID}, Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Retry: OOM — raised thresholds (Th_universal=10, Th_learned=9)"
echo "Manifest: ${MANIFEST}"
echo "═══════════════════════════════════════════════════════════"

# Run KS4 with raised detection thresholds to reduce clustering memory
python "${SCRIPT_DIR}/run_kilosort4.py" ${SLURM_ARRAY_TASK_ID} \
    --manifest "${MANIFEST}" \
    --Th-universal 10 \
    --Th-learned 9
