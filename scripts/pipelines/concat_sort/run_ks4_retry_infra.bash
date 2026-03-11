#!/usr/bin/env bash
#
# run_ks4_retry_infra.bash — Re-run 20 tasks that failed due to conda/python
# not found (infrastructure error on re-submission nodes).
#
# Tasks 117-136 (windows 29-33, all 4 shanks) failed with:
#   "conda: command not found" on gpu-sr670-21 and gpu-sr675-31
# This is a pure infrastructure issue — same parameters as the original run.
#
# Submit:
#   sbatch run_ks4_retry_infra.bash
#
#SBATCH -J ks4_infra
#SBATCH -o Documents/ks4/logs_ks4_concat/ks4_retry_infra-%A_%a.out
#SBATCH -e Documents/ks4/logs_ks4_concat/ks4_retry_infra-%A_%a.err
#SBATCH -N 1
#SBATCH -t 0-17:59
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
# Tasks 117-136 (20 tasks, windows 29-33 × 4 shanks)
#SBATCH --array=117-136%6
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
echo "Retry: INFRA (same thresholds as original run)"
echo "Manifest: ${MANIFEST}"
echo "═══════════════════════════════════════════════════════════"

# Run KS4 — same parameters as original (default thresholds)
python "${SCRIPT_DIR}/run_kilosort4.py" ${SLURM_ARRAY_TASK_ID} --manifest "${MANIFEST}"
