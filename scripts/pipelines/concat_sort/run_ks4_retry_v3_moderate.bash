#!/usr/bin/env bash
#
# run_ks4_retry_v3_moderate.bash — Re-run 10 tasks that were infra failures
# originally and now revealed as OOM (never had raised thresholds).
#
# These tasks ran with default Th_universal=9, Th_learned=8 and OOM'd.
# Now retry with moderate threshold increase: Th_universal=10, Th_learned=9.
#
# Tasks: 117,119,123,125,127,128,129,132,133,135
#   W09: (none here)
#   W29: shk0(117), shk2(119)
#   W30: shk2(123)
#   W31: shk0(125), shk2(127), shk3(128)
#   W32: shk0(129), shk3(132)
#   W33: shk0(133), shk2(135)
#
# Submit:
#   sbatch run_ks4_retry_v3_moderate.bash
#
#SBATCH -J ks4_v3mod
#SBATCH -o Documents/ks4/logs_ks4_concat/ks4_retry_v3mod-%A_%a.out
#SBATCH -e Documents/ks4/logs_ks4_concat/ks4_retry_v3mod-%A_%a.err
#SBATCH -N 1
#SBATCH -t 0-17:59
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Increased to 192G for task 128 which had system RAM OOM
#SBATCH --mem=192G
#SBATCH --array=117,119,123,125,127,128,129,132,133,135%6
# Only L40S (48GB VRAM) — avoid A100-40GB since many of these OOM'd there
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
echo "Retry V3: MODERATE thresholds (Th_universal=10, Th_learned=9)"
echo "Manifest: ${MANIFEST}"
echo "═══════════════════════════════════════════════════════════"

# Run KS4 with raised thresholds
python "${SCRIPT_DIR}/run_kilosort4.py" ${SLURM_ARRAY_TASK_ID} \
    --manifest "${MANIFEST}" \
    --Th-universal 10 \
    --Th-learned 9
