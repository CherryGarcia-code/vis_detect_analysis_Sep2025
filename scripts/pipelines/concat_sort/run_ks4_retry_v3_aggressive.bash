#!/usr/bin/env bash
#
# run_ks4_retry_v3_aggressive.bash — Re-run 5 tasks that OOM'd even with
# raised thresholds (Th_universal=10, Th_learned=9).
#
# These are the stubbornly-high-spike-count windows. Retry with more
# aggressive thresholds: Th_universal=11, Th_learned=10.
#
# Tasks: 39,40,111,115,116
#   W09: shk2(39), shk3(40)
#   W27: shk2(111)
#   W28: shk2(115), shk3(116)
#
# Submit:
#   sbatch run_ks4_retry_v3_aggressive.bash
#
#SBATCH -J ks4_v3agg
#SBATCH -o Documents/ks4/logs_ks4_concat/ks4_retry_v3agg-%A_%a.out
#SBATCH -e Documents/ks4/logs_ks4_concat/ks4_retry_v3agg-%A_%a.err
#SBATCH -N 1
#SBATCH -t 0-17:59
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --array=39,40,111,115,116%4
# Only L40S (48GB VRAM) — best chance with largest GPU
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
echo "Retry V3: AGGRESSIVE thresholds (Th_universal=11, Th_learned=10)"
echo "Manifest: ${MANIFEST}"
echo "═══════════════════════════════════════════════════════════"

# Run KS4 with aggressively raised thresholds
python "${SCRIPT_DIR}/run_kilosort4.py" ${SLURM_ARRAY_TASK_ID} \
    --manifest "${MANIFEST}" \
    --Th-universal 11 \
    --Th-learned 10
