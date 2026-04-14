#!/usr/bin/env bash
#
# retry_failed_ks4_jobs.bash — Retry only the failed concat-sort jobs
#
# Re-submits the 65 jobs that failed with CUDA busy/unavailable errors.
# Uses the same corrected parameters (nblocks=1, drift_smoothing=[3,3,3])
# but with reduced concurrency to avoid GPU resource contention.
#
# Before submitting:
#   1. Verify cluster GPU availability and queue status
#   2. Consider running during off-peak hours for better GPU access
#
# Submit:
#   sbatch retry_failed_ks4_jobs.bash
#
#SBATCH -J ks4_retry
#SBATCH -o Documents/ks4/logs_ks4_retry/ks4_retry-%A_%a.out
#SBATCH -e Documents/ks4/logs_ks4_retry/ks4_retry-%A_%a.err
#SBATCH -N 1
#SBATCH -t 0-17:59
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
# Reduced concurrency: only 3 jobs at once (was %6)
#SBATCH --array=1-65%3
# Target GPUs with >=40 GB VRAM (L40S, A100-40GB, H100)
#SBATCH --nodelist=gpu-sr675-31,gpu-sr670-20,gpu-sr670-21,gpu-sr670-22,gpu-sr670-23

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Path to ks4_run_manifest.json (Linux/ceph path)
MANIFEST="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/ks4_runs/ks4_run_manifest.json"

# Directory containing run_kilosort4.py
SCRIPT_DIR="$HOME/Documents/ks4"

# KS4 conda environment
CONDA_ENV="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/conda_envs/kilosort4"

# List of failed job IDs (1-based, from log analysis)
FAILED_JOBS=(15 33 38 39 40 42 43 44 46 57 62 66 76 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 119 120 121 122 123 124 125 126 127 128 129 130 132 133 134 135 136)

# ═══════════════════════════════════════════════════════════════════════
# CORRECTED PARAMETERS (same as successful jobs)
# ═══════════════════════════════════════════════════════════════════════
NBLOCKS=1
DRIFT_SMOOTH="3.0 3.0 3.0"

# ═══════════════════════════════════════════════════════════════════════

# Reduce GPU memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load modules
module load miniconda/23.10.0
module load cuda/11.8

# Source conda shell functions
source $(conda info --base)/etc/profile.d/conda.sh

# Activate KS4 environment
conda activate "${CONDA_ENV}"

# Create logs directory
mkdir -p ~/Documents/ks4/logs_ks4_retry

# Map array task ID to actual failed job ID
if [ ${SLURM_ARRAY_TASK_ID} -le ${#FAILED_JOBS[@]} ]; then
    ACTUAL_JOB_ID=${FAILED_JOBS[$((SLURM_ARRAY_TASK_ID - 1))]}
else
    echo "ERROR: Array task ID ${SLURM_ARRAY_TASK_ID} exceeds failed jobs array size"
    exit 1
fi

# ── Clear old completion marker and KS4 outputs for this job ──
RUN_DIR=$(python -c "
import json, sys
with open('${MANIFEST}') as f:
    m = json.load(f)
idx = ${ACTUAL_JOB_ID} - 1
if idx < len(m['windows']):
    d = m['windows'][idx]['run_dir']
    # Translate Windows path to Linux
    d = d.replace('X:/public/', '/ceph/mrsic_flogel/public/')
    d = d.replace('X:\\\\public\\\\', '/ceph/mrsic_flogel/public/')
    d = d.replace('\\\\', '/')
    print(d)
")

if [ -n "${RUN_DIR}" ] && [ -d "${RUN_DIR}" ]; then
    echo "Clearing old KS4 outputs in: ${RUN_DIR}"
    rm -f "${RUN_DIR}/ks4_complete.txt"
    # Remove old KS4 output files (preserves params.py and probe files)
    rm -f "${RUN_DIR}"/spike_times.npy "${RUN_DIR}"/spike_clusters.npy
    rm -f "${RUN_DIR}"/spike_templates.npy "${RUN_DIR}"/templates.npy
    rm -f "${RUN_DIR}"/templates_ind.npy "${RUN_DIR}"/amplitudes.npy
    rm -f "${RUN_DIR}"/channel_map.npy "${RUN_DIR}"/channel_positions.npy
    rm -f "${RUN_DIR}"/cluster_*.tsv "${RUN_DIR}"/ops.npy
    rm -f "${RUN_DIR}"/spike_positions.npy "${RUN_DIR}"/spike_datasets.npy
    rm -f "${RUN_DIR}"/whitening_mat.npy "${RUN_DIR}"/whitening_mat_inv.npy
    rm -f "${RUN_DIR}"/pc_features.npy "${RUN_DIR}"/pc_feature_ind.npy
    rm -f "${RUN_DIR}"/similar_templates.npy "${RUN_DIR}"/template_features.npy
    rm -rf "${RUN_DIR}"/drift_*.png "${RUN_DIR}"/probe_*.png
    rm -rf "${RUN_DIR}"/summary_*.png "${RUN_DIR}"/*.png
fi

# Print job info
echo "═══════════════════════════════════════════════════════════"
echo "RETRY Job Array ${SLURM_ARRAY_TASK_ID}/${#FAILED_JOBS[@]} → Actual Job ${ACTUAL_JOB_ID}"
echo "SLURM Job ${SLURM_JOB_ID}, Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Manifest: ${MANIFEST}"
echo "Parameters: nblocks=${NBLOCKS}, drift_smoothing=${DRIFT_SMOOTH}"
echo "═══════════════════════════════════════════════════════════"

# Run KS4 with corrected parameters
python "${SCRIPT_DIR}/run_kilosort4.py" ${ACTUAL_JOB_ID} \
    --manifest "${MANIFEST}" \
    --nblocks ${NBLOCKS} \
    --drift-smoothing ${DRIFT_SMOOTH}

echo "Retry job ${ACTUAL_JOB_ID} completed with exit code $?"