#!/usr/bin/env bash
#
# run_ks4_concat.bash — SLURM array job for the concat-sort KS4 pipeline (BG_046)
#
# Sorts per-shank concatenated windows with Kilosort 4.
#
# Before submitting:
#   1. Run split_by_shank.py        → shank_split_manifest.json
#   2. Run build_concat_windows.py  → ks4_run_manifest.json  (prints total N runs)
#   3. Set --array=1-N below where N = total runs from step 2
#   4. Set MANIFEST below to the Linux path of ks4_run_manifest.json
#   5. Ensure the concat_sort scripts are in SCRIPT_DIR
#
# Submit:
#   sbatch run_ks4_concat.bash
#
#SBATCH -J ks4_concat
#SBATCH -o Documents/ks4/logs_ks4_concat/ks4_concat-%A_%a.out
#SBATCH -e Documents/ks4/logs_ks4_concat/ks4_concat-%A_%a.err
#SBATCH -N 1
#SBATCH -t 0-17:59
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=1-136%6
# Only target GPUs with enough VRAM for clustering step (>=40 GB)
#SBATCH --nodelist=gpu-sr675-31,gpu-sr670-20,gpu-sr670-21,gpu-sr670-22,gpu-sr670-23

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these for your setup
# ═══════════════════════════════════════════════════════════════════════

# Path to ks4_run_manifest.json (Linux/ceph path)
MANIFEST="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/ks4_runs/ks4_run_manifest.json"

# Directory containing run_kilosort4.py (copy it here from the repo)
SCRIPT_DIR="$HOME/Documents/ks4"

# KS4 conda environment
CONDA_ENV="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/conda_envs/kilosort4"

# ═══════════════════════════════════════════════════════════════════════

# Reduce GPU memory fragmentation — crucial for large concat sorts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load modules
module load miniconda/23.10.0
module load cuda/11.8

# Source conda shell functions
source $(conda info --base)/etc/profile.d/conda.sh

# Activate KS4 environment
conda activate "${CONDA_ENV}"

# Create logs directory
mkdir -p ~/Documents/ks4/logs_ks4_concat

# Print job info
echo "═══════════════════════════════════════════════════════════"
echo "SLURM Job ${SLURM_JOB_ID}, Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Manifest: ${MANIFEST}"
echo "═══════════════════════════════════════════════════════════"

# Run KS4 for this array task (1-based index)
python "${SCRIPT_DIR}/run_kilosort4.py" ${SLURM_ARRAY_TASK_ID} --manifest "${MANIFEST}"
