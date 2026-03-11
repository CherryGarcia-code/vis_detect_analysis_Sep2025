#!/usr/bin/env bash
#
# run_split_shank.bash — SLURM array job to split NP2.0 AP binaries by shank (BG_046)
#
# Runs 38 sessions in parallel using SLURM array tasks.
# Each task processes one session (~1.5 hrs on /ceph).
# Sessions that already have complete output are auto-skipped.
# No GPU needed — purely I/O bound.
#
# Before submitting:
#   1. Copy this script + split_by_shank.py to ~/Documents/ks4/
#   2. Copy learning_session_selection.json to ~/Documents/ks4/
#   3. Verify paths in CONFIGURATION below
#
# Submit:
#   sbatch run_split_shank.bash
#
# After all jobs complete, merge manifests:
#   # (load modules + activate env first, or submit as another sbatch)
#   python ~/Documents/ks4/split_by_shank.py \
#       --sessions-json ~/Documents/ks4/learning_session_selection.json \
#       --processed-data-root "/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Processed data" \
#       --output-dir "/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/shank_split" \
#       --merge-manifests
#
#SBATCH -J shank_split
#SBATCH -o Documents/ks4/logs_shank_split/shank_split-%A_%a.out
#SBATCH -e Documents/ks4/logs_shank_split/shank_split-%A_%a.err
#SBATCH -N 1
#SBATCH -t 0-03:59
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=1-38%20

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these for your setup
# ═══════════════════════════════════════════════════════════════════════

# Path to the session selection JSON (copy from local repo or put on /ceph)
SESSIONS_JSON="$HOME/Documents/ks4/learning_session_selection.json"

# Root of processed data on /ceph (Linux equivalent of X:\...\Processed data)
DATA_ROOT="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Processed data"

# Output directory for shank-split binaries (on /ceph — fast I/O)
OUTPUT_DIR="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/shank_split"

# Chunk size in seconds (controls memory; 300s is fine with 32G RAM)
CHUNK_SECONDS=300

# Directory containing split_by_shank.py
SCRIPT_DIR="$HOME/Documents/ks4"

# Conda environment (kilosort4 env has numpy + scipy)
CONDA_ENV="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/conda_envs/kilosort4"

# ═══════════════════════════════════════════════════════════════════════

# Load modules (this is why 'conda' doesn't work without it!)
module load miniconda/23.10.0

# Source conda shell functions so 'conda activate' works
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
conda activate "${CONDA_ENV}"

# Create log directory
mkdir -p ~/Documents/ks4/logs_shank_split

# Print job info
echo "═══════════════════════════════════════════════════════════"
echo "SLURM Job ${SLURM_JOB_ID}, Array Task ${SLURM_ARRAY_TASK_ID} / 38"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Sessions JSON: ${SESSIONS_JSON}"
echo "Output dir: ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════"

# Verify numpy/scipy are available
python -c "import numpy; import scipy; print(f'numpy {numpy.__version__}, scipy {scipy.__version__}')"
if [ $? -ne 0 ]; then
    echo "ERROR: numpy or scipy not found in conda env. Install with:"
    echo "  conda install -n kilosort4 numpy scipy"
    exit 1
fi

# Run the split for THIS session only (1-based index from SLURM_ARRAY_TASK_ID)
python "${SCRIPT_DIR}/split_by_shank.py" \
    --sessions-json "${SESSIONS_JSON}" \
    --processed-data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --chunk-seconds ${CHUNK_SECONDS} \
    --session-index ${SLURM_ARRAY_TASK_ID}

EXIT_CODE=$?

echo "═══════════════════════════════════════════════════════════"
echo "Finished with exit code ${EXIT_CODE}"
echo "═══════════════════════════════════════════════════════════"

exit ${EXIT_CODE}
