#!/usr/bin/env bash
#
# run_split_shank_retry.bash — Re-run the 5 OOM-killed shank-split sessions
#
# Tasks 9, 24, 26, 29, 32 were OOM-killed with 16G + 300s chunks.
# Fix: more memory (32G) + smaller chunks (120s) + exclude enc1-node4/enc1-node9.
#
# The corrupted output has already been deleted, so --skip-existing
# will correctly re-process these 5 while skipping the 33 good ones.
#
# Submit:
#   sbatch run_split_shank_retry.bash
#
#SBATCH -J shank_retry
#SBATCH -o Documents/ks4/logs_shank_split/shank_retry-%A_%a.out
#SBATCH -e Documents/ks4/logs_shank_split/shank_retry-%A_%a.err
#SBATCH -N 1
#SBATCH -t 0-05:59
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=9,24,26,29,32
#SBATCH --exclude=enc1-node4,enc1-node9

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — same as run_split_shank.bash
# ═══════════════════════════════════════════════════════════════════════

SESSIONS_JSON="$HOME/Documents/ks4/learning_session_selection.json"
DATA_ROOT="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/Processed data"
OUTPUT_DIR="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/concat_sort/shank_split"
CHUNK_SECONDS=120
SCRIPT_DIR="$HOME/Documents/ks4"
CONDA_ENV="/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/conda_envs/kilosort4"

# ═══════════════════════════════════════════════════════════════════════

module load miniconda/23.10.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

mkdir -p ~/Documents/ks4/logs_shank_split

echo "═══════════════════════════════════════════════════════════"
echo "RETRY: SLURM Job ${SLURM_JOB_ID}, Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname), Memory: 32G, Chunks: ${CHUNK_SECONDS}s"
echo "═══════════════════════════════════════════════════════════"

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
