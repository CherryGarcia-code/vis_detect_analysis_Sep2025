#!/usr/bin/env bash
#
# run_unitmatch_subject.sbatch -- UnitMatch 3.2.9 on ALL sessions of ONE
# subject, in a single batch. Parametrized clone of BG_046's
# run_unitmatch_cluster.sh for BG_031 / BG_038 / BG_039.
#
# UnitMatch is CPU-only; the only constraint is RAM. extract_metric_scores
# builds (n_timepoints, N, N) float64 arrays (~23 * N^2 * 8 bytes, several
# live at once). For BG_046 (~6679 units) peak was ~180-230 GB -> needs a
# high-memory `a100` node (510 GB). The new subjects have fewer sessions, but
# size the job from the printed n_units (see the pre-flight line in the log:
# "n_units=...  ((t,N,N) array ~XX GB)") and adjust --mem if needed.
#
# Submit from the subject's unit_match dir so the slurm logs land beside it:
#   cd /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_031/unit_match
#   sbatch /path/to/run_unitmatch_subject.sbatch BG_031
#
#SBATCH -J unitmatch
#SBATCH -o slurm-unitmatch-%A.out
#SBATCH -e slurm-unitmatch-%A.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-12:00
#SBATCH -p a100
#SBATCH --mem=400G
#
# --- ALTERNATIVE: the `gpu` partition (220 GB nodes -- starts sooner, tighter
#     on RAM; fine for the smaller subjects). To use it, delete the two
#     `#SBATCH` lines above (-p a100 / --mem=400G) and remove ONE `#` from each
#     line below: -------------------------------------------------------------
##SBATCH -p gpu
##SBATCH --mem=215G
# If the scheduler rejects a CPU-only job on a GPU partition, also un-comment:
##SBATCH --gres=gpu:1
# ----------------------------------------------------------------------------

set -e

SUBJECT="${1:?usage: sbatch run_unitmatch_subject.sbatch <SUBJECT>  (e.g. BG_031)}"

CEPH=/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect
UM_DIR=${CEPH}/wEPhys/${SUBJECT}/unit_match
CONDA_ENV=${CEPH}/conda_envs/unitmatch

# Activate the ceph conda env WITHOUT the module system (same pattern as
# run_unitmatch_cluster.sh / the concat-sort KS4 retry scripts).
export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# Keep BLAS threading bounded to the allocated CPUs.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "════════════════════════════════════════════════════════════"
echo "SLURM job ${SLURM_JOB_ID:-?} on $(hostname)  --  subject ${SUBJECT}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}   Mem: ${SLURM_MEM_PER_NODE} MB"
echo "Env:  ${CONDA_ENV}"
echo "UM:   ${UM_DIR}"
echo "════════════════════════════════════════════════════════════"

# Pre-flight: fail loudly before the scheduler burns the allocation.
[ -d "${UM_DIR}/input" ]            || { echo "MISSING input dir: ${UM_DIR}/input"; exit 3; }
[ -f "${UM_DIR}/run_unitmatch_all.py" ] || { echo "MISSING ${UM_DIR}/run_unitmatch_all.py"; exit 3; }
mkdir -p "${UM_DIR}/logs"

python --version
python -c "import UnitMatchPy, numpy, scipy, pandas; print('UnitMatchPy import OK; numpy', numpy.__version__)"

N_SESS=$(find "${UM_DIR}/input" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Sessions found in input: ${N_SESS}"
echo "════════════════════════════════════════════════════════════"

# Single batch over every session (batch-size >= n_sessions => one batch).
python -u "${UM_DIR}/run_unitmatch_all.py" \
    --input   "${UM_DIR}/input" \
    --out-dir "${UM_DIR}/output/all_sessions" \
    --batch-size "${N_SESS}" --overlap 0

echo "DONE -- output in ${UM_DIR}/output/all_sessions"
