#!/usr/bin/env bash
#
# cluster_recovery_harness.sbatch -- B8 Phase 2 FULL-POWER recovery verdict.
#
# Runs scripts/analysis/decision_latents/cluster_recovery_harness.py at full
# config on a CPU node: point recovery (n_rep>=100, parametric-bootstrap
# coverage), the 3x3 which-dial-varies confusion matrix (n_rep>=50), the
# shrunk + Hessian vetoes, and the recovery_gate -> per (dial x regime)
# latent_trust verdict. Writes recovery_results.json + fig_b8_F6_recovery.png.
#
# The harness parallelises over REPLICATE PROCESSES, so BLAS is pinned to ONE
# thread per process (oversubscription would kill the Pool scaling).
#
# ─────────────────────────────────────────────────────────────────────────────
# EXACT RUN RECIPE (copy-paste, edit the three vars):
#
#   CEPH=/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect
#   # 1) clone the B8 Phase-2 branch to a ceph working dir:
#   git clone -b feature/B8-phase2-generative-latents \
#       git@github.com:CherryGarcia-code/vis_detect_analysis_Sep2025.git \
#       "$CEPH/work/b8_recovery_repo"
#   # 2) set the three knobs and submit:
#   export B8_REPO="$CEPH/work/b8_recovery_repo"
#   export CONDA_ENV="$CEPH/conda_envs/unitmatch"          # any env w/ numpy+scipy+matplotlib
#   export OUTDIR="$CEPH/work/b8_recovery_out/$(date +%Y%m%d_%H%M)"
#   cd "$B8_REPO/scripts/analysis/decision_latents"
#   sbatch cluster_recovery_harness.sbatch
#
# (Override the partition below to your site's CPU partition if it is not `cpu`.)
# ─────────────────────────────────────────────────────────────────────────────
#
#SBATCH -J b8_recovery
#SBATCH -o slurm-b8_recovery-%A.out
#SBATCH -e slurm-b8_recovery-%A.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 0-08:00
#SBATCH -p cpu
# ^^^ CHANGE `-p cpu` to your site's CPU partition (e.g. `-p fast`, `-p compute`)
#     if `cpu` does not exist. The job is CPU-only; no GPU is needed.

set -euo pipefail

CEPH="${CEPH:-/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect}"

# ── Parameterised inputs (override via env before sbatch; see recipe above) ──
# B8_REPO: the repo clone whose src/ holds the visdetect package + this script.
#          Default = the repo root inferred from THIS script's location, so an
#          in-place submit (cd into the clone, sbatch) works without setting it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"      # scripts/analysis/decision_latents -> repo root
B8_REPO="${B8_REPO:-${DEFAULT_REPO}}"

# CONDA_ENV: any conda env with numpy + scipy + matplotlib (the unitmatch env
#            on ceph has them). Override if you keep a dedicated analysis env.
CONDA_ENV="${CONDA_ENV:-${CEPH}/conda_envs/unitmatch}"

# OUTDIR: where recovery_results.json + fig_b8_F6_recovery.png land.
OUTDIR="${OUTDIR:-${B8_REPO}/FIGURES/decision_latents/BG_046/recovery_cluster}"

HARNESS="${B8_REPO}/scripts/analysis/decision_latents/cluster_recovery_harness.py"

# ── Activate the ceph conda env WITHOUT the module system (project pattern,
#    same as run_unitmatch_subject.sbatch). ──
export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# ── visdetect importable WITHOUT install (editable install may point elsewhere). ──
export PYTHONPATH="${B8_REPO}/src:${PYTHONPATH:-}"

# ── BLAS = 1 thread: the harness is PROCESS-parallel (Pool over replicates), so
#    each worker must keep BLAS single-threaded or the cores oversubscribe. ──
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg

CPUS="${SLURM_CPUS_PER_TASK:-16}"

echo "════════════════════════════════════════════════════════════"
echo "SLURM job ${SLURM_JOB_ID:-?} on $(hostname)  --  B8 recovery harness"
echo "  CPUs   : ${CPUS}"
echo "  B8_REPO: ${B8_REPO}"
echo "  ENV    : ${CONDA_ENV}"
echo "  OUTDIR : ${OUTDIR}"
echo "  PYTHONPATH: ${PYTHONPATH}"
echo "════════════════════════════════════════════════════════════"

# ── Pre-flight: fail loudly before the scheduler burns the allocation. ──
[ -f "${HARNESS}" ] || { echo "MISSING harness: ${HARNESS}"; exit 3; }
mkdir -p "${OUTDIR}"

python --version
python -c "import numpy, scipy; print('core imports OK; numpy', numpy.__version__, 'scipy', scipy.__version__)"
python -c "import matplotlib" 2>/dev/null && echo "matplotlib OK (F6 figure will be drawn)" || echo "WARNING: matplotlib missing -- JSON will still be written; F6 skipped (render locally from recovery_results.json)"
python -c "from visdetect.analysis import decision_latents_generative as d; print('visdetect OK:', d.__file__)"
echo "════════════════════════════════════════════════════════════"

# ── Full-config run (defaults: n_rep_point=100, n_rep_confusion=50,
#    n_trials=2000, n_restarts=4, bootstrap=500). --cpus = the allocation. ──
python -u "${HARNESS}" \
    --out-dir "${OUTDIR}" \
    --cpus "${CPUS}" \
    --n-rep-point 100 \
    --n-rep-confusion 50 \
    --n-trials 2000 \
    --n-restarts 4 \
    --bootstrap 500 \
    --seed 0

echo "DONE -- results + F6 figure in ${OUTDIR}"
