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
# RUNTIME: ~4-6h on ~128 gpu_lowp cores (the default fast path below), or ~12h
# on 32 cpu cores (the fallback). n_trials=800 (the VALIDATED point-recovery
# config). The run is RESUMABLE: the harness checkpoints recovery_results.json
# after every completed stage, so a timeout / preemption is never a total loss.
#
# The harness parallelises over REPLICATE PROCESSES, so BLAS is pinned to ONE
# thread per process (oversubscription would kill the Pool scaling). CPU-only:
# NO GPU is requested even on gpu_lowp.
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
# Partition / cpus / walltime are overridable AT SUBMIT TIME with sbatch CLI flags,
# which take PRECEDENCE over the #SBATCH defaults below (see CPU FALLBACK) -- no edit.
#
# ─────────────────────────────────────────────────────────────────────────────
# CPU FALLBACK (policy-clean, no preemption):
#   gpu_lowp (the DEFAULT below) is LOW-PRIORITY / PREEMPTIBLE. If the job is
#   preempted, just RESUBMIT THE SAME sbatch into the SAME --out-dir (OUTDIR) and
#   it RESUMES from the last checkpoint (no work lost).
#
#   To instead run policy-clean on the dedicated cpu partition, SUBMIT WITH CLI
#   OVERRIDES (these BEAT the #SBATCH defaults below; no file edit needed):
#       sbatch -p cpu --cpus-per-task=32 -t 0-18:00 cluster_recovery_harness.sbatch
#       (~12h overnight on 32 cores; CPUS below auto-follows SLURM_CPUS_PER_TASK)
# ─────────────────────────────────────────────────────────────────────────────
#
#SBATCH -J b8_recovery
#SBATCH -o slurm-b8_recovery-%A.out
#SBATCH -e slurm-b8_recovery-%A.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=16G
#SBATCH -t 0-08:00
#SBATCH -p gpu_lowp
# ^^^ DEFAULT = the FAST PATH: gpu_lowp, 128 cores, ~4-6h. CPU-only (no --gres=gpu).
#     gpu_lowp is low-priority/preemptible -> if preempted, resubmit the SAME
#     sbatch into the SAME OUTDIR and it RESUMES (checkpointing). For a
#     policy-clean run see the CPU FALLBACK block above.

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

# OUTDIR: where recovery_results.json + fig_b8_F6_recovery.png land. RESUBMIT
#         into the SAME OUTDIR to resume after a timeout / preemption.
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

# CPUS resolves from the SLURM allocation -- the #SBATCH --cpus-per-task above, or a
# `sbatch --cpus-per-task=N` CLI override (both set SLURM_CPUS_PER_TASK).
CPUS="${SLURM_CPUS_PER_TASK:-128}"

echo "════════════════════════════════════════════════════════════"
echo "SLURM job ${SLURM_JOB_ID:-?} on $(hostname)  --  B8 recovery harness"
echo "  CPUs   : ${CPUS}"
echo "  PART   : ${SLURM_JOB_PARTITION:-?}"
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

# ── Full-config run (n_rep_point=100, n_rep_confusion=50, n_trials=800 [VALIDATED],
#    n_restarts=4, bootstrap=500). --cpus = the allocation. RESUMABLE: resubmit
#    into the SAME OUTDIR to continue after a timeout / preemption. ──
python -u "${HARNESS}" \
    --out-dir "${OUTDIR}" \
    --cpus "${CPUS}" \
    --n-rep-point 100 \
    --n-rep-confusion 50 \
    --n-trials 800 \
    --n-restarts 4 \
    --bootstrap 500 \
    --seed 0

echo "DONE -- results + F6 figure in ${OUTDIR}"
