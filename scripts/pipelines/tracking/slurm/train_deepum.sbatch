#!/usr/bin/env bash
#
# train_deepum.sbatch -- DeepUnitMatch fine-tuning on BG_046 (plan Task 9).
# One MODE per submission: clip_warm_frozen | clip_warm_unfrozen | ae | clip_scratch
#
# Paths + activation mirror the real DeepUM A100 run (job 3026330,
# run_deepunitmatch_cluster.sh). Unlike that CPU-only inference run, TRAINING
# requests a GPU (--gres). The ceph `unitmatch` env already has a CUDA torch
# (2.12.0+cu130), so tools/check_gpu_torch.py should print cuda_available True
# on an a100 node.
#
# Stage the train_*.py + train_deepum_common.py + tools/ into UM_DIR (next to
# run_deepunitmatch_all.py + UnitMatchPy), then from the cluster login node:
#   sbatch train_deepum.sbatch clip_warm_frozen
# (or `export DEEPUM_TRACK=<dir>` if you stage them elsewhere.)
#
#SBATCH -J deepum_ft
#SBATCH -o /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/unit_match/logs/deepum_ft-%A.out
#SBATCH -e /ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect/wEPhys/BG_046/unit_match/logs/deepum_ft-%A.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-06:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
# (training needs a GPU but not the a100's huge RAM; the `gpu` partition usually
#  has free nodes when `a100` is jammed. Override at submit time if you prefer:
#  sbatch -p a100 slurm/train_deepum.sbatch <mode>)

set -e

CEPH=/ceph/mrsic_flogel/public/projects/BeJG_20230130_VisDetect
UM_DIR=${CEPH}/wEPhys/BG_046/unit_match
CONDA_ENV=${CEPH}/conda_envs/unitmatch
TRAIN_ROOT=${UM_DIR}/UnitMatchPy/DeepUnitMatch/processed_waveforms
OUT_BASE=${UM_DIR}/output/BG_046_finetune
# Where the train_*.py + train_deepum_common.py + tools/ are staged.
TRACK="${DEEPUM_TRACK:-${UM_DIR}}"

# Activate the ceph conda env WITHOUT the module system (same as job 3026330).
export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "════════════════════════════════════════════════════════════"
echo "SLURM job ${SLURM_JOB_ID:-?} on $(hostname)"
echo "Env: ${CONDA_ENV}   TRACK: ${TRACK}"
echo "════════════════════════════════════════════════════════════"

# Pre-flight: fail loudly before burning GPU time.
[ -d "${TRAIN_ROOT}" ] || { echo "MISSING TRAIN_ROOT: ${TRAIN_ROOT}"; exit 3; }
[ -f "${TRACK}/train_deepum_clip.py" ] || { echo "scripts not in TRACK=${TRACK}"; exit 3; }
GPU_CHECK="${TRACK}/tools/check_gpu_torch.py"; [ -f "${GPU_CHECK}" ] || GPU_CHECK="${TRACK}/check_gpu_torch.py"
python "${GPU_CHECK}"   # must print cuda_available True

MODE="${1:-clip_warm_frozen}"
case "${MODE}" in
  clip_warm_frozen)
    python -u "${TRACK}/train_deepum_clip.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/rung1_warm_frozen" --init shipped --freeze fcblock ;;
  clip_warm_unfrozen)
    python -u "${TRACK}/train_deepum_clip.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/rung2_warm_unfrozen" --init shipped --freeze none ;;
  ae)
    python -u "${TRACK}/train_deepum_ae.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/ae" ;;
  clip_scratch)
    AE_CKPT=$(ls -t "${OUT_BASE}/ae"/ae_epoch_*.pt | head -1)
    python -u "${TRACK}/train_deepum_clip.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/rung3_scratch" --init "${AE_CKPT}" --freeze fcblock ;;
  *) echo "unknown MODE ${MODE}"; exit 2 ;;
esac
echo "done ${MODE}"
