#!/bin/bash
#SBATCH --job-name=deepum_ft
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=deepum_ft_%j.out

# DeepUnitMatch fine-tuning launcher (plan Task 8). One MODE per submission:
#   clip_warm_frozen | clip_warm_unfrozen | ae | clip_scratch
# Paths marked CONFIRM are best-known defaults from
# memory/neuron_tracking_may2026.md; validate against the real ceph layout
# (cross-check the prior UM/DeepUM run scripts for jobs 3026330 / 3025315)
# before the first real run. Task 0 (tools/check_gpu_torch.py) must pass first.

set -euo pipefail
echo "SLURM job $SLURM_JOB_ID on $(hostname)"

# --- ceph-conda activation (mirrors KS4/UM run scripts) ---
CONDA_ENV="/ceph/.../conda_envs/unitmatch"        # CONFIRM
export PATH="${CONDA_ENV}/bin:${PATH}"

# --- repo + data ---
REPO="/ceph/.../vis_detect_analysis_Sep2025"      # CONFIRM (feature/deepum-finetune checkout)
TRACK="${REPO}/scripts/pipelines/tracking"
TRAIN_ROOT="/ceph/.../UnitMatchPy/DeepUnitMatch/processed_waveforms"   # CONFIRM
OUT_BASE="${REPO}/data/unit_match/output/BG_046_finetune"

python "${TRACK}/tools/check_gpu_torch.py"   # fail fast if CUDA missing

MODE="${1:-clip_warm_frozen}"   # clip_warm_frozen | clip_warm_unfrozen | ae | clip_scratch

case "${MODE}" in
  clip_warm_frozen)
    python "${TRACK}/train_deepum_clip.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/rung1_warm_frozen" --init shipped --freeze fcblock ;;
  clip_warm_unfrozen)
    python "${TRACK}/train_deepum_clip.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/rung2_warm_unfrozen" --init shipped --freeze none ;;
  ae)
    python "${TRACK}/train_deepum_ae.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/ae" ;;
  clip_scratch)
    AE_CKPT=$(ls -t "${OUT_BASE}/ae"/ae_epoch_*.pt | head -1)
    python "${TRACK}/train_deepum_clip.py" --train-root "${TRAIN_ROOT}" \
      --out-dir "${OUT_BASE}/rung3_scratch" --init "${AE_CKPT}" --freeze fcblock ;;
  *) echo "unknown MODE ${MODE}"; exit 2 ;;
esac
echo "done ${MODE}"
